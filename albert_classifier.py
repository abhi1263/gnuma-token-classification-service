import os
import uuid
import json
import transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, \
    DataCollatorForTokenClassification, TrainerCallback
from datasets import load_dataset, load_metric
import numpy as np
from messages import AMQPProducer

task = "ner"  # NLP task that we intend to perform
model_checkpoint = "albert-base-v2"  # huggingface model name and version
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)  # using the associated pretrained tokenizer

# Initializing the dataset
datasets = load_dataset("conll2003")
label_list = datasets["train"].features[f"{task}_tags"].feature.names


# Data pre-processing
def tokenize_and_align_labels(examples):
    label_all_tokens = True
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)

# Definining the compute metrics for predictions
metric = load_metric("seqeval")


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def start_training(hyperparameters, model_id):
    """ Fine tuning the model """
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))

    dataset_ids = hyperparameters['train_ids']
    # dataset = fetch_dataset(dataset_ids)

    os.makedirs("./albert/" + str(model_id), mode=0o777)

    # Initializing the Training arguments
    args = TrainingArguments(
        output_dir="./albert/" + str(model_id),
        learning_rate=float(hyperparameters['learning_rate']),
        num_train_epochs=float(hyperparameters['num_train_epochs']),
        per_device_train_batch_size=int(hyperparameters['per_device_train_batch_size']),
        per_device_eval_batch_size=int(hyperparameters['per_device_eval_batch_size']),
        warmup_steps=int(hyperparameters['warmup_steps']),
        weight_decay=float(hyperparameters['weight_decay']),
        adam_beta1=float(hyperparameters['adam_beta1']),
        adam_beta2=float(hyperparameters['adam_beta2']),
        adam_epsilon=float(hyperparameters['adam_epsilon']),
        logging_steps=10,
        save_steps=100,
        save_strategy="steps",
        logging_dir=f"./albert/{model_id}/logs/",
        logging_strategy="steps",
        log_level="debug",
    )

    """Data collator will batch the processed examples together while applying padding to make them all the same size 
    """
    data_collator = DataCollatorForTokenClassification(tokenizer)

    """ Initializing the trainer """
    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[CustomCallback],
    )

    # Saving all the parameters of the models in a JSON file
    model_metadata = {
        'model_id': str(model_id),
        'dataset_id': hyperparameters['dataset_id'],
        'train_ids': hyperparameters['train_ids'],
        'val_ids': hyperparameters['val_ids'],
        'training_status': "incomplete",
        'interrupted': False,
        "learning_rate": hyperparameters['learning_rate'],
        "num_train_epochs": hyperparameters['num_train_epochs'],
        "per_device_train_batch_size": hyperparameters['per_device_train_batch_size'],
        "per_device_eval_batch_size": hyperparameters['per_device_eval_batch_size'],
        "warmup_steps": hyperparameters['warmup_steps'],
        "weight_decay": hyperparameters['weight_decay'],
        "adam_beta1": hyperparameters['adam_beta1'],
        "adam_beta2": hyperparameters['adam_beta2'],
        "adam_epsilon": hyperparameters['adam_epsilon'],
        "logging_steps": 10,
        "save_steps": 100,
        "save_strategy": "steps",
        "logging_dir": f"./albert/{model_id}/logs/",
        "logging_strategy": "steps",
        "log_level": "debug",

    }
    with open(f'./albert/{model_id}/model_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(model_metadata, f, ensure_ascii=False, indent=4)

   # try:
    # Starting the training
    trainer.train()
    trainer.save_model(f"./albert/{model_id}/")
    #message_broker = AMQPProducer()
    #message_broker.send_progress_update('albert', model_id, 0, 0, 0, True, metrics={})
    """except:
        print("Training failed")
        message_broker = AMQPProducer()
        message_broker.send_training_failed('albert', model_id, )"""


def resume_training(model_id):
    incomplete_model = './albert/' + model_id
    model = AutoModelForTokenClassification.from_pretrained(incomplete_model, num_labels=len(label_list))

    # Initializing the Hyperparameters
    with open(f'./albert/{model_id}/model_metadata.json') as json_file:
        data = json.load(json_file)
        args = TrainingArguments(
            output_dir="./albert/" + str(model_id) + "/",
            learning_rate=float(data['learning_rate']),
            num_train_epochs=float(data['num_train_epochs']),
            per_device_train_batch_size=int(data['per_device_train_batch_size']),
            per_device_eval_batch_size=int(data['per_device_eval_batch_size']),
            warmup_steps=int(data['warmup_steps']),
            weight_decay=float(data['weight_decay']),
            logging_steps=int(data['logging_steps']),
            #evaluation_strategy=str(data['evaluation_strategy']),
        )

    """Data collator will batch the processed examples together while applying padding to make them all the same size 
        """
    data_collator = DataCollatorForTokenClassification(tokenizer)

    """ Initializing the trainer """
    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[CustomCallback]
    )
    try:
        if os.path.isdir("./albert/" + str(model_id) + "/checkpoint-100"):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
        #message_broker = AMQPProducer()
        #message_broker.send_progress_update('albert', model_id, 0, 0, 0, True, metrics={})
    except:
        print("Training failed")
        message_broker = AMQPProducer()
        message_broker.send_training_failed('albert', model_id)


class CustomCallback(TrainerCallback):
    """Custom callback which sends progress update message and interrupts training"""

    def on_step_end(self, args, state, control, logs=None, **kwargs):
        model_id = args.output_dir.split('/')[-1]
        with open(args.output_dir + '/model_metadata.json') as f:
            data = json.load(f)
            if data['interrupted']:
                print('Interrupt requested. ABorting!')
                control.should_training_stop = True
                control.should_save = True
                control.should_evaluate = False
                message_broker = AMQPProducer()
                message_broker.send_interrupt('albert', model_id)

    def on_log(self, args, state, control, logs=None, **kwargs):
        classifier = 'albert'
        model_id = args.output_dir.split('/')[-1]
        current_step = state.global_step
        total_step = state.max_steps
        epoch = state.epoch

        if 'loss' in logs and 'learning_rate' in logs:
            loss = logs['loss']
            learning_rate = logs['learning_rate']
            metrics = [
                {
                    "key": "loss",
                    "value": loss,
                },
                {
                    "key": "learning_rate",
                    "value": learning_rate
                }
            ]
            message_broker = AMQPProducer()
            message_broker.send_progress_update(classifier, str(model_id), current_step, total_step, epoch, False,
                                                metrics)
        elif 'eval_loss' in logs and 'eval_precision' in logs and 'eval_recall' in logs and 'eval_f1' in logs:
            metrics = [
                {
                    "key": "eval_loss",
                    "value": logs['eval_loss'],
                },
                {
                    "key": "eval_precision",
                    "value": logs['eval_precision']
                },
                {
                    "key": "eval_recall",
                    "value": logs['eval_recall'],
                },
                {
                    "key": "eval_f1",
                    "value": logs['eval_f1']
                }
            ]
            message_broker = AMQPProducer()
            message_broker.send_progress_update(classifier, str(model_id), current_step, total_step, epoch, False,
                                                metrics)
