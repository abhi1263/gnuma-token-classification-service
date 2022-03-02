import json, uuid, shutil, os
from flask_restful import Resource, Api, reqparse, request, abort
from flask import jsonify, Response
import albert_classifier, convbert_classifier
from redis_connection import redis_connection
from rq import Queue, cancel_job
from rq.registry import StartedJobRegistry
from messages import AMQPProducer

# Setting up a Redis queue and initialize the queue
queue = Queue(connection=redis_connection)


class StartTraining(Resource):
    def post(self, classifier):
        try:
            hyperparameters = {}

            json_data = request.get_json()
            hyperparameters = {
                "dataset_id": json_data['dataset_id'],
                "train_ids": list(json_data['train_ids']),
                "val_ids": list(json_data['val_ids']),
                "learning_rate": json_data['hyper_parameters']['learning_rate'],
                "num_train_epochs": json_data['hyper_parameters']['num_train_epochs'],
                "per_device_train_batch_size": json_data['hyper_parameters']['per_device_train_batch_size'],
                "per_device_eval_batch_size": json_data['hyper_parameters']['per_device_eval_batch_size'],
                "warmup_steps": json_data['hyper_parameters']['warmup_steps'],
                "weight_decay": json_data['hyper_parameters']['weight_decay'],
                "adam_beta1": json_data['hyper_parameters']['adam_beta1'],
                "adam_beta2": json_data['hyper_parameters']['adam_beta2'],
                "adam_epsilon": json_data['hyper_parameters']['adam_epsilon'],
            }

            if classifier == "albert":
                model_id = str(uuid.uuid4())
                job = queue.enqueue_call(func=albert_classifier.start_training, args=(hyperparameters, model_id),
                                         job_id=model_id)
                return jsonify({'model_id': f'{model_id}'})

            elif classifier == "convbert":
                model_id = str(uuid.uuid4())
                job = queue.enqueue_call(func=convbert_classifier.start_training, args=(hyperparameters, model_id),
                                         job_id=model_id)
                return jsonify({'model_id': f'{model_id}'})
            else:
                return Response('', status=400)
        except:
            return Response('', status=400)


class ResumeTraining(Resource):
    def post(self, classifier, model_id):
        if os.path.exists(f"./{classifier}/{model_id}"):
            hyperparameters = {}
            if classifier == "albert":
                job = queue.enqueue_call(func=albert_classifier.resume_training, args=(model_id,),
                                         job_id=model_id)
                return jsonify({'message': "Training has been resumed from last checkpoint"})

            elif classifier == "convbert":
                job = queue.enqueue_call(func=convbert_classifier.resume_training, args=(model_id,),
                                         job_id=model_id)
                return jsonify({'message': "Training has been resumed from last checkpoint"})
            else:
                return Response('', 400)
        else:
            return Response('', 400)


class PauseTraining(Resource):
    def put(self, classifier, model_id):
        registry = StartedJobRegistry('default', connection=redis_connection)
        running_job_ids = registry.get_job_ids()
        queued_job_ids = registry.get_queue().job_ids

        with open(f"./{classifier}/{model_id}/model_metadata.json", "r") as f:
            json_object = json.load(f)
            json_object['interrupted'] = True
        with open(f"./{classifier}/{model_id}/model_metadata.json", "w", encoding='utf-8') as f:
            json.dump(json_object, f, ensure_ascii=False, indent=4)

        """# If the training has already begun, then send an interrupt request
        if model_id in running_job_ids:
            with open(f"./{classifier}/{model_id}/model_metadata.json", "r") as f:
                json_object = json.load(f)
                json_object['interrupted'] = True
            with open(f"./{classifier}/{model_id}/model_metadata.json", "w", encoding='utf-8') as f:
                json.dump(json_object, f, ensure_ascii=False, indent=4)
        elif model_id in queued_job_ids:
            # If train task is still in queue then delete the task
            cancel_job(model_id)"""


class InterruptTraining(Resource):
    def delete(self, classifier, model_id):
        registry = StartedJobRegistry('default', connection=redis_connection)
        running_job_ids = registry.get_job_ids()
        queued_job_ids = registry.get_queue().job_ids

        if model_id in running_job_ids:
            with open(f"./{classifier}/{model_id}/model_metadata.json", "r") as f:
                json_object = json.load(f)
                json_object['interrupted'] = True
            with open(f"./{classifier}/{model_id}/model_metadata.json", "w", encoding='utf-8') as f:
                json.dump(json_object, f, ensure_ascii=False, indent=4)

        """# If the training has already begun, then send an interrupt request
        if model_id in running_job_ids:
            with open(f"./{classifier}/{model_id}/model_metadata.json", "r") as f:
                json_object = json.load(f)
                json_object['interrupted'] = True
            with open(f"./{classifier}/{model_id}/model_metadata.json", "w", encoding='utf-8') as f:
                json.dump(json_object, f, ensure_ascii=False, indent=4)
        elif model_id in queued_job_ids:
            #If train task is still in queue then delete the task
            #cancel_job(model_id)
            pass"""

        # Deleting the model folder
        if os.path.exists(f"./{classifier}/{model_id}"):
            shutil.rmtree(f"./{classifier}/{model_id}")


class ShowInfoOrDeleteModel(Resource):
    def delete(self, classifier, model_id):
        if os.path.exists(f"./{classifier}/{model_id}"):
            shutil.rmtree(f"./{classifier}/{model_id}")
            return Response("", status=200, )
        else:
            return Response("", status=404, )

    def get(self, classifier, model_id):
        response = {}
        try:
            with open(f'./{classifier}/{model_id}/model_metadata.json') as json_file:
                data = json.load(json_file)
                response.update({
                    'model_name': "albert",
                    'dataset_id': data['dataset_id'],
                    'train_ids': data['train_ids'],
                    'val_ids': data['val_ids'],
                    'label_mapping': {},
                    'model_id': data['model_id'],
                    'status': data['training_status'],
                    'hyper_parameters': {
                        "learning_rate": data['learning_rate'],
                        "num_train_epochs": data['num_train_epochs'],
                        "per_device_train_batch_size": data['per_device_train_batch_size'],
                        "per_device_eval_batch_size": data['per_device_eval_batch_size'],
                        "warmup_steps": data['warmup_steps'],
                        "weight_decay": data['weight_decay'],
                        "adam_beta1": data['adam_beta1'],
                        "adam_beta2": data['adam_beta2'],
                        "adam_epsilon": data['adam_epsilon'],
                    },
                })
            return jsonify(response)
        except:
            return Response("The response body goes here", status=404, )


class ListOfModels(Resource):
    def get(self, classifier):
        response = []
        try:
            for model in next(os.walk(f'./{classifier}'))[1]:
                i = 0
                with open(f'./albert/{model}/model_metadata.json') as json_file:
                    data = json.load(json_file)
                    response.append({
                        'model_id': data['model_id'],
                        'model_name': "albert",
                        'dataset_id': data['dataset_id'],
                        'status': data['training_status'],
                    })
                    i = i + 1

            return jsonify(response)
        except:
            return Response("", status=400, )
