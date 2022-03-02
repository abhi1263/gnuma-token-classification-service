import pika
import json
from configparser import ConfigParser
from threading import Thread
import time


class AMQPProducer:
    def __init__(self):
        config: ConfigParser = ConfigParser()
        config.read('config.ini')

        # Flask server config details
        #server_host = config['flask_server']['host']
        #server_port = config['flask_server']['port']
        server_host = 'f936-95-90-218-215.ngrok.io'
        server_port = '443'

        # RabbitMQ config details
        rabbit_mq_host = config['rabbitmq']['host']
        rabbit_mq_port = config['rabbitmq']['port']
        rabbit_mq_user = config['rabbitmq']['user']
        rabbit_mq_pass = config['rabbitmq']['pass']

        self._credentials = pika.PlainCredentials(rabbit_mq_user, rabbit_mq_pass)
        self._connection_params = pika.ConnectionParameters(host=rabbit_mq_host, port=rabbit_mq_port,
                                                            credentials=self._credentials)
        self.server_host = server_host
        self.server_port = server_port

    def send_startup_message(self):
        connection = pika.BlockingConnection(self._connection_params)
        channel = connection.channel()

        message_for_albert = {
            "address": f"https://{self.server_host}:{self.server_port}/albert",
            "classifier_name": "albert",
            "hyper_parameters": [
                {
                    "name": "num_train_epochs",
                    "type": "DOUBLE",
                    "value_list": None,
                    "optional": False,
                    "default": 3.0,
                    "description": "Total number of training epochs to perform (if not an integer, will perform the decimal part percents of the last epoch before stopping training).",
                    "lower_bound": 0,
                    "upper_bound": 10,

                },
                {
                    "name": "learning_rate",
                    "type": "DOUBLE",
                    "value_list": None,
                    "optional": False,
                    "default": 5e-5,
                    "description": "The initial learning rate for AdamW optimizer.",
                    "lower_bound": 5e-10,
                    "upper_bound": 10,

                },
                {
                    "name": "per_device_train_batch_size",
                    "type": "INTEGER",
                    "value_list": None,
                    "optional": False,
                    "default": 8,
                    "description": "The batch size per GPU/TPU core/CPU for training.",
                    "lower_bound": 0,
                    "upper_bound": 200,

                },
                {
                    "name": "per_device_eval_batch_size",
                    "type": "INTEGER",
                    "value_list": None,
                    "optional": False,
                    "default": 8,
                    "description": "The batch size per GPU/TPU core/CPU for evaluation.",
                    "lower_bound": 0,
                    "upper_bound": 200,

                },
                {
                    "name": "warmup_steps",
                    "type": "INTEGER",
                    "value_list": None,
                    "optional": False,
                    "default": 0,
                    "description": "Number of steps used for a linear warmup from 0 to learning_rate.",
                    "lower_bound": 0,
                    "upper_bound": 1000000,

                },
                {
                    "name": "weight_decay",
                    "type": "DOUBLE",
                    "value_list": None,
                    "optional": False,
                    "default": 0,
                    "description": "The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in AdamW optimizer.",
                    "lower_bound": 0.0,
                    "upper_bound": 10000000.0,

                },
                {
                    "name": "adam_beta1",
                    "type": "DOUBLE",
                    "value_list": None,
                    "optional": False,
                    "default": 0.9,
                    "description": "The beta1 hyperparameter for the AdamW optimizer.",
                    "lower_bound": -999.0,
                    "upper_bound": 1000.0,

                },
                {
                    "name": "adam_beta2",
                    "type": "DOUBLE",
                    "value_list": None,
                    "optional": False,
                    "default": 0.999,
                    "description": "The beta2 hyperparameter for the AdamW optimizer.",
                    "lower_bound": -999.0,
                    "upper_bound": 1000.0,

                },
                {
                    "name": "adam_epsilon",
                    "type": "DOUBLE",
                    "value_list": None,
                    "optional": False,
                    "default": 1e-8,
                    "description": "The epsilon hyperparameter for the AdamW optimizer.",
                    "lower_bound": 1e-12,
                    "upper_bound": 1000.0,

                },
            ]
        }
        message_for_convbert = {
            "address": f"https://{self.server_host}:{self.server_port}/convbert",
            "classifier_name": "convbert",
            "hyper_parameters": [
                {
                    "name": "num_train_epochs",
                    "type": "DOUBLE",
                    "value_list": None,
                    "optional": False,
                    "default": 3.0,
                    "description": "Total number of training epochs to perform (if not an integer, will perform the decimal part percents of the last epoch before stopping training).",
                    "lower_bound": 0,
                    "upper_bound": 10,

                },
                {
                    "name": "learning_rate",
                    "type": "DOUBLE",
                    "value_list": None,
                    "optional": False,
                    "default": 5e-5,
                    "description": "The initial learning rate for AdamW optimizer.",
                    "lower_bound": 5e-10,
                    "upper_bound": 10,

                },
                {
                    "name": "per_device_train_batch_size",
                    "type": "INTEGER",
                    "value_list": None,
                    "optional": False,
                    "default": 8,
                    "description": "The batch size per GPU/TPU core/CPU for training.",
                    "lower_bound": 0,
                    "upper_bound": 200,

                },
                {
                    "name": "per_device_eval_batch_size",
                    "type": "INTEGER",
                    "value_list": None,
                    "optional": False,
                    "default": 8,
                    "description": "The batch size per GPU/TPU core/CPU for evaluation.",
                    "lower_bound": 0,
                    "upper_bound": 200,

                },
                {
                    "name": "warmup_steps",
                    "type": "INTEGER",
                    "value_list": None,
                    "optional": False,
                    "default": 0,
                    "description": "Number of steps used for a linear warmup from 0 to learning_rate.",
                    "lower_bound": 0,
                    "upper_bound": 1000000,

                },
                {
                    "name": "weight_decay",
                    "type": "DOUBLE",
                    "value_list": None,
                    "optional": False,
                    "default": 0,
                    "description": "The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in AdamW optimizer.",
                    "lower_bound": 0.0,
                    "upper_bound": 10000000.0,

                },
                {
                    "name": "adam_beta1",
                    "type": "DOUBLE",
                    "value_list": None,
                    "optional": False,
                    "default": 0.9,
                    "description": "The beta1 hyperparameter for the AdamW optimizer.",
                    "lower_bound": -999.0,
                    "upper_bound": 1000.0,

                },
                {
                    "name": "adam_beta2",
                    "type": "DOUBLE",
                    "value_list": None,
                    "optional": False,
                    "default": 0.999,
                    "description": "The beta2 hyperparameter for the AdamW optimizer.",
                    "lower_bound": -999.0,
                    "upper_bound": 1000.0,

                },
                {
                    "name": "adam_epsilon",
                    "type": "DOUBLE",
                    "value_list": None,
                    "optional": False,
                    "default": 1e-8,
                    "description": "The epsilon hyperparameter for the AdamW optimizer.",
                    "lower_bound": 1e-12,
                    "upper_bound": 1000.0,

                },
            ]
        }

        message1 = json.dumps(message_for_albert)
        message2 = json.dumps(message_for_convbert)

        channel.basic_publish(exchange='GNUMAExchange',
                              routing_key='Classifier.Albert',
                              properties=pika.BasicProperties(
                                  headers={'event': 'ClassifierStart'}
                              ),
                              body=message1)
        channel.basic_publish(exchange='GNUMAExchange',
                              routing_key='Classifier.Convbert',
                              properties=pika.BasicProperties(
                                  headers={'event': 'ClassifierStart'}
                              ),
                              body=message2)
        connection.close()

    def send_progress_update(self, classifier, model_id, current_step, total_step, epoch, finished, metrics=[]):
        connection = pika.BlockingConnection(self._connection_params)
        channel = connection.channel()

        message = {
            "address": f"https://{self.server_host}:{self.server_port}/{classifier}",
            "model_id": model_id,
            "finished": finished,
            "current_step": current_step,
            "total_steps": total_step,
            "epoch": epoch,
            "metrics": metrics
        }

        message = json.dumps(message)
        channel.basic_publish(exchange='GNUMAExchange',
                              routing_key='Classifier.Albert',
                              properties=pika.BasicProperties(
                                  headers={'event': 'TrainingUpdate'}
                              ),
                              body=message)
        connection.close()

    def send_interrupt(self, classifier, model_id):
        connection = pika.BlockingConnection(self._connection_params)
        channel = connection.channel()

        message = {
            "address": f"https://{self.server_host}:{self.server_port}/{classifier}",
            "model_id": model_id,
            "paused": True,
        }

        message = json.dumps(message)
        channel.basic_publish(exchange='GNUMAExchange',
                              routing_key='Classifier.Albert',
                              properties=pika.BasicProperties(
                                  headers={'event': 'ClassifierInterrupt'}
                              ),
                              body=message)
        connection.close()


    def send_training_failed(self, classifier, model_id):
        connection = pika.BlockingConnection(self._connection_params)
        channel = connection.channel()

        message = {
            "address": f"https://{self.server_host}:{self.server_port}/{classifier}",
            "model_id": model_id,
            "error_message": "Classifier crashed!",
        }

        message = json.dumps(message)
        channel.basic_publish(exchange='GNUMAExchange',
                              routing_key='Classifier.Albert',
                              properties=pika.BasicProperties(
                                  headers={'event': 'ClassifierError'}
                              ),
                              body=message)
        connection.close()


class AMQPClassifierAlive(Thread):
    def __init__(self):
        super().__init__()
        config: ConfigParser = ConfigParser()
        config.read('config.ini')

        # Flask server config details
        server_host = config['flask_server']['host']
        server_port = config['flask_server']['port']

        # RabbitMQ config details
        rabbit_mq_host = config['rabbitmq']['host']
        rabbit_mq_port = config['rabbitmq']['port']
        rabbit_mq_user = config['rabbitmq']['user']
        rabbit_mq_pass = config['rabbitmq']['pass']

        self._credentials = pika.PlainCredentials(rabbit_mq_user, rabbit_mq_pass)
        self._connection_params = pika.ConnectionParameters(host=rabbit_mq_host, port=rabbit_mq_port,
                                                            credentials=self._credentials)
        self._interrupted = True
        self.server_host = server_host
        self.server_port = server_port

    def run(self):
        self._interrupted = False
        connection = pika.BlockingConnection(self._connection_params)
        channel = connection.channel()

        message1 = {
            "address": f"https://{self.server_host}:{self.server_port}/albert"
        }
        message2 = {
            "address": f"https://{self.server_host}:{self.server_port}/convbert"
        }

        message1 = json.dumps(message1)
        message2 = json.dumps(message2)

        while True:
            channel.basic_publish(exchange='GNUMAExchange',
                                  routing_key='Classifier.Albert',
                                  properties=pika.BasicProperties(
                                      headers={'event': 'ClassifierAlive'}
                                  ),
                                  body=message1)
            channel.basic_publish(exchange='GNUMAExchange',
                                  routing_key='Classifier.Convbert',
                                  properties=pika.BasicProperties(
                                      headers={'event': 'ClassifierAlive'}
                                  ),
                                  body=message2)
            time.sleep(10)

        connection.close()

    def stop(self):
        print("Stopping classifier-alive producer")
        self._interrupted = True
