from flask import Flask, jsonify
from configparser import ConfigParser
from flask_restful import Resource, Api, reqparse, request, abort
from resources import StartTraining, ResumeTraining, PauseTraining, ShowInfoOrDeleteModel, ListOfModels, InterruptTraining
from messages import AMQPProducer, AMQPClassifierAlive

if __name__ == '__main__':
    config: ConfigParser = ConfigParser()
    config.read('config.ini')

    # Flask server config details
    host = config['flask_server']['host']
    port = config['flask_server']['port']

    app = Flask(__name__)
    api = Api(app)

    api.add_resource(StartTraining, '/<string:classifier>/train')
    api.add_resource(ResumeTraining, '/<string:classifier>/continue/<string:model_id>')
    api.add_resource(PauseTraining, '/<string:classifier>/pause/<string:model_id>')
    api.add_resource(ShowInfoOrDeleteModel, '/<string:classifier>/models/<string:model_id>')
    api.add_resource(ListOfModels, '/<string:classifier>/models')
    api.add_resource(InterruptTraining, '/<string:classifier>/interrupt/<string:model_id>')

    rabbitmq_producer = AMQPProducer()
    rabbitmq_producer.send_startup_message()

    classifier_alive_producer = AMQPClassifierAlive()
    classifier_alive_producer.start()

    app.run(host=host, port=port)

    classifier_alive_producer.stop()
    classifier_alive_producer.join()
