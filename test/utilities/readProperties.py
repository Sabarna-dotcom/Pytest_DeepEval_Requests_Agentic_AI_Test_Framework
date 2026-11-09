import configparser
import os


config = configparser.RawConfigParser()
config_path = os.path.join(os.path.dirname(__file__), '..', 'configurations', 'config.ini')
config.read(config_path)


class ReadConfig:

    @staticmethod
    def getLLMUrl():
        url = config.get('commonInfo', 'baseURL')
        return url

    @staticmethod
    def getEvaluationModel():
        evaluation_model = config.get('commonInfo', 'evaluation_model')
        return evaluation_model

    @staticmethod
    def getTemperature():
        temperature = config.get('commonInfo', 'temperature')
        return temperature

    @staticmethod
    def getThreshold():
        return config.getfloat("commonInfo", "threshold")