from keras.models import model_from_json
import numpy as np
from tensorflow import keras
import tensorflow as tf
import numpy as np
import time

class DriverBehaviourModel(object):

    Behaviour_LIST = ['safe driving',
                     'texting',
                     'texting',
                     'talking on phone',
                     'operating the radio',
                     'reaching behind']

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model._make_predict_function()

    def predict_emotion(self, img):
            self.preds = self.loaded_model.predict(img)
            return DriverBehaviourModel.Behaviour_LIST[np.argmax(self.preds)]