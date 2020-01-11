import cv2
from model import DriverBehaviourModel
import numpy as np
import tensorflow as tf
import keras as k
from mtcnn.mtcnn import MTCNN
from PIL import Image


session = tf.Session(graph=tf.Graph())
with session.graph.as_default():
    k.backend.set_session(session)
    model = DriverBehaviourModel("model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self):
    	#Arguement '0' takes feed from camera
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        pixels = np.asarray(fr)
        detector = MTCNN()
        box = detector.detect_faces(pixels)[0]['box']
        fc = pixels[box[1]:box[1]+box[3],box[0]:box[0]+box[2]]
        roi = cv2.resize(fc, (224, 224))
        with session.graph.as_default():
            k.backend.set_session(session)
            pred = model.predict_emotion(roi[np.newaxis, :, :])

        cv2.putText(fr, pred, (box[0], box[1]), font, 2, (0 ,0, 255), 3)
        cv2.rectangle(fr,(box[0], box[1]),(box[0]+box[2],box[1]+box[3]),(0,255,0),2)


        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()
