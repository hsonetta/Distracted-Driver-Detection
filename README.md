# Distracted-Driver-Detection

Distracted Driver detection using Keras, MTCNN, OpenCV and Flask.

[![Watch the video](https://i.imgur.com/SyhvFfX.png)](https://youtu.be/EVk4aAk-l5Q)

The dataset used for this project was utilized from kaggle. You can find the original dataset available here : https://www.kaggle.com/c/state-farm-distracted-driver-detection

This dataset consists of thousands of images showing a variety of behaviors exhibited by drivers while driving. From this set, I selected a subset of behaviors which consisted of : Safe Driving, Texting, talking on phone, operating radio, reaching behind. The final size of dataset consisted of 12000 images.

Tools Used: Google Colab, Jupyter Notebook, Eclipse

Workflow:

1. Trained VGG16 model to recognize the distracted drivers.

2. Used MTCNN (Multi-task Cascade Convolutional Neural Network) to detect profile face of humans in an image.
Reference : https://github.com/ipazc/mtcnn

3. After detection of human face in an image, predicted the probabilities of the behaviour in the frame using trained model weights.

4. Deployed the model on flask to make real time predictions. (Either live camera feed or upload a video)


Files:

model.py : This class will give us the predictions of our previously trained model.

camera.py : This file implements a camera class that does the following operations: 

- Get the image stream from our input (Webcam feed or from video)
- Detect faces with MTCNN and add bounding boxes
- Rescale the images and send them to our trained deep learning model 
- get the predictions back from our trained model and add the label to each frame and return the final image stream

main.py : Lastly, our main script will create a Flask app that will render our image predictions into a web page.
