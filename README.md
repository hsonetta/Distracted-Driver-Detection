# Distracted-Driver-Detection

Distracted Driver detection using Keras, MTCNN and Flask

[![Watch the video](https://i.imgur.com/SyhvFfX.png)](https://youtu.be/n35xB_-uSYE)

The dataset used for this project was utilized from kaggle. You can find the original dataset available here : https://www.kaggle.com/c/state-farm-distracted-driver-detection

This dataset consists of thousands of images showing a variety of behaviors exhibited by drivers while driving. From this set, I selected a subset of behaviors which consisted of : Safe Driving, Texting, talking on phone, operating radio, reaching behind. The final size of dataset consisted of 12000 images.

Tools Used: Google Colab, Jupyter Notebook, Eclipse

Workflow:

1. Trained VGG16 model to recognize the distracted drivers.

2. Used MTCNN (Multi-task Cascade Convolutional Neural Network) to detect profile face of humans in an image.
Reference : https://github.com/ipazc/mtcnn

3. After detection of human face in an image, predicted the probabilities of the behaviour in the frame using trained model weights.

4. Deployed the model on flask to make real time predictions. (Either live camera feed or upload a video)
