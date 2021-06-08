from tensorflow.keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import layers,models
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import matplotlib.pyplot as plt

def build_model(img_width, img_height):
    # Initializing weights with Imagenet weights
    mobilenet = applications.MobileNetV2(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))

    # freezing the layers except last 6 layers
    for layer in mobilenet.layers[:-6]:
        layer.trainable = False

    #debugging
    for layer in mobilenet.layers:
        print(layer.name, layer.trainable)

    # Create the model
    model = models.Sequential()

    # Add the mobilenet convolutional base model
    model.add(mobilenet)

    # Add new layers
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))

    return model

def train_model(model, img_width, img_height):
    train_data_dir = "C:/Users/hsone/Desktop/Extras/Projects/Distracted-Driver-Detection-master/Big_dataset/train"
    validation_data_dir = "C:/Users/hsone/Desktop/Extras/Projects/Distracted-Driver-Detection-master/Big_dataset/validation"

    train_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()

    train_batch_size = 56
    valid_batch_size = 8

    train_generator = train_datagen.flow_from_directory(directory=train_data_dir,
                                                        target_size=(img_height, img_width),
                                                        batch_size=train_batch_size,
                                                        class_mode="categorical",
                                                        shuffle=True)

    validation_generator = test_datagen.flow_from_directory(validation_data_dir,
                                                            target_size=(img_height, img_width),
                                                            batch_size=valid_batch_size,
                                                            class_mode="categorical",
                                                            shuffle=True)

    opt = Adam(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # checkpoint will save the best weights
    checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples / train_generator.batch_size,
        epochs=1,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples / validation_generator.batch_size,
        verbose=1,
        callbacks=callbacks_list
    )

    return history

def visualization(history):
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.suptitle('Optimizer : Adam', fontsize=10)
    plt.ylabel('Loss', fontsize=16)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend(loc='upper right')

    plt.subplot(1, 2, 2)
    plt.ylabel('Accuracy', fontsize=16)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.show()

def main():
    img_width, img_height = 224, 224
    model = build_model(img_width, img_height)
    history = train_model(model, img_width, img_height)
    visualization(history)

if __name__ == '__main__':
    main()

