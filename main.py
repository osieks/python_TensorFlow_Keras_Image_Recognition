import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10

import pandas as pd
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import os
import numpy as np
from tensorflow import keras

class MyKerasModel:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
        self.history = None

    def build_model(self):
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.2),
            keras.layers.BatchNormalization(),

            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.BatchNormalization(),

            keras.layers.Flatten(),
            keras.layers.Dropout(0.2),

            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(self.num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        return model

    def train(self, x_train, y_train, epochs=10, batch_size=32, validation_split=0.2):
        self.history = self.model.fit(x_train, y_train, epochs=epochs, 
                                      batch_size=batch_size, validation_split=validation_split)
        return self.history

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def predict(self, x):
        return self.model.predict(x)

    def save_model(self, filepath):
        self.model.save(filepath)
        if self.history:
            np.save(filepath + '_history.npy', self.history.history)
        print(f"Model and training history saved to {filepath} and {filepath}_history.npy")

    @classmethod
    def load_model(cls, filepath):
        model = keras.models.load_model(filepath)
        instance = cls(model.input_shape[1:], model.output_shape[-1])  # Create a new instance
        instance.model = model
        
        history_path = filepath + '_history.npy'
        if os.path.exists(history_path):
            history_from_file = np.load(history_path, allow_pickle=True).item()
            instance.history = type('History', (), {'history': history_from_file})()
            print(f"Model and history loaded from {filepath}")
        else:
            print(f"Model loaded from {filepath}, but history file not found.")
        return instance


def main(model_name='my_cifar10_model', model_num=0, training=True):
    seed = 21
    np.random.seed(seed)
    tf.random.set_seed(seed)

    print(tf.__version__)
    print(tf.keras.__version__)

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    class_num = y_test.shape[1]

    model_path = f"{model_name}{model_num}.keras"

    modelObj = MyKerasModel(input_shape=X_train.shape[1:], num_classes=class_num)
    
    if not training and os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        modelObj = MyKerasModel.load_model(model_path)
        modelObj.model.summary()
    elif not training and not os.path.exists(model_path):
        print(f"Model {model_path} does not exist. Exiting.")
        return
    elif os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        modelObj = MyKerasModel.load_model(model_path)
        modelObj.model.summary()
        print("Loaded existing model. Training for 10 more epochs.")
        # Corrected method call (removed explicit 'self')
    else:
        print(f"Model {model_path} does not exist. Training new model.")
        modelObj.model.summary()
        print("Training new model for 50 epochs.")
        # Adjusted to 50 epochs as you initially mentioned for a new model, if required
        #modelObj.train(X_train, y_train, epochs=3, batch_size=64, validation_split=0.2)
    
    if training:
        modelObj.history = modelObj.model.fit(X_train, y_train, epochs=3, batch_size=64, validation_split=0.2)

    # Corrected method call (removed explicit 'self')
    modelObj.save_model(filepath=f"{model_name}{model_num+1}.keras")

    if modelObj.history:
        pd.DataFrame(modelObj.history.history).plot(figsize=(8, 5))
        plt.grid(True)
        #plt.gca().set_ylim(0, 2)
        plt.title('Model training history')
        plt.show()

    score = modelObj.evaluate(X_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

if __name__ == '__main__':
    # model_num=0 for new model
    main(model_name='my_cifar10_model', model_num=2, training=True)
