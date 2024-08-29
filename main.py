import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.constraints import max_norm  # Note: it's max_norm, not maxnorm
from tensorflow.keras.utils import to_categorical  # This replaces np_utils
from tensorflow.keras.datasets import cifar10

import pandas as pd
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class MyKerasModel:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        model = keras.Sequential([
            #od szczegółu do ogółu
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.2),
            keras.layers.BatchNormalization(),

            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.BatchNormalization(),
            #keras.layers.MaxPooling2D((2, 2)),
            #keras.layers.Conv2D(64, (3, 3), activation='relu'),
            
            # after convolutional layers, we need to Flatten the data
            keras.layers.Flatten(),
            keras.layers.Dropout(0.2),

            # Dense to create the first densely connected layer.
            #Beware of dense layers. Since they're fully-connected, having just a couple of layers here instead of a single one significantly bumps the number of learn-able parameters upwards.
            #keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),

            #Finally, the softmax activation function selects the neuron with the highest probability as its output
            keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # The Adaptive Moment Estimation (Adam) algorithm is a very commonly used optimizer, and a very sensible default optimizer to try out.
        # other like Nadam, RMSprop, etc.
        #keeping track of accuracy and validation accuracy to make sure we avoid overfitting CNN badly
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        return model

    def train(self, x_train, y_train, epochs=10, batch_size=32, validation_split=0.2):
        return self.model.fit(x_train, y_train, epochs=epochs, 
                              batch_size=batch_size, validation_split=validation_split)

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def predict(self, x):
        return self.model.predict(x)
    
    def summary(self):
        return self.model.summary()
    def save_model(self, filepath):
        # Save the model architecture and weights
        self.model.save(filepath)
        # Save the training history
        np.save(filepath + '_history.npy', self.history.history)
        print(f"Model and training history saved to {filepath}")

    @classmethod
    def load_model(cls, filepath):
        # Load the model architecture and weights
        loaded_model = keras.models.load_model(filepath)
        instance = cls(loaded_model.input_shape[1:], loaded_model.output_shape[1])
        instance.model = loaded_model
        # Load the training history if it exists
        history_path = filepath + '_history.npy'
        if os.path.exists(history_path):
            history = np.load(history_path, allow_pickle=True).item()
            instance.history = type('History', (), {'history': history})()
        print(f"Model loaded from {filepath}")
        return instance
    

def main(model_name='my_cifar10_model',model_num=0, training=True):
    # Set random seed for purposes of reproducibility
    seed = 21

    print(tf.__version__)
    print(tf.keras.__version__)

    # Loading in the data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # One-hot encode outputs
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    class_num = y_test.shape[1]

    model_path = model_name+str(model_num)+'.h5'

    if training == False:
        print("Loading existing model from", model_path)
        model = MyKerasModel.load_model(model_path)
        model.summary()
    elif os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        # Load the model if it exists
        model = MyKerasModel.load_model(model_path)
        model.summary()
        print("Loaded existing model. Training for 10 more epochs.")
        history = model.train(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)
    else:
        print(f"Model {model_path} does not exist. Training new model.")
        # Create and train a new model if it doesn't exist
        model = MyKerasModel(input_shape=X_train.shape[1:], num_classes=class_num)
        model.summary()
        print("Training new model for 50 epochs.")
        history = model.train(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2)
    
    # Save the model after training
    model.save_model('my_cifar10_model'+str(model_num+1)+'.h5')

    # Train the model
    history = model.train(X_train, y_train, epochs=25, batch_size=32, validation_split=0.2)

    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.title('Model training history')
    plt.show()

    # Evaluate the model
    score = model.evaluate(X_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    #if you want to train a new model, set model_num = 0
    main(model_name='my_cifar10_model',model_num=0)