from keras.utils import Sequence
from scipy.misc import imread
import keras
import numpy as np
#from keras.applications.resnet50 import preprocess_input

# This class was inspired by the following blog post:
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html

class DataGenerator(Sequence):
    def __init__(self, list_IDs, labels, batch_size=32, input_shape=(224, 224, 3),
                 n_classes=14951, shuffle=True):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indices = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __data_generation(self, list_IDs_temp):
        # Initialization
        X = np.empty((self.batch_size, *self.input_shape))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            # We read the image, normalize it and rescale it before storing.
            X[i,] = imread(ID, mode='RGB')

            # Store class
            y[i] = self.labels[ID]
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y