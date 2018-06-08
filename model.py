import numpy as np
import keras.backend as K
import tensorflow as tf
import os
from scipy.misc import imread
from keras.utils import Sequence
from keras import applications
from keras.models import Model
from keras.layers import  BatchNormalization, Flatten, Dense
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras import optimizers, regularizers
from keras.preprocessing.image import ImageDataGenerator

img_width, img_height = 224, 224
classes = 14951
train_data_dir = 'data'
batch_size = 50
input_shape = (img_width, img_height, 3)

run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)

class trainSequence(Sequence):
    def __init__(self, data_dir, batch_size, val=0.15):
        assert(val < 0.5)
        self.data_dir = data_dir
        self.batch_size = batch_size
        class_dirs = os.listdir(data_dir)
        self.image_list = []
        for class_dir in class_dirs:
            class_dir_path = os.path.join(data_dir, class_dir)
            imgs = os.listdir(class_dir_path)
            for img in imgs:
                self.image_list.append((os.path.join(data_dir, class_dir, img), int(class_dir)))
        np.random.shuffle(self.image_list)
        val_idx = int(val*len(self.image_list))
        self.val = self.image_list[:val_idx]
        self.image_list = self.image_list[val_idx:]
        
    def __len__(self):
        return np.ceil(len(self.image_list) / self.batch_size)
        
    def __getitem__(self, idx):
        batch_x, batch_y = zip(*self.image_list[idx * self.batch_size : (idx + 1) * self.batch_size])
        return np.array([imread(x) for x in batch_x]), np.array(batch_y)
    
    def get_val(self):
        return self.val
    
class valSequence(Sequence):
    def __init__(self, data, batch_size):
        self.val = data
        self.batch_size = batch_size
    
    def __len__(self):
        return np.ceil(len(self.val) / self.batch_size)
    
    def __getitem__(self, idx):
        batch_x, batch_y = zip(*self.val[idx * self.batch_size : (idx + 1) * self.val])
        return np.array([imread(x) for x in batch_x]), np.array(batch_y)
        

def declare_model(reg=0):
    resnet = applications.resnet50.ResNet50(include_top=False,
     weights="imagenet",
     input_shape=input_shape,
     pooling=None
    )
    for layer in resnet.layers[:-5]:
        layer.trainable = False

    x = resnet.output
    # x = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu', kernel_regularizer=regularizer)(x)
    # x = BatchNormalization()(x)
    # x = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu', kernel_regularizer=regularizer)(x)
    # x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(15000, activation='relu', kernel_regularizer=regularizers.l2(reg))(x)
    scores = Dense(classes, activation='softmax', kernel_regularizer=regularizers.l2(reg))(x)

    return Model(inputs=resnet.inputs, outputs=scores)

def compile_and_train(model, data_dir, epochs, mean, std, checkpoint_dir='', lr=1e-3, steps_per_epoch=None, use_tfboard=False):
    opt = optimizers.Adam(lr=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', gap], options=run_opts)

    # Data generators
    generator = ImageDataGenerator(validation_split=0.2)

    train_iterator = generator.flow_from_directory(data_dir, 
                                                   target_size=(img_height, img_width), 
                                                   class_mode='categorical', 
                                                   batch_size=batch_size, 
                                                   subset='training')
    val_iterator = generator.flow_from_directory(data_dir, 
                                                 target_size=(img_height, img_width), 
                                                 class_mode='categorical', 
                                                 batch_size=batch_size, 
                                                 subset='validation')

    train_iterator = img_gen_normalized(train_iterator, mean, std)
    val_iterator = img_gen_normalized(val_iterator, mean, std)
#     train_iterator = trainSequence(data_dir, batch_size)
#     val_iterator = valSequence(train_iterator.get_val(), batch_size)
    

    # Note: checkout the LearningRateScheduler callback.
    # Callbacks
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=1, verbose=0, mode='auto')
    callbacks = [early]

    if use_tfboard:
        tfboard = TensorBoard(batch_size=batch_size)
        callbacks.append(tfboard)

    if checkpoint_dir:
        checkpoint = ModelCheckpoint(checkpoint_dir + "/epoch{epoch:02d}_checkpoint.hdf5", monitor='val_acc',
                                     verbose=0, save_best_only=True, save_weights_only=False,
                                     mode='auto', period=1)
        callbacks.append(checkpoint)

    # Training
    print('Beginning training')
    hist = model.fit_generator(train_iterator,
                               steps_per_epoch=steps_per_epoch, 
                               epochs=epochs,
                               verbose=0, 
                               validation_data=val_iterator, 
                               callbacks=callbacks,
                               validation_steps=100,
                               max_queue_size=10,
                               workers=4,
                               use_multiprocessing=False)
    print('Training concluded')

    return hist


def normalize_and_rescale(x, mean, std):
    # We do the rescaling here to prevent rescaling happening after normalization
    return (x - mean) / (255 * std)

def img_gen_normalized(generator, mean, std):
    for x, y in generator:
        yield normalize_and_rescale(x, mean, std), y

def predict(model, test_dir, mean, std):
    generator = ImageDataGenerator()

    test_iterator = img_gen_normalized(generator.flow_from_directory(test_dir, class_mode='categorical',
                                                                     target_size=(img_height, img_width),
                                                                     batch_size=batch_size), mean, std)

    preds = model.predict_generator(test_iterator)

    return preds

def evaluate(model, test_dir, mean, std):
    # Test data generator
    generator = ImageDataGenerator()

    test_iterator = img_gen_normalized(generator.flow_from_directory(test_dir, class_mode='categorical',
                                                                     target_size=(img_height, img_width),
                                                                     batch_size=batch_size), mean, std)

    # Evaluation
    # See model.metrics_names to see the corresponding metric names
    metrics = model.evaluate_generator(test_iterator)

    return metrics

def gap(y_true, y_pred):
    # If numbers returned are unreasonable it is because y_pred is the predicted label and not the confidence.
    inds = K.argmax(y_true, axis=1)
    summands = tf.gather(y_pred, inds)
    sm =  K.sum(summands)
    denom = tf.shape(y_true)[0]
    denom = tf.to_float(denom)
    return sm/denom
