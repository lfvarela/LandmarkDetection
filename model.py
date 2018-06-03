import numpy as np
from keras import applications
from keras.models import Model
from keras.layers import  Conv2D, BatchNormalization, Flatten, Dense
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras import optimizers, regularizers
from keras.preprocessing.image import ImageDataGenerator

img_width, img_height = 224, 224
classes = 14951
train_data_dir = 'data'
batch_size = 50
input_shape = (img_width, img_height, 3)

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
    x = Dense(30000, activation='relu', kernel_regularizer=regularizers.l2(reg))(x)
    scores = Dense(classes, activation='softmax', kernel_regularizer=regularizers.l2(reg))(x)

    return Model(input=resnet.input, output=scores)

def compile_and_train(model, data_dir, epochs, checkpoint_dir='', lr=1e-3, steps_per_epoch=None, use_tfboard=False):
    opt = optimizers.Adam(lr=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', gap])

    # Data generators
    generator = ImageDataGenerator(rescale=1/255, validation_split=0.15)

    train_iterator = generator.flow_from_directory(data_dir, class_mode='categorical', batch_size=batch_size, subset='training')
    val_iterator = generator.flow_from_directory(data_dir, class_mode='categorical', batch_size=batch_size, subset='validation')

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
    hist = model.fit_generator(train_iterator, batch_size=batch_size,
                               steps_per_epoch=steps_per_epoch, epochs=epochs,
                               verbose=1, validation_data=val_iterator, callbacks=callbacks)

    return hist


def normalize_and_rescale(x, mean, std):
    # We do the rescaling here to prevent rescaling happening after normalization
    return (x - mean) / (255 * std)

def img_gen_normalized(generator, normalize, mean, std):
    for x, y in generator:
        yield normalize(x, mean, std), y

def predict(model, test_dir, mean, std):
    generator = ImageDataGenerator()

    test_iterator = img_gen_normalized(generator.flow_from_directory(test_dir, class_mode='categorical',
                                                                    batch_size=batch_size),
                                       normalize_and_rescale, mean, std)

    preds = model.predict_generator(test_iterator)

    return preds

def evaluate(model, test_dir, mean, std):
    # Test data generator
    generator = ImageDataGenerator()

    test_iterator = img_gen_normalized(generator.flow_from_directory(test_dir, class_mode='categorical',
                                                                     batch_size=batch_size),
                                       normalize_and_rescale, mean, std)

    # Evaluation
    # See model.metrics_names to see the corresponding metric names
    metrics = model.evaluate_generator(test_iterator)

    return metrics

def gap(y_true, y_pred):
    # If numbers returned are unreasonable it is because y_pred is the predicted label and not the confidence.
    np.sum(y_pred[np.argmax(y_true, axis=1)])/y_true.shape[0]