from utils import *
import keras
from keras.layers import Input, Dense, Flatten
from keras.applications.mobilenet import MobileNet
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator


def get_labels(data):

    # create labels
    key = dict()
    key['neutral'] = np.array([1, 0, 0, 0, 0])
    key['anger'] = np.array([0, 1, 0, 0, 0])
    key['surprise'] = np.array([0, 0, 1, 0, 0])
    key['smile'] = np.array([0, 0, 0, 1, 0])
    key['sad'] = np.array([0, 0, 0, 0, 1])

    return np.array([key[i[3]] for i in data])

def get_data(data, shape):
    return np.array([preprocess(img[1], shape) for img in data])

def get_MobileNet_v2 (shape):

    input_tensor = Input(shape=shape)

    model = MobileNet(input_tensor=input_tensor, alpha=1.0,include_top=False, weights=None)

    output = Flatten()(model.output)
    output = Dense(5, activation='softmax')(output)
    model = keras.Model(model.input, output)

    return model

def build(train_x, train_y, test_x, test_y, model):

    # to compensate for the fact that some faces are tilted.
    # normally i do this with open cv cascade classifier and get rotationMatrix2D
    # however I've noticed that some of the images miss the eye detection
    # so i go through generation of all examples as dataset is not large

    data_generator = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=.1,
        horizontal_flip=True)

    CALLBACK_PATIENCE = 50
    BATCH_SIZE = 32
    NUM_EPOCHS = 50


    early_stop = EarlyStopping('val_loss', patience=CALLBACK_PATIENCE)
    checkpoint = ModelCheckpoint('model', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(
        'val_loss', factor=0.1, patience=int(CALLBACK_PATIENCE / 4), verbose=1)
    callbacks = [early_stop, reduce_lr, checkpoint]

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit_generator(data_generator.flow(train_x, train_y, BATCH_SIZE),
                        steps_per_epoch=len(train_x) / BATCH_SIZE,
                        epochs=NUM_EPOCHS, verbose=1, callbacks=callbacks,
                        validation_data=(test_x, test_y))
    return model