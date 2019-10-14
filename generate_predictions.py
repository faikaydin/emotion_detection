import tensorflow as tf
from tensorflow import keras
import numpy as np
import model
import utils
import cv2

test_dir = '/Users/ezra/Downloads/Face_data_split/test'
test = utils.load_data_to_memory(test_dir)
test_x = model.get_data(test, 48)
test_y = model.get_labels(test)

model_ = keras.models.load_model('model.h5')
test_x = np.expand_dims(test_x, -1)
model_.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# running evaluation of the test set
model_.evaluate(test_x, test_y)

# producing the test set images
test_x = tf.cast(test_x, tf.float32)
predictions = model_.predict(test_x)

key_ = dict()
key_['0'] = 'neutral'
key_['1'] = 'anger'
key_['2'] = 'surprise'
key_['3'] = 'smile'
key_['4'] = 'sad'

predictions = [key_[str(np.argmax(i))] for i in predictions]
outdir = '/Users/ezra/Desktop/predictions/'
for i in range(len(predictions)):

    temp = cv2.rectangle(test[i][2], (test[i][4][0], test[i][4][1]), (test[i][4][2], test[i][4][3]), (0, 255, 0), 3)
    text = predictions[i]
    temp = cv2.putText(temp, text, (test[i][4][0], test[i][4][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imwrite(outdir + test[i][0] + '_prediction.jpg', temp)
