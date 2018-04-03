from keras.models import load_model
from PIL import Image
import os
import numpy as np
import tensorflow as tf

model_name = './weights.hdf5'
model = load_model(model_name)
img_dir = './'
for img in os.listdir(img_dir):
    if img.endswith('.jpg'):
        img = Image.open(img)
        img = img.resize((256, 256))
        x = np.asarray(img, dtype = 'float32' )
        x = x * 1.0 / 255
        x = np.expand_dims(x, axis=0)
        results = model.predict(x)
        index = tf.argmax(results,1)
        #print('Predicted:', decode_predictions(results, top=5)[0])
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print sess.run(index)
