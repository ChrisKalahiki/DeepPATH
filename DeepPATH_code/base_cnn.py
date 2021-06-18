'''
By: Chris Kalahiki

Testing the cropped data set on a generic CNN model. 
Just need a quick script to run everything.
'''

import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import random, os, collections, io
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
# import matplotlib as pyplot

# Import the data
'''
I would look at the code from the DeepPATH repo for this. Or my old street sign repo.
'''
def make_train_and_test_sets():
    """Split the data into train and test sets and get the label classes."""
    train_examples, test_examples = [], []
    shuffler = random.Random()
    is_root = True
    for (dirname, subdirs, filenames) in tf.io.gfile.walk('/zfs/dzrptlab/breastcancer/data_cropped/'):
        # The root directory gives us the classes
        if is_root:
            subdirs = sorted(subdirs)
            classes = collections.OrderedDict(enumerate(subdirs))
            label_to_class = dict([(x, i) for i, x in enumerate(subdirs)])
            is_root = False
        # The sub directories give us the image files for training.
        else:
            filenames.sort()
            shuffler.shuffle(filenames)
            full_filenames = [os.path.join(dirname, f) for f in filenames]
            label = dirname.split('/')[-1] # '/' for linux and '\\' for windows
            label_class = label_to_class[label]
            examples = list(zip(full_filenames, [label_class] * len(filenames)))
            num_train = int(len(filenames) * 0.7)
            train_examples.extend(examples[:num_train])
            test_examples.extend(examples[num_train:])
    shuffler.shuffle(train_examples)
    shuffler.shuffle(test_examples)
    
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for x in train_examples:
        tmp = Image.open(x[0]).resize((1116, 2011))
        data = np.asarray(tmp)
        x_train.append(data)
#         x_train.append(x[0])
        tmp = np.asarray([x[1]])
        y_train.append(tmp)
    for y in test_examples:
        tmp = Image.open(y[0]).resize((1116, 2011))
        data = np.asarray(tmp)
        x_test.append(data)
#         x_test.append(y[0])
        tmp = np.asarray([y[1]])
        y_test.append(tmp)
    return x_train, y_train, x_test, y_test, classes

TRAIN_SAMPLE, TRAIN_LABEL, TEST_SAMPLE, TEST_LABEL, CLASSES = make_train_and_test_sets()

TRAIN_SAMPLE = np.array(TRAIN_SAMPLE)
TEST_SAMPLE = np.array(TRAIN_SAMPLE)
TRAIN_LABEL = np.array(TRAIN_LABEL)
TEST_LABEL = np.array(TEST_LABEL)


# This is the Keras-style CNN in TensorFlow
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(2011, 1116, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

# model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(TRAIN_SAMPLE, TRAIN_LABEL, epochs=10, 
                    validation_data=(TEST_SAMPLE, TEST_LABEL))

# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(TEST_SAMPLE,  TEST_LABEL, verbose=2)

print(test_acc)
