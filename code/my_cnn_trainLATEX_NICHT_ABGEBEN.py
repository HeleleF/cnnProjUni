#pylint: disable=C0111
import os 

import cv2
import numpy as np
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


def create_model(input_shape=(32, 32, 3), n_classes=5):
    '''
    use 6 convolutional layers and 1 fully-connected layer
    '''

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))
 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))
 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))
 
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))
     
    return model


def load_data(img_dir):
    '''
    load and prepare the dataset
    '''
    data = []
    labels = []
    for root, directories, filenames in os.walk(img_dir):
        for filename in filenames: 
            f = os.path.join(root,filename)
            image = cv2.imread(f)
            image = img_to_array(image)
            data.append(image)
            label = f.split(os.path.sep)[-2]
            labels.append(label)
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # binarize the labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    return data,labels


###############################################################################
# load the data from specified folder
data, labels = load_data("flowers")

# partition the data into training and testing splits using 85% of
# the data for training and the remaining 15% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.15)

model1 = create_model(input_shape=trainX.shape[1:], n_classes=5)
batch_size = 32
epochs = 100
model1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# construct the image generator for data augmentation
aug = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

# history = model1.fit(trainX, trainY, batch_size=batch_size, epochs=epochs, verbose=1, 
#                    validation_data=(testX, testY))

history = model1.fit_generator(aug.flow(trainX, trainY, batch_size=batch_size),
    validation_data=(testX, testY),	
	epochs=epochs, verbose=1, workers=4)

print("[INFO] Evaluating the model...")
model1.evaluate(testX, testY, verbose=1)

print("[INFO] Saving the model...")
model1.save("mymodel.dat")

print("[INFO] Finished!")
