#pylint: disable=C0111,E1101
import os
import itertools

import cv2
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from keras.backend.tensorflow_backend import set_session
from keras.callbacks import Callback
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.optimizers import SGD

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6 #0.4 seems to work
set_session(tf.Session(config=config))

IMG_SIZE = 32

clas_names = []

# pylint: disable=
class MyMetrics(Callback):
    '''
    calculate some metrics during training
    taken from https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2
    '''
    def __init__(self, tX=None, tY=None):
        super().__init__()

        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

        # https://github.com/keras-team/keras/issues/10472
        self.validation_data = (tX, tY)

    def on_epoch_end(self, epoch, logs=None):
   
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]

        # https://stackoverflow.com/questions/45890328/sklearn-metrics-for-multiclass-classification
        _val_f1 = f1_score(val_targ, val_predict, average='micro')
        _val_recall = recall_score(val_targ, val_predict, average='micro')
        _val_precision = precision_score(val_targ, val_predict, average='micro')

        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)

        print(" - val_f1: {:.4f} - val_precision: {:.4f} - val_recall: {:.4f}".format(_val_f1, _val_precision, _val_recall))
        return


def create_model(input_shape=(32, 32, 3), n_classes=5, show=False):
    '''
    #use convolutional layers and 1 fully-connected layer
    '''

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.2))

    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.2))

    model.add(Conv2D(8, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(8, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(n_classes, activation='softmax'))

    if show: 
        model.summary()
 
    return model


def load_data(img_dir):
    '''
    load and prepare the dataset
    '''
    tdata = []
    tlabels = []

    # traverse main dir
    for root, _, filenames in os.walk(img_dir):
        print("[INFO] Processing {:s}...".format(root))
        for filename in filenames:
            # get full path to image
            img_file = os.path.join(root, filename)

            # read image and resize
            image = cv2.imread(img_file)
            rimage = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        	# convert image and append to data
            rimage = img_to_array(rimage)
            tdata.append(rimage)

            # append corresponding label
            label = img_file.split(os.path.sep)[-2]
            tlabels.append(label)


    # [0-255] -> [0-1]
    tdata = np.array(tdata, dtype="float") / 255.0
    tlabels = np.array(tlabels)

    # binarize the labels
    label_bin = LabelBinarizer()
    tlabels = label_bin.fit_transform(tlabels)

    np.save("flowers_data.npy", tdata)
    np.save("flowers_labels.npy", tlabels)
    return tdata, tlabels


def prepare_data(main_img_dir):

    if os.path.exists("flowers_data.npy") and os.path.exists("flowers_labels.npy"):

        # load numpy files (faster than always loading and resizing all images)
        print("[INFO] Loading dataset from numpy files...")
        data = np.load("flowers_data.npy")
        labels = np.load("flowers_labels.npy")
    else:
        # load the data from specified folder
        print("[INFO] Loading dataset...")
        data, labels = load_data(main_img_dir)

    return train_test_split(data, labels, test_size=0.25)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def show_stats(his, metr):

    fig = plt.figure()
    plt.plot(his.history['acc'])
    plt.plot(his.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("accuracy.png")
    plt.show()    

    plt.plot(his.history['loss'])
    plt.plot(his.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("loss.png")
    plt.show() 

    plt.plot(metr.val_precisions)
    plt.plot(metr.val_recalls)
    plt.title('model precision/recall')
    plt.ylabel('precision/recall')
    plt.xlabel('epoch')
    plt.legend(['precision', 'recall'], loc='upper left')
    plt.savefig("prec_rec.png")
    plt.show()
    
    plt.plot(metr.val_f1s)
    plt.title('model f1 score')
    plt.ylabel('f1 score')
    plt.xlabel('epoch')
    plt.legend(['f1 score'], loc='upper left')
    plt.savefig("f1score.png")
    plt.show()
    

def start_main():

    # get data for train and test
    (trainX, testX, trainY, testY) = prepare_data("./flowers")

    print("[INFO] Using {:d} samples for training and {:d} samples for testing".format(trainX.shape[0], testX.shape[0]))

    model1 = create_model(input_shape=trainX.shape[1:], n_classes=5, show=True)
    batch_size = 64 # smaller -> slower
    epochs = 100 # higher -> nearly always useless after some point around 100 or less -> stagnant curve

    print("[INFO] Compiling  model...")
    model1.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy']) # loss -> categorical because mutlilabel, optimizer -> no big changes

    # construct the image generator for data augmentation
    aug = ImageDataGenerator(
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

    mymetrics = MyMetrics(testX, testY)


    print("[INFO] Training the model...")
    history = model1.fit_generator(aug.flow(trainX, trainY, batch_size=batch_size),
                                   validation_data=(testX, testY),
	                                  epochs=epochs, verbose=1, callbacks=[mymetrics])

    print("[INFO] Evaluating the model...")
    scores = model1.evaluate(testX, testY, verbose=1)

    y_pred = model1.predict(testX)

    cnf_matrix = confusion_matrix(testY, y_pred)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

    plt.show()

    print("Accuracy: {:.4f}".format(scores[1]))
    print("Precision: {:.4f}".format(mymetrics.val_precisions[-1]))
    print("Recall: {:.4f}".format(mymetrics.val_recalls[-1]))
    print("F1 score: {:.4f}".format(mymetrics.val_f1s[-1]))

    print("[INFO] Saving the model...")
    model1.save("mymodel.dat")

    show_stats(history, mymetrics)
    print("[INFO] Finished!")


start_main()
