#pylint: disable=C0111,E1101
import os
import itertools

import cv2
import numpy as np
import tensorflow as tf

from itertools import cycle

import matplotlib.pyplot as plt

from keras.backend.tensorflow_backend import set_session
from keras.callbacks import Callback
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator, img_to_array

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6 #0.4 seems to work
set_session(tf.Session(config=config))

IMG_SIZE = 32

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
    create the cNN model
    '''

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))

    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
 
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(n_classes, activation='softmax'))

    if show:
        model.summary()

    return model


def load_data(img_dir):
    '''
    loads and prepares the dataset
    '''
    tdata = []
    tlabels = []

    # traverse main dir
    for root, _, filenames in os.walk(img_dir):
        print("[INFO] Processing {:s}...".format(root))
        for filename in filenames:
            # get full path to image
            img_file = os.path.join(root, filename)

            # prepare image
            image = cv2.imread(img_file)
            rimage = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            rimage = img_to_array(rimage)
            tdata.append(rimage)

            # prepare corresponding label
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
    '''
    populates data and labels 
    '''

    if os.path.exists("flowers_data.npy") and os.path.exists("flowers_labels.npy"):

        # load numpy files (faster than always loading and resizing all images)
        print("[INFO] Loading dataset from numpy files...")
        data = np.load("flowers_data.npy")
        labels = np.load("flowers_labels.npy")
    else:
        # load the data from specified folder
        print("[INFO] Loading dataset...")
        data, labels = load_data(main_img_dir)

    return train_test_split(data, labels, test_size=0.2), os.listdir(main_img_dir)


def plot_confusion_matrix(cm, classes, normalize=False, title='ConfMatrix', cmap=plt.cm.Blues):
    """
    prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    taken from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

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
    plt.savefig("./images/" + title + ".pdf")


def calc_rocs(y_test, y_pred, class_names):
    '''
    calculates the roc curves and auc values for all
    taken from http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    '''
    ###################################################################################

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    n_classes = len(class_names)

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    # Compute macro-average ROC curve and ROC area
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]), color='red', linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]), color='blue', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC (micro and macro average)')
    plt.legend(loc="lower right")
    plt.savefig("./images/ROC-AUC(micro and macro average).pdf")
    #plt.show()

    colors = ['aqua', 'darkorange', 'green', 'red', 'yellow']

    plt.figure()
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve for {0:s} (area = {1:0.2f})'.format(class_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC (all classes)')
    plt.legend(loc="lower right")
    plt.savefig("./images/ROC-AUC(all classes).pdf")
    plt.show()

    ###################################################################################

    
def show_stats(his, metr):

    plt.plot(his.history['acc'])
    plt.plot(his.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("./images/accuracy.pdf")
    plt.show()

    plt.plot(his.history['loss'])
    plt.plot(his.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("./images/loss.pdf")
    plt.show()

    plt.plot(metr.val_precisions)
    plt.plot(metr.val_recalls)
    plt.title('model precision/recall')
    plt.ylabel('precision/recall')
    plt.xlabel('epoch')
    plt.legend(['precision', 'recall'], loc='upper left')
    plt.savefig("./images/prec_rec.pdf")
    plt.show()

    plt.plot(metr.val_f1s)
    plt.title('model f1 score')
    plt.ylabel('f1 score')
    plt.xlabel('epoch')
    plt.legend(['f1 score'], loc='upper left')
    plt.savefig("./images/f1score.pdf")
    plt.show()


def start_main():

    # get data for train and test
    (x_train, x_test, y_train, y_test), class_names = prepare_data("./flowerOrig")

    print("[INFO] Using {:d} samples for training and {:d} samples for testing".format(x_train.shape[0], x_test.shape[0]))

    model1 = create_model(input_shape=x_train.shape[1:], n_classes=len(class_names), show=True)

    batch_size = 64 # smaller -> slower
    epochs = 100 # higher -> nearly always useless after some point around 100 or less

    print("[INFO] Compiling  model...")
    # loss -> categorical because mutlilabel, optimizer -> no big changes
    model1.compile(optimizer="rmsprop", loss='categorical_crossentropy', metrics=['accuracy']) 

    # construct the image generator for data augmentation
    aug = ImageDataGenerator(
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

    mymetrics = MyMetrics(x_test, y_test)

    print("[INFO] Training the model...")
    history = model1.fit_generator(aug.flow(x_train, y_train, batch_size=batch_size),
                                   validation_data=(x_test, y_test),
	                                  epochs=epochs, verbose=1, callbacks=[mymetrics])

    print("[INFO] Evaluating the model...")
    scores = model1.evaluate(x_test, y_test, verbose=1)

    print("Accuracy: {:.4f}".format(scores[1]))
    print("Precision: {:.4f}".format(mymetrics.val_precisions[-1]))
    print("Recall: {:.4f}".format(mymetrics.val_recalls[-1]))
    print("F1 score: {:.4f}".format(mymetrics.val_f1s[-1]))

    show_stats(history, mymetrics)

    # confusion matrix
    y_pred = model1.predict(x_test)

    cnf_matrix = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

    #plt.show()

    calc_rocs(y_test, y_pred, class_names)

    print("[INFO] Saving the model...")
    model1.save("mymodel.dat")

    print("[INFO] Finished!")


start_main()
