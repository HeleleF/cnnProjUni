#pylint: disable=C0111,E1101
import os

import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

import tensorflow as tf

if tf.test.is_built_with_cuda():

    from keras.backend.tensorflow_backend import set_session # pylint: disable=C0412

    CFG = tf.ConfigProto()
    CFG.gpu_options.per_process_gpu_memory_fraction = 0.4 #0.4 seems to work
    set_session(tf.Session(config=CFG))

MODEL = load_model("mymodel.dat")
CLASSES = os.listdir("./flowers")
TEST_DIR = "./flowers_test"
IMG_SIZE = 32

LINE_HEIGHT = 30

X_OFFSET = 10
Y_OFFSET = 30

WRITE_RESULT = False

# loop over all files in the test directory...
for ifile in os.listdir(TEST_DIR):

    # ...only take the images
    if ifile.endswith(".jpg"):

        # load the image
        img = cv2.imread(os.path.join(TEST_DIR, ifile))
        out = img.copy()

        # prepare image
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        img = img.astype("float") / 255.0
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)

        # classify it using the model
        probabilities = MODEL.predict(img)[0]

        # sort results
        sorted_idx = np.argsort(probabilities)[::-1]
        results = ["{0:s} : {1:3.2f}".format(CLASSES[ix], probabilities[ix]) for ix in sorted_idx if probabilities[ix] > 0.005]

        print("Prediction for {0:s}: ".format(ifile))

        # print results on the image
        for i, line in enumerate(results):
            cv2.putText(out, line, (X_OFFSET, Y_OFFSET + i * LINE_HEIGHT),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1,
                        thickness=1, color=(0, 0, 255))

            print("\t" + line)

        # show
        cv2.imshow(ifile, out)

        if WRITE_RESULT:
            cv2.imwrite("./images/classified" + ifile, out)

cv2.waitKey(0)
cv2.destroyAllWindows()
