import os

import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

MODEL = load_model("mymodel.dat")
CLASSES = os.listdir("flowers")
TEST_DIR = "flowers_test"

for file in os.listdir(TEST_DIR):
    if file.endswith(".jpg"):

        # load the image
        img = cv2.imread(file)
        out = img.copy()

        # prepare image
        img = cv2.resize(img, (32, 32))
        img = img.astype("float") / 255.0
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)

        # predict
        proba = MODEL.predict(img)[0]
        idx = np.argmax(proba)

        # show
        text = "{:s} : {:.2f}".format(CLASSES[idx], proba[idx])
        cv2.putText(out, text, (10, 30), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1, thickness=2, color=(255, 0, 0))

        cv2.imshow(file, out)

cv2.waitKey(0)
cv2.destroyAllWindows()
