

from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.applications import MobileNetV2
from keras.layers import AveragePooling2D
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Model
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.applications import vgg16 
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint


from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths

import matplotlib.pyplot as plt
import numpy as np
import os
import cv2 as cv
import os

## no may eat up all your disks
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

path = r'/home/fredson/project_detect/dataset/'


print("Load Images...")

imagePaths = list(paths.list_images(path))

print(len(imagePaths))
data = []
labels = []
for path_image in imagePaths:

    label = path_image.split(os.path.sep)[-2]
    image = cv.imread(path_image)
    #print('1', image.shape)
    image = cv.resize(image, (224,224), interpolation=cv.INTER_CUBIC)
    
    image = img_to_array(image)
    image = preprocess_input(image)
    
    data.append(image)
    labels.append(label)

data = np.array(data,  dtype="float32")
labels = np.array(labels)

lb = LabelBinarizer()
f_labels = lb.fit_transform(labels)
f_labels = to_categorical(f_labels)

(trainX, testX, trainY, testY) = train_test_split(data, f_labels, test_size=0.20, stratify=labels, random_state=42)

## shape of the images
print(trainX.shape)
print(testX.shape)
print(trainY.shape)
print(testY.shape)

a_data= ImageDataGenerator( rotation_range=20, vertical_flip = True, horizontal_flip=True,shear_range=0.15, zoom_range=0.15, fill_mode="nearest")


Mobi = MobileNetV2(include_top=False, weights="imagenet", input_shape=(224, 224, 3))

headModel = Mobi.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)
model = Model(inputs=Mobi.input, outputs= headModel)
model.summary()

## Hypermeters
print("Create model...")
opt = Adam(lr= 0.00001, decay= 0.000001)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

## save model of best epoch and finally train
es = EarlyStopping( monitor='val_loss', patience=5, mode='min', verbose=1)

mc = ModelCheckpoint('models/callbacks/model{epoch:001d}.h5' , monitor='val_loss', mode='min', verbose=1, save_best_only=True, save_freq='epoch')

## train
h_train = model.fit( a_data.flow(trainX, trainY, batch_size= 5), validation_data=(testX, testY),epochs=30, callbacks = [es,mc])

print("salving model detector...")

model.save('models/weigths/m_detector.h5')

## generate archie txt with results
archive = open('result/result.txt', 'a')
pred = model.predict(testX, batch_size=5)

pred = np.argmax(pred, axis=1)

r_info  = classification_report(testY.argmax(axis=1), pred , target_names=lb.classes_)

archive.write(r_info)
archive.close()

##save graphics of the train 
plt.figure(figsize=[8,6])
plt.plot(h_train.history['loss'] , label = "train loss" )
plt.plot(h_train.history['val_loss'] , label = "validation loss" )
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.savefig('result/graphics/loss.png' , format='png') 

plt.figure(figsize=[8,6])
plt.plot(h_train.history['accuracy'] , label = "train acc" )
plt.plot(h_train.history['val_accuracy'] , label = "validation acc" )
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.savefig('result/graphics/acurracy.png' , format='png') 

