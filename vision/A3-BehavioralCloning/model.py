import csv
import cv2
import numpy as np

lines = []

with open('data/driving_log.csv') as csvDataFile:
    csvReader =csv.reader(csvDataFile)
    for line in csvReader:
        lines.append(line)

# this represents the features
images = []
# this represents the labels to predict
measurements = []

print ("there are %d lines in the file"%(len(lines)))

for line in lines:

    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path =filename

    # print("current path", current_path)

    image = cv2.imread(current_path)
    images.append(image)
    # images.append(cv2.flip(image, 1))
    images.append(cv2.imread(line[1]))
    images.append(cv2.imread(line[2]))


    # extracting the steering wheel as labels
    # print ("steering angle", line[3])


    measurement = float(line[3])

    # print ("measurement", measurement)


    measurements.append(measurement)
    # measurements.append(-1*measurement)
    measurements.append(measurement+0.25)
    measurements.append(measurement-0.25)


X_train = np.array(images)
y_train = np.array(measurements)
print("AUGMENTED")


from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


# Testing the data on a simplistic neural network

model = Sequential()
print("Sequential")

model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160, 320, 3)))
print("Lambda applied")

model.add(Cropping2D(cropping=((70, 25), (0, 0))))
print("Cropping applied")

# Nvidia network architecture

model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))



model.compile(loss='mse', optimizer='adam')

history_object=model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=16)

model.save('model.h5')

# print(history_object.history.keys())

# plot the training and validation loss for each epoch


import matplotlib.pyplot as plt


plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
