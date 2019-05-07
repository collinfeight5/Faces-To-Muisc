from keras.models import Sequential
from keras.layers import Conv2D, Cropping2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D
import numpy as np
import pandas as pd
import cv2 as cv


# assuming fer2013.csv file is in current directory for now
# will change to use os later
data = pd.read_csv('fer2013.csv')

# Data processing
# Find Image size for input to CNN so training can be done on other data
image = data['pixels']
total_samples = image.shape[0]
#image_shape = image.shape[1:]

# store data in csv in numpy arrays that can be used for CNN
labels = np.array(data['emotion'])
images = np.array(data['pixels'])


# Build CNN model
activation_type = 'elu'
dropout = .5
model = Sequential()
batch_size = 100
epochs = 3


model.add(Conv2D(16, kernel_size=(3,3), activation=activation_type, input_shape=(48,48,1)))
model.add(Conv2D(28, kernel_size=(3,3), activation=activation_type))
model.add(Conv2D(40, kernel_size=(3,3), activation='elu'))
# stride is same as pool size since not specified
#model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(dropout))
model.add(Dense(1000, activation=activation_type))
model.add(Dense(100, activation=activation_type))
model.add(Dense(1, activation=activation_type))
model.compile(optimizer='adam', loss='mse')
#model.reset_states()
model.fit(images, labels, batch_size=batch_size, epochs=epochs, validation_split=.2, shuffle=True)
