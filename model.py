import csv
import cv2
import numpy as np
import sklearn
import random

def open_csv_file(csv_file):
    lines=[]
    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    lines = lines[1:]
    return lines

def histogram(samples):
    bins=[0,0,0,0,0,0,0,0,0,0]
    for line in samples:
        val = float(line[3])
        if val >= -1. and val <=-0.8:
            bins[0]+=1
        elif val > -0.8 and val <=-0.6:
            bins[1]+=1
        elif val > -0.6 and val <=-0.4:
            bins[2]+=1
        elif val > -0.4 and val <=-0.2:
            bins[3]+=1
        elif val > -0.2 and val <=0.:
            bins[4]+=1
        elif val > 0. and val <=0.2:
            bins[5]+=1
        elif val > 0.2 and val <=0.4:
            bins[6]+=1
        elif val > 0.4 and val <=0.6:
            bins[7]+=1
        elif val > 0.6 and val <=0.8:
            bins[8]+=1
        elif val > 0.8 and val <=1.:
            bins[9]+=1
    print(bins)

def flatten_histogram(samples):
    bins=[0,0,0,0,0,0,0,0,0,0]
    new_samples=[]
    for line in samples:
        val = float(line[3])
        if ((val >= -1. and val <= -0.8 and random.random() < 1. ) or
            (val > -0.8 and val <= -0.6 and random.random() < 1. ) or
            (val > -0.6 and val <= -0.4 and random.random() < 1. ) or
            (val > -0.4 and val <= -0.2 and random.random() < 0.6) or
            (val > -0.2 and val <= 0.   and random.random() < 0.2) or
            (val > 0.   and val <= 0.2  and random.random() < 0.2) or
            (val > 0.2  and val <= 0.4  and random.random() < 0.6) or
            (val > 0.4  and val <= 0.6  and random.random() < 1. ) or
            (val > 0.6  and val <= 0.8  and random.random() < 1. ) or
            (val > 0.8  and val <= 1.   and random.random() < 1. )):
            new_samples.append(line)
    return new_samples

def import_images(lines, path):
    images=[]
    measurements=[]
    for line in lines:
        for i in range(3):
            source_path = line[i]
            filename = source_path.split('/')[-1]
            current_path = path + filename
            image = cv2.imread(current_path)
            images.append(image)
            measurement = float(line[3])
            correction = 0.28
            if   i == 0:
                measurements.append(measurement)
            elif i == 1:
                measurements.append(measurement + correction)
            elif i == 2:
                measurements.append(measurement - correction)
    return images, measurements

def augment_images(in_images, in_measurements, out_images, out_measurements):
    for image, measurement in zip(in_images, in_measurements):
        out_images.append(image)
        out_measurements.append(measurement)
        #if measurement != 0.: # dont flip images with zero steering angle
        out_images.append(cv2.flip(image,1))
        out_measurements.append(measurement*-1.0)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        for offset in range(0, num_samples, batch_size):
            aug_images=[]
            aug_angles=[]
            batch_samples = samples[offset : offset + batch_size]
            images, angles = import_images(batch_samples, '../car-sim-data/IMG/')
            augment_images(images, angles, aug_images, aug_angles)
            X_train = np.array(aug_images)
            y_train = np.array(aug_angles)
            yield sklearn.utils.shuffle(X_train, y_train)


samples = open_csv_file('../car-sim-data/driving_log.csv')


histogram(samples)
samples = flatten_histogram(samples)
histogram(samples)


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.1)


train_generator = generator(train_samples)
validation_generator = generator(validation_samples)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

model = Sequential()

model.add(Lambda(lambda x: x / 255.0 - 0.5,
    input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24,5,5,activation='relu',subsample=(2,2)))
model.add(Convolution2D(36,5,5,activation='relu',subsample=(2,2)))
model.add(Convolution2D(48,5,5,activation='relu',subsample=(2,2)))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=3*2*len(train_samples),
    validation_data=validation_generator, nb_val_samples=3*2*len(validation_samples),
    nb_epoch=7)


model.save('model.h5')
exit()
