# Building a program to recognize hand symbols about the alphabet
# Group of :
# Nguyen Vo Phong Thao  - 20119029
# Nguyen Anh Hao        - 20119222

# Use libraries
import keras
import tensorflow
import h5py
import matplotlib

# Part 1 - Building the CNN
#importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras import optimizers

# Initialing the CNN
classifier = Sequential([
        Convolution2D(32, (3, 3), (1, 1), input_shape = (64, 64, 3), activation = 'relu'),
        MaxPooling2D(pool_size=(2,2)),
        Convolution2D(32, (3, 3), (1, 1), activation = 'relu'),
        MaxPooling2D(pool_size=(2,2)),
        Convolution2D(64, (3, 3), (1, 1), activation = 'relu'),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(256, activation = 'relu'),
        Dropout(0.5),
        Dense(26, activation = 'softmax'),
])

#Compiling The CNN
classifier.compile(
              optimizer = optimizers.SGD(lr = 0.01), 
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

#Part 2 Fittting the CNN to the image
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

training_set = train_datagen.flow_from_directory(
        'mydata/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

test_datagen = ImageDataGenerator(
        rescale=1./255)

test_set = test_datagen.flow_from_directory(
        'mydata/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

model = classifier.fit(
        training_set,
        steps_per_epoch=800,
        epochs=50,
        validation_data = test_set,
        validation_steps = len(test_set),)

#Saving the model
classifier.save('Trained_model.h5')

print(model.history.keys())
import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(model.history['accuracy'])
plt.plot(model.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('result/model_accuracy.png')

# summarize history for loss
plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('result/model_loss.png')

from PIL import Image
im = Image.open('result/model_accuracy.png')
im.show()
im = Image.open('result/model_loss.png')
im.show()






