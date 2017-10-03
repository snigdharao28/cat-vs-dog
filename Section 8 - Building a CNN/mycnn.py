# -*- coding: utf-8 -*-



# Part 1- Building the CNN
#sequential package is used to initialize our NN
from keras.models import Sequential
#used to make the convolution step to add the convolution layers
from keras.layers import Convolution2D
#used to proceed to pooling step
from keras.layers import MaxPooling2D
#used to flatten or convert the pool values to an array
from keras.layers import Flatten
#used to add the fully connected layers of the ANN
from keras.layers import Dense

# initializing the CNN

classifier = Sequential()

# step 1 - convolution

classifier.add(Convolution2D(32, (3, 3), input_shape= (64,64,3), activation= 'relu'))

#part 2 - pooling

classifier.add(MaxPooling2D(pool_size=(2, 2)))

# second convolutional layer
classifier.add(Convolution2D(32, (3, 3), activation= 'relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

#part 3- flattening

classifier.add(Flatten())

#part 4 - full connection

classifier.add(Dense(units = 128, activation= 'relu'))
classifier.add(Dense(units = 1, activation= 'sigmoid'))

# compiling the CNN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


#part 2 - fitting our CNN to the images
from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(
                            training_set,
                            steps_per_epoch=2000,
                            epochs=5,
                            validation_data=test_set,
                            validation_steps=400)


# making a new prediction
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg',target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] ==1:
    prediction = 'dog'
else:
    prediction = 'cat'


#redo this code section
test_image1 = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg',target_size=(64,64))
test_image1 = image.img_to_array(test_image1)
test_image1 = np.expand_dims(test_image1, axis = 0)
result1 = classifier.predict(test_image1)
training_set.class_indices
if result1[0][0] == 1:
    pred = 'dog'
else:
    pred = 'cat'