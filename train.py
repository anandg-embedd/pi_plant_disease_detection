# Importing all necessary libraries

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import BatchNormalization
from keras import backend as K
 
img_width, img_height = 224, 224

train_data_dir = 'dataset/train'
validation_data_dir = 'dataset/test'
nb_train_samples = None
nb_validation_samples = 4
epochs = 15
batch_size = 16
classes_number = 2 # Cars and Planes

def main():
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    # Initializing the CNN
    model = Sequential()
    # Convolution Step 1
    model.add(Convolution2D(96, 11, strides = (4, 4), padding = 'valid', input_shape=input_shape, activation = 'relu'))
    # Max Pooling Step 1
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))
    model.add(BatchNormalization())
    # Convolution Step 2
    model.add(Convolution2D(256, 11, strides = (1, 1), padding='valid', activation = 'relu'))
    # Max Pooling Step 2
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding='valid'))
    model.add(BatchNormalization())
    # Convolution Step 3
    model.add(Convolution2D(384, 3, strides = (1, 1), padding='valid', activation = 'relu'))
    model.add(BatchNormalization())
    # Convolution Step 4
    model.add(Convolution2D(384, 3, strides = (1, 1), padding='valid', activation = 'relu'))
    model.add(BatchNormalization())
    # Convolution Step 5
    model.add(Convolution2D(256, 3, strides=(1,1), padding='valid', activation = 'relu'))
    # Max Pooling Step 3
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))
    model.add(BatchNormalization())
    # Flattening Step
    model.add(Flatten())
    # Full Connection Step
    model.add(Dense(units = 4096, activation = 'relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Dense(units = 4096, activation = 'relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Dense(units = 1000, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(units = classes_number, activation = 'softmax'))


##    model = Sequential()
##    model.add(Conv2D(32, (2, 2), input_shape=input_shape))
##    model.add(Activation('relu'))
##    model.add(MaxPooling2D(pool_size=(2, 2)))
##    
##    model.add(Conv2D(32, (2, 2)))
##    model.add(Activation('relu'))
##    model.add(MaxPooling2D(pool_size=(2, 2)))
##    
##    model.add(Conv2D(64, (2, 2)))
##    model.add(Activation('relu'))
##    model.add(MaxPooling2D(pool_size=(2, 2)))
##    
##    model.add(Flatten())
##    model.add(Dense(64))
##    model.add(Activation('relu'))
##    model.add(Dropout(0.5))
##    # 1 For binary classification and CLASS_NUMBER for categorical classification
##    model.add(Dense(classes_number))
##    model.add(Activation('sigmoid'))

    # It can be 'categorical_crossentropy' or 'binary_crossentropy'
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical' # binary | categorical | sparse | input | None
    )
    
    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical' # binary | categorical | sparse | input | None
    )
    
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples, # batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples, # batch_size
        verbose=1,
        shuffle=True
    )

    model.save('model\categorical_vehicle_model_saved.h5')

if __name__ == '__main__':
    main()
