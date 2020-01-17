from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization


def model():

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(1, 3), input_shape=(32, 32, 3)))
    model.add(Conv2D(32, kernel_size=(3, 1)))
    model.add(BatchNormalization(axis=-1))  # normalize
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 1)))
    model.add(Conv2D(64, (1, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # model.add(Dropout(0.5))

    model.add(Dense(4))
    model.add(Activation('softmax'))

    # model.summary()

    return model
