from keras.models import *
from keras.layers import *
from keras.optimizers import *

IMG_SIZE = 512


def unet(pretrained_weights=None, input_size=(IMG_SIZE, IMG_SIZE, 3), num_class=2):
    inputs = Input(input_size)
    conv1 = Conv2D(64, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, (1, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization(momentum=0.9)(conv1)
    conv1 = Conv2D(64, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = Conv2D(64, (1, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = BatchNormalization(momentum=0.9)(pool1)
    conv2 = Conv2D(128, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = Conv2D(128, (1, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization(momentum=0.9)(conv2)
    conv2 = Conv2D(128, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = Conv2D(128, (1, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = BatchNormalization(momentum=0.9)(pool2)
    conv3 = Conv2D(256, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = Conv2D(256, (1, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization(momentum=0.9)(conv3)
    conv3 = Conv2D(256, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = Conv2D(256, (1, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = BatchNormalization(momentum=0.9)(pool3)
    conv4 = Conv2D(512, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = Conv2D(512, (1, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization(momentum=0.9)(conv4)
    conv4 = Conv2D(512, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = Conv2D(512, (1, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = BatchNormalization(momentum=0.9)(pool4)
    conv5 = Conv2D(1024, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = Conv2D(1024, (1, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization(momentum=0.9)(conv5)
    conv5 = Conv2D(1024, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = Conv2D(1024, (1, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))

    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = BatchNormalization(momentum=0.9)(merge6)
    conv6 = Conv2D(512, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = Conv2D(512, (1, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization(momentum=0.9)(conv6)
    conv6 = Conv2D(512, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = Conv2D(512, (1, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = BatchNormalization(momentum=0.9)(merge7)
    conv7 = Conv2D(256, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = Conv2D(256, (1, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization(momentum=0.9)(conv7)
    conv7 = Conv2D(256, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = Conv2D(256, (1, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization(momentum=0.9)(conv7)
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = BatchNormalization(momentum=0.9)(merge8)
    conv8 = Conv2D(128, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = Conv2D(128, (1, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization(momentum=0.9)(conv8)
    conv8 = Conv2D(128, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = Conv2D(128, (1, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization(momentum=0.9)(conv8)
    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = BatchNormalization(momentum=0.9)(merge9)
    conv9 = Conv2D(64, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(64, (1, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization(momentum=0.9)(conv9)
    conv9 = Conv2D(64, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(64, (1, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization(momentum=0.9)(conv9)
    conv9 = Conv2D(num_class, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization(momentum=0.9)(conv9)
    if num_class == 2:
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
        loss_function = 'binary_crossentropy'
    else:
        conv10 = Conv2D(num_class, 1, activation='relu')(conv9)
        loss_function = 'mse'
    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss=loss_function, metrics=["accuracy"])
    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model

# def focal_loss(gamma=2., alpha=.25):
#     def focal_loss_fixed(y_true, y_pred):
#         pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#         pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
#         return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon()+pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
#     return focal_loss_fixed

# def dice_coef(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# def dice_coef_loss(y_true, y_pred):
#     return -dice_coef(y_true, y_pred)
