<<<<<<< HEAD
from tensorflow.keras import layers, optimizers
from tensorflow.keras.models import Model, Sequential
import tensorflow as tf

@tf.function
def ssim_loss(gt, y_pred, max_val=1.0):
    return 1 - tf.reduce_mean(tf.image.ssim(gt, y_pred, max_val=max_val))

def autoencoder(shape, latentDim=100):
    height, width, depth = shape
    encoder = Sequential(
        [
            layers.Flatten(input_shape=(height, width, depth)),
            layers.Dense(512),
            layers.LeakyReLU(),
            layers.Dropout(0.5),
            layers.Dense(256),
            layers.LeakyReLU(),
            layers.Dropout(0.5),
            layers.Dense(128),
            layers.LeakyReLU(),
            layers.Dropout(0.5),
            layers.Dense(64),
            layers.LeakyReLU(),
            layers.Dropout(0.5),
            layers.Dense(latentDim),
            layers.LeakyReLU(),
        ]
    )
    
    decoder = Sequential(
        [
            layers.Dense(64, input_shape=(latentDim,)),
            layers.LeakyReLU(),
            layers.Dropout(0.5),
            layers.Dense(128),
            layers.LeakyReLU(),
            layers.Dropout(0.5),
            layers.Dense(256),
            layers.LeakyReLU(),
            layers.Dropout(0.5),
            layers.Dense(512),
            layers.LeakyReLU(),
            layers.Dropout(0.5),
            layers.Dense(height * width * depth),
            layers.Activation("sigmoid"),
            layers.Reshape((height, width, depth)),
        ]
    )
    
    img = layers.Input(shape=(height, width, depth))
    latent_vector = encoder(img)
    output = decoder(latent_vector)
    
    model = Model(inputs=img, outputs=output)
    
    # input_img = layers.Input(shape=(height,width,depth))

    # h = layers.Conv2D(32, (4, 4), strides=2, activation=layers.LeakyReLU(alpha=0.2), padding='same')(input_img)
    # h = layers.Conv2D(32, (4, 4), strides=2, activation=layers.LeakyReLU(alpha=0.2), padding='same')(h)
    # # h = layers.Conv2D(32, (4, 4), strides=2, activation=layers.LeakyReLU(alpha=0.2), padding='same')(h)
    # h = layers.Conv2D(32, (3, 3), strides=1, activation=layers.LeakyReLU(alpha=0.2), padding='same')(h)
    # h = layers.Conv2D(64, (4, 4), strides=2, activation=layers.LeakyReLU(alpha=0.2), padding='same')(h)
    # h = layers.Conv2D(64, (3, 3), strides=1, activation=layers.LeakyReLU(alpha=0.2), padding='same')(h)
    # h = layers.Conv2D(128, (4, 4), strides=2, activation=layers.LeakyReLU(alpha=0.2), padding='same')(h)
    # h = layers.Conv2D(64, (3, 3), strides=1, activation=layers.LeakyReLU(alpha=0.2), padding='same')(h)
    # h = layers.Conv2D(32, (3, 3), strides=1, activation=layers.LeakyReLU(alpha=0.2), padding='same')(h)
    # encoded = layers.Conv2D(latentDim, (8, 8), strides=1, activation='linear', padding='valid')(h)

    # h = layers.Conv2DTranspose(32, (8, 8), strides=1, activation=layers.LeakyReLU(alpha=0.2), padding='valid')(encoded)
    # h = layers.Conv2D(64, (3, 3), strides=1, activation=layers.LeakyReLU(alpha=0.2), padding='same')(h)
    # h = layers.Conv2D(128, (3, 3), strides=1, activation=layers.LeakyReLU(alpha=0.2), padding='same')(h)
    # h = layers.Conv2DTranspose(64, (4, 4), strides=2, activation=layers.LeakyReLU(alpha=0.2), padding='same')(h)
    # h = layers.Conv2D(64, (3, 3), strides=1, activation=layers.LeakyReLU(alpha=0.2), padding='same')(h)
    # h = layers.Conv2DTranspose(32, (4, 4), strides=2, activation=layers.LeakyReLU(alpha=0.2), padding='same')(h)
    # h = layers.Conv2D(32, (3, 3), strides=1, activation=layers.LeakyReLU(alpha=0.2), padding='same')(h)
    # h = layers.Conv2DTranspose(32, (4, 4), strides=2, activation=layers.LeakyReLU(alpha=0.2), padding='same')(h)
    # # h = layers.Conv2DTranspose(32, (4, 4), strides=2, activation=layers.LeakyReLU(alpha=0.2), padding='same')(h)

    # decoded = layers.Conv2DTranspose(depth, (4, 4), strides=2, activation='sigmoid', padding='same')(h)

    # model=Model(input_img, decoded)
    optimizer = optimizers.Adam(learning_rate=2e-4)

    model.compile(optimizer=optimizer, loss=ssim_loss, metrics=["mae"])

    return model
=======
from tensorflow.keras import layers, optimizers
from tensorflow.keras.models import Model, Sequential
import tensorflow as tf

@tf.function
def ssim_loss(gt, y_pred, max_val=1.0):
    return 1 - tf.reduce_mean(tf.image.ssim(gt, y_pred, max_val=max_val))

def autoencoder(shape, latentDim=100):
    height, width, depth = shape
    encoder = Sequential(
        [
            layers.Flatten(input_shape=(height, width, depth)),
            layers.Dense(512),
            layers.LeakyReLU(),
            layers.Dropout(0.5),
            layers.Dense(256),
            layers.LeakyReLU(),
            layers.Dropout(0.5),
            layers.Dense(128),
            layers.LeakyReLU(),
            layers.Dropout(0.5),
            layers.Dense(64),
            layers.LeakyReLU(),
            layers.Dropout(0.5),
            layers.Dense(latentDim),
            layers.LeakyReLU(),
        ]
    )
    
    decoder = Sequential(
        [
            layers.Dense(64, input_shape=(latentDim,)),
            layers.LeakyReLU(),
            layers.Dropout(0.5),
            layers.Dense(128),
            layers.LeakyReLU(),
            layers.Dropout(0.5),
            layers.Dense(256),
            layers.LeakyReLU(),
            layers.Dropout(0.5),
            layers.Dense(512),
            layers.LeakyReLU(),
            layers.Dropout(0.5),
            layers.Dense(height * width * depth),
            layers.Activation("sigmoid"),
            layers.Reshape((height, width, depth)),
        ]
    )
    
    img = layers.Input(shape=(height, width, depth))
    latent_vector = encoder(img)
    output = decoder(latent_vector)
    
    model = Model(inputs=img, outputs=output)
    
    # input_img = layers.Input(shape=(height,width,depth))

    # h = layers.Conv2D(32, (4, 4), strides=2, activation=layers.LeakyReLU(alpha=0.2), padding='same')(input_img)
    # h = layers.Conv2D(32, (4, 4), strides=2, activation=layers.LeakyReLU(alpha=0.2), padding='same')(h)
    # # h = layers.Conv2D(32, (4, 4), strides=2, activation=layers.LeakyReLU(alpha=0.2), padding='same')(h)
    # h = layers.Conv2D(32, (3, 3), strides=1, activation=layers.LeakyReLU(alpha=0.2), padding='same')(h)
    # h = layers.Conv2D(64, (4, 4), strides=2, activation=layers.LeakyReLU(alpha=0.2), padding='same')(h)
    # h = layers.Conv2D(64, (3, 3), strides=1, activation=layers.LeakyReLU(alpha=0.2), padding='same')(h)
    # h = layers.Conv2D(128, (4, 4), strides=2, activation=layers.LeakyReLU(alpha=0.2), padding='same')(h)
    # h = layers.Conv2D(64, (3, 3), strides=1, activation=layers.LeakyReLU(alpha=0.2), padding='same')(h)
    # h = layers.Conv2D(32, (3, 3), strides=1, activation=layers.LeakyReLU(alpha=0.2), padding='same')(h)
    # encoded = layers.Conv2D(latentDim, (8, 8), strides=1, activation='linear', padding='valid')(h)

    # h = layers.Conv2DTranspose(32, (8, 8), strides=1, activation=layers.LeakyReLU(alpha=0.2), padding='valid')(encoded)
    # h = layers.Conv2D(64, (3, 3), strides=1, activation=layers.LeakyReLU(alpha=0.2), padding='same')(h)
    # h = layers.Conv2D(128, (3, 3), strides=1, activation=layers.LeakyReLU(alpha=0.2), padding='same')(h)
    # h = layers.Conv2DTranspose(64, (4, 4), strides=2, activation=layers.LeakyReLU(alpha=0.2), padding='same')(h)
    # h = layers.Conv2D(64, (3, 3), strides=1, activation=layers.LeakyReLU(alpha=0.2), padding='same')(h)
    # h = layers.Conv2DTranspose(32, (4, 4), strides=2, activation=layers.LeakyReLU(alpha=0.2), padding='same')(h)
    # h = layers.Conv2D(32, (3, 3), strides=1, activation=layers.LeakyReLU(alpha=0.2), padding='same')(h)
    # h = layers.Conv2DTranspose(32, (4, 4), strides=2, activation=layers.LeakyReLU(alpha=0.2), padding='same')(h)
    # # h = layers.Conv2DTranspose(32, (4, 4), strides=2, activation=layers.LeakyReLU(alpha=0.2), padding='same')(h)

    # decoded = layers.Conv2DTranspose(depth, (4, 4), strides=2, activation='sigmoid', padding='same')(h)

    # model=Model(input_img, decoded)
    optimizer = optimizers.Adam(learning_rate=2e-4, decay=1e-5)

    model.compile(optimizer=optimizer, loss=ssim_loss, metrics=["mae"])

    return model
>>>>>>> 32b79daf70cddd95b178c0acb52a8790bb856392
