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

    optimizer = optimizers.Adam(learning_rate=2e-4, decay=1e-5)
    
    model.compile(optimizer=optimizer, loss=ssim_loss, metrics=["mae"])

    return model
