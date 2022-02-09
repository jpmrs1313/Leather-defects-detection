from tensorflow.keras import layers, optimizers
from tensorflow.keras.models import Model, Sequential


def autoencoder(shape, latentDim):
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

    return model
