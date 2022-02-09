import tensorflow as tf
import matplotlib.pyplot as plt
import glob
from sklearn.feature_extraction import image as extraction
import numpy as np


def load_image(file, shape):
    height, width, depth = shape
    # Read image bytes
    image = tf.io.read_file(file)
    # Load image
    image = tf.io.decode_png(image, channels=depth, dtype=tf.uint8)
    image = tf.image.resize(image, (height, width))
    image = tf.cast(image, tf.float32) / 255.0

    return image


def exctract_patches(image, shape):
    height, width, depth = shape
    image = tf.expand_dims(image, 0)

    patches = tf.image.extract_patches(
        images=image,
        sizes=[1, height, width, 1],
        strides=[1, height, width, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )
    patches = tf.reshape(patches, [-1, height, width, depth])

    return patches


def recon_im(patches: np.ndarray, image_shape):
    """Reconstruct the image from all patches.
        Patches are assumed to be square and overlapping depending on the patch_size. The image is constructed
         by filling in the patches from left to right, top to bottom, averaging the overlapping parts.
    Parameters
    -----------
    patches: 4D ndarray with shape (patch_number,patch_height,patch_width,channels)
        Array containing extracted patches. If the patches contain colour information,
        channels are indexed along the last dimension: RGB patches would
        have `n_channels=3`.
    Returns
    -----------
    reconstructedim: ndarray with shape (height, width, channels)
                      or ndarray with shape (height, width) if output image only has one channel
                    Reconstructed image from the given patches
    """
    image_height, image_width, image_depth = image_shape

    patch_size = patches.shape[1]  # patches assumed to be square

    # Assign output image shape based on patch sizes
    rows = ((image_height - patch_size) // patch_size) * \
        patch_size + patch_size
    cols = ((image_width - patch_size) // patch_size) * patch_size + patch_size

    reconim = np.zeros((rows, cols, image_depth))
    divim = np.zeros((rows, cols, image_depth))

    p_c = (
        cols - patch_size + patch_size
    ) / patch_size  # number of patches needed to fill out a row

    totpatches = patches.shape[0]
    initr, initc = 0, 0

    # extract each patch and place in the zero matrix and sum it with existing pixel values

    reconim[initr:patch_size, initc:patch_size] = patches[
        0
    ]  # fill out top left corner using first patch
    divim[initr:patch_size, initc:patch_size] = np.ones(patches[0].shape)

    patch_num = 1

    while patch_num <= totpatches - 1:
        initc = initc + patch_size
        reconim[initr: initr + patch_size, initc: patch_size + initc] += patches[
            patch_num
        ]
        divim[initr: initr + patch_size, initc: patch_size + initc] += np.ones(
            patches[patch_num].shape
        )

        if np.remainder(patch_num + 1, p_c) == 0 and patch_num < totpatches - 1:
            initr = initr + patch_size
            initc = 0
            reconim[initr: initr + patch_size, initc:patch_size] += patches[
                patch_num + 1
            ]
            divim[initr: initr + patch_size, initc:patch_size] += np.ones(
                patches[patch_num].shape
            )
            patch_num += 1
        patch_num += 1
    # Average out pixel values
    reconstructedim = reconim / divim

    return reconstructedim


image_shape = (256, 256, 3)
patch_shape = (32, 32, 3)
BS = 32


path = r"C:/Users/jpmrs/OneDrive/Desktop/Dissertação/code/Unsupervised approach/Data/mvtec/leather"
train_path = path + "/test"
files = glob.glob(train_path + "/**/*.png", recursive=True)

ds = tf.data.Dataset.from_tensor_slices(files).shuffle(1024)

n_images = len(ds)

ds = ds.map(
    lambda image: (
        tf.py_function(
            func=load_image,
            inp=[image, image_shape],
            Tout=tf.float32,
        )
    ),
    num_parallel_calls=tf.data.AUTOTUNE,
)

autoencoder = tf.keras.models.load_model("model_test")

for image in ds:

    patches = exctract_patches(image, patch_shape)
    predictions = autoencoder.predict(patches)
    inv = recon_im(predictions, image_shape)

    fig = plt.figure(figsize=(10, 7))
    rows = 1
    columns = 2
    fig.add_subplot(rows, columns, 1)

    plt.imshow(image)
    plt.axis("off")

    fig.add_subplot(rows, columns, 2)
    plt.imshow(inv)
    plt.axis("off")
    plt.show()
    break
