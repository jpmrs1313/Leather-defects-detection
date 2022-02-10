import string
import tensorflow as tf
import numpy as np
import cv2


def load_image(file_path: string, shape: tf.Tensor) -> tf.Tensor:
    """Load image 

    Parameters
    -----------
    file_path: image directory string
    shape: image shape tf.Tensor (height, width, depth)

    Returns
    -----------
    image: image tf.Tensor ([height, width, channels])
    """
    height, width, depth = shape

    # Read image bytes
    image = tf.io.read_file(file_path)
    # Load image
    image = tf.io.decode_image(image, channels=depth, dtype=tf.uint8)
    image = tf.image.resize(image, (height, width))
    image = tf.cast(image, tf.float32) / 255.0
    return image


def pre_process(image: np.ndarray) -> np.ndarray:
    """Image preprocessing function, at this moment just apply gaussian blur. However others techniques can be added to the pipeline

    Parameters
    -----------
    image: numpy array

    Returns
    -----------
    image: numpy array
    """

    image = cv2.GaussianBlur(image, (5, 5), 0)

    return image


def extract_patches(image: tf.Tensor, patch_shape: tf.Tensor) -> tf.Tensor:
    """Split an image in patches with specific shape

    Parameters
    -----------
    image: image tf.Tensor ([height, width, channels])
    shape: patches shape tf.Tensor (height, width, depth)
    Returns
    -----------
    patches: tf.Tensor ([number of patches, height, width ,depth])
    """

    height, width, depth = patch_shape

    # convert [height,width,depth] to [1,height,width,depth]
    image = tf.expand_dims(image, 0)

    # exctract patches without overlaping, just split the image in patches
    # to achieve that just keep the size and stride argument equal
    patches = tf.image.extract_patches(
        images=image,
        sizes=[1, height, width, 1],
        strides=[1, height, width, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )

    # Ex: convert [1, 8, 8, 32*32*3] to [64,32,32,3]
    patches = tf.reshape(patches, [-1, height, width, depth])
    return patches


def augment_using_ops(image: tf.Tensor) -> tf.Tensor:
    """Image augmentation - create new image using geometric transformations 

    Parameters
    -----------
    image: image tf.Tensor ([height, width, channels])
    Returns
    -----------
    image: image tf.Tensor ([height, width, channels])
    """

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_hue(image, 0.03)
    image = tf.image.random_brightness(image, 0.03)


    return image


def split_data(dataset: tf.Tensor,batch_size: int)-> tf.Tensor:
    """Split data in train 70%, validation 15% and test 15%

    Parameters
    -----------
    dataset:  tf.Tensor 
    Returns
    -----------
    train_dataset, val_dataset, test_dataset
    """
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = int(0.15 * len(dataset))

    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    val_dataset = test_dataset.skip(test_size)
    test_dataset = test_dataset.take(test_size)

    train_dataset = train_dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset
