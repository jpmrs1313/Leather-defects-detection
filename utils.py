import string
import tensorflow as tf
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

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
    #height, width, depth = shape
    height, width, depth = shape
    # Read image bytes
    image = tf.io.read_file(file_path)
    # Load image
    image = tf.io.decode_image(image, dtype=tf.uint8)
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

def get_threshold(dataset,autoencoder,cfg):
    total_rec_ssim = []
    for batch, __ in dataset:
        for image in batch:
            ssim_residual_map= get_residual_map(image,autoencoder,cfg)
            total_rec_ssim.append(ssim_residual_map)
    total_rec_ssim = np.array(total_rec_ssim)
    ssim_threshold = float(np.percentile(total_rec_ssim, [98]))
    
    return ssim_threshold

def get_residual_map(image, autoencoder):
    image = tf.expand_dims(image, 0)
    result = autoencoder.predict(image)

    image=image.numpy()
    image = np.squeeze(image)
    result = np.squeeze(result)

    ssim_residual_map = ssim(image, result, win_size=11, full=True, channel_axis = -1)[1]
    ssim_residual_map = 1 - np.mean(ssim_residual_map, axis=2)

    return ssim_residual_map