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

def pre_process(image: np.ndarray, greyscale) -> np.ndarray:
    """Image preprocessing function, at this moment just apply gaussian blur. However others techniques can be added to the pipeline

    Parameters
    -----------
    image: numpy array
    greyscale: str "True" or "False"

    Returns
    -----------
    image: numpy array
    """
    greyscale = greyscale.decode("utf-8") 

    if(greyscale == "True"): 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    if(greyscale == "True"): 
        image = image[:, :, np.newaxis]

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

    # height, width, depth = patch_shape
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
    #total_rec_l1 = []
    for batch, __ in dataset:
        for image in batch:
            ssim_residual_map, l1_residual_map = get_residual_map(image,autoencoder,cfg)
            total_rec_ssim.append(ssim_residual_map)
            #total_rec_l1.append(l1_residual_map)
    total_rec_ssim = np.array(total_rec_ssim)
    total_rec_l1 = np.array(total_rec_l1)
    ssim_threshold = float(np.percentile(total_rec_ssim, [98]))
    #l1_threshold = float(np.percentile(total_rec_l1, [98]))
    
    return ssim_threshold #l1_threshold

def get_residual_map(image, autoencoder,cfg):

    if(cfg.patches=="True"):

        if(cfg.grey_scale == "True"):
            image_shape = (cfg.image_size, cfg.image_size,1)
            patch_shape = (cfg.patch_size, cfg.patch_size,1)
        else:
            image_shape = (cfg.image_size, cfg.image_size, 3)
            patch_shape = (cfg.patch_size, cfg.patch_size, 3)

        patches =  extract_patches(image, patch_shape)
        predictions = autoencoder.predict(patches)
        result = recon_im(predictions, image_shape)
            
    else:
        image = tf.expand_dims(image, 0)
        result = autoencoder.predict(image)

    image=image.numpy()
    image = np.squeeze(image)
    result = np.squeeze(result)

    if(cfg.patches=="True"):
        ssim_residual_map = 1 - ssim(image, result, win_size=11, full=True)[1]
        #l1_residual_map = np.abs(image / 255. - result / 255.)
    else:
        ssim_residual_map = ssim(image, result, win_size=11, full=True, multichannel=True)[1]
        ssim_residual_map = 1 - np.mean(ssim_residual_map, axis=2)
        #l1_residual_map = np.mean(np.abs(image / 255. - result / 255.), axis=2)

    return ssim_residual_map #l1_residual_map