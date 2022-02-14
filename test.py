import tensorflow as tf
import matplotlib.pyplot as plt
import glob
from utils import load_image, pre_process, extract_patches, recon_im,get_residual_map
from networks import ssim_loss
from options import Options
from tensorflow.keras import optimizers
import numpy as np


# parse argument variables
cfg = Options().parse()

if(cfg.grey_scale == "True"):
    image_shape = (cfg.image_size, cfg.image_size,1)
    patch_shape = (cfg.patch_size, cfg.patch_size,1)
else:
    image_shape = (cfg.image_size, cfg.image_size, 3)
    patch_shape = (cfg.patch_size, cfg.patch_size, 3)

# read all image file paths
image_paths = []
[image_paths.extend(glob.glob(cfg.test_data_dir + '/**/' + '*.' + e)) for e in ['png', 'jpg']]

ds = tf.data.Dataset.from_tensor_slices(image_paths).shuffle(1024)

n_images = len(ds)

# load images from paths
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

# pre-process images
ds = ds.map(
    lambda image: (tf.numpy_function(
        func=pre_process, inp=[image,cfg.grey_scale], Tout=tf.float32)),
    num_parallel_calls=tf.data.AUTOTUNE,
)

autoencoder = tf.keras.models.load_model("model2", compile=False)
optimizer = optimizers.Adam(learning_rate=2e-4, decay=1e-5)
autoencoder.compile(optimizer=optimizer, loss=ssim_loss, metrics=["mae"])

if(cfg.patches=="True"):
    for image in ds:
        patches =  extract_patches(image, patch_shape)
        predictions = autoencoder.predict(patches)
        result = recon_im(predictions, image_shape)

        fig = plt.figure(figsize=(10, 7))
        rows = 1
        columns = 2
        fig.add_subplot(rows, columns, 1)

        plt.imshow(image,cmap='gray')
        plt.axis("off")

        fig.add_subplot(rows, columns, 2)
        plt.imshow(result,cmap='gray')
        plt.axis("off")
        plt.show()
else:
    results = autoencoder.predict(ds)
    for result, image in zip(results,ds):
        ssim_residual_map = get_residual_map(image,autoencoder, cfg)

        depr_mask = np.ones((256,256)) * 0.2
        depr_mask[5:256-5, 5:256-5] = 1

        ssim_residual_map *= depr_mask

        mask = np.zeros((256,256))
        mask[ssim_residual_map > 0.2488967776298523] = 1
        mask[ssim_residual_map <= 0.2488967776298523] = 0
     
        fig = plt.figure(figsize=(10, 7))
        rows = 1
        columns = 2
        fig.add_subplot(rows, columns, 1)

        plt.imshow(image,cmap='gray')
        plt.axis("off")

        fig.add_subplot(rows, columns, 2)
        plt.imshow(mask,cmap='gray')
        plt.axis("off")
        plt.show()



