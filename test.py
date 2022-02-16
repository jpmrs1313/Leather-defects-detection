import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import numpy as np
import cv2
from utils import load_image, pre_process, get_residual_map, extract_patches, recon_im
from networks import ssim_loss
from options import Options
from sklearn.metrics import roc_auc_score, auc

# parse argument variables
cfg = Options().parse()

image_shape = (cfg.image_size, cfg.image_size,3)
patch_shape = (cfg.patch_size, cfg.patch_size, 3)

# read all image file paths
image_paths = []
[image_paths.extend(glob.glob(cfg.test_data_dir + '/**/' + '*.' + e)) for e in ['png', 'jpg']]

ground_truth_paths  = []
[ground_truth_paths.extend(glob.glob(cfg.ground_truth_data_dir + '/**/' + '*.' + e)) for e in ['png', 'jpg']]

ds = tf.data.Dataset.from_tensor_slices((image_paths,ground_truth_paths)).shuffle(1024)

n_images = len(ds)

# load images from paths
ds = ds.map(
    lambda path1, path2: (
        tf.py_function(
            func=load_image,
            inp=[path1, image_shape],
            Tout=tf.float32,
        ),
         tf.py_function(
            func=load_image,
            inp=[path2, image_shape],
            Tout=tf.float32,
        )
    ),
    num_parallel_calls=tf.data.AUTOTUNE,
)

# pre-process images
ds = ds.map(
    lambda image, ground_truth: (tf.numpy_function(
        func=pre_process, inp=[image], Tout=tf.float32), ground_truth),
    num_parallel_calls=tf.data.AUTOTUNE,
)

autoencoder = tf.keras.models.load_model("model2", compile=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, decay=1e-5)
autoencoder.compile(optimizer=optimizer, loss=ssim_loss, metrics=["mae"])

for image,y_true in ds:

    ssim_residual_map,l1_residual_map = get_residual_map(image,autoencoder)
    y_pred = np.zeros((256,256), np.uint8)
    y_pred[ssim_residual_map > 0.23572287082672116] = 1
    #y_pred[l1_residual_map > 0.00021582189743639965] = 1

    fig = plt.figure(figsize=(10, 7))
    rows = 1
    columns = 2
    fig.add_subplot(rows, columns, 1)

    plt.imshow(y_pred,cmap='gray')
    plt.title("prediction")
    plt.axis("off")

    fig.add_subplot(rows, columns, 2)
    plt.imshow(y_true,cmap='gray')
    plt.title("ground_truth")
    plt.axis("off")
    plt.show()





