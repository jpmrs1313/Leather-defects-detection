import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import numpy as np
import cv2
from utils import load_image, pre_process, get_residual_map
from networks import ssim_loss
from options import Options
from sklearn.metrics import roc_auc_score, auc

# parse argument variables
cfg = Options().parse()

image_shape = (cfg.image_size, cfg.image_size,3)

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

final_x, final_y = [], []

score = []
for image,y_true in ds:
    y_true=y_true.numpy().astype(np.uint8)
    ret2,y_true = cv2.threshold(y_true,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    y_true= y_true /255.0
    y_true=y_true.astype(np.uint8)
  
    ssim_residual_map = get_residual_map(image,autoencoder)
    y_pred = np.zeros((256,256), np.uint8)
    y_pred[ssim_residual_map > 0.19186024546623237] = 1
    
    print(np.unique(y_pred))
    print(np.unique(y_true))
    print(type(y_pred), y_pred.dtype)
    print(type(y_true), y_true.dtype)

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

    y_pred = y_pred.flatten()
    y_true = y_true.flatten()

    print(roc_auc_score(y_pred, y_true))


# for result, ds_element in zip(results,ds):
#     image, ground_truth = ds_element

#     ssim_residual_map = get_residual_map(image,autoencoder, cfg)
#     mask = np.zeros((256,256))
#     mask[ssim_residual_map > 0.19186024546623237] = 1
    
#     fig = plt.figure(figsize=(10, 7))
#     rows = 1
#     columns = 2
#     fig.add_subplot(rows, columns, 1)

#     plt.imshow(mask,cmap='gray')
#     plt.title("prediction")
#     plt.axis("off")

#     fig.add_subplot(rows, columns, 2)
#     plt.imshow(ground_truth,cmap='gray')
#     plt.title("ground_truth")
#     plt.axis("off")
#     plt.show()



