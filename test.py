<<<<<<< HEAD
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import numpy as np
from utils import *
from networks import *
from sklearn.metrics import roc_auc_score
import segmentation_models as sm
from options import Options

# parse argument variables
cfg = Options().parse()

image_shape = (cfg.image_size, cfg.image_size, 1)
patch_shape = (cfg.patch_size,cfg.patch_size,1)

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
            inp=[path1,image_shape],
            Tout=tf.float32,
        ),
         tf.py_function(
            func=load_mask,
            inp=[path2,image_shape],
            Tout=tf.float32,
        )
    ),
    num_parallel_calls=tf.data.AUTOTUNE,
)

#pre-process
ds = ds.map(
    lambda image,ground_truth: (
        tf.numpy_function(func=pre_process, inp=[image], Tout=tf.float32),
        ground_truth
        ),
    num_parallel_calls=tf.data.AUTOTUNE,
)

IOU=sm.metrics.IOUScore()
FScore=sm.metrics.FScore()

y_true=[]
y_pred=[]
i=0

if(cfg.patches=="True"):
    #extract patches from images
    ds = ds.map(
        lambda image, ground_truth: (
            tf.py_function(func=extract_patches, inp=[
                image, patch_shape], Tout=tf.float32),
            ground_truth
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    autoencoder = autoencoder(patch_shape, 100)
    autoencoder.load_weights("checkpoints/model1_patches/")
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, decay=1e-5)
    autoencoder.compile(optimizer=optimizer, loss=ssim_loss, metrics=["mae"])

    for patches,ground_truth in ds:
        ground_truth = ground_truth.numpy()
        if(not np.any(ground_truth)): continue
        
        patches_pred=[]
        
        for patch in patches:
            ssim_residual_map, l1_residual_map = get_residual_map(patch,autoencoder)
            image_pred = np.zeros((cfg.patch_size,cfg.patch_size), np.uint8)
            
            #model1 0.49108871459960923     #model2 0.13881855189800252
            image_pred[ssim_residual_map >0.49108871459960923] = 1
            patches_pred.append(image_pred)

        patches_pred=np.array(patches_pred)
        image_pred=recon_im(patches_pred, image_shape)

        #plot_two_images(image_pred,ground_truth)

        y_pred.append(image_pred)
        y_true.append(ground_truth)
else:   
    autoencoder = autoencoder(image_shape, 100)
    autoencoder.load_weights("checkpoints/model2/")
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, decay=1e-5)
    autoencoder.compile(optimizer=optimizer, loss=ssim_loss, metrics=["mae"])
    
    for image,ground_truth in ds:
        ground_truth = ground_truth.numpy()
        if(not np.any(ground_truth)): continue

        ssim_residual_map, _ = get_residual_map(image,autoencoder)
        image_pred = np.zeros((cfg.image_size,cfg.image_size), np.float32)
            
        #model1 0.5898075258731843    #model2 0.25707741141319285
        image_pred[ssim_residual_map > 0.25707741141319285] = 1 

        #plot_two_images(image_pred,ground_truth)

        y_pred.append(image_pred)
        y_true.append(ground_truth)

y_true=np.array(y_true)
y_pred=np.array(y_pred)

print("IOU "+ str(IOU(y_pred,y_true)))
print("FScore " + str(FScore(y_pred,y_true)))

y_pred = y_pred.flatten()
y_true = y_true.flatten()

print("AUC " + str(roc_auc_score(y_pred,y_true)))
=======
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





>>>>>>> 32b79daf70cddd95b178c0acb52a8790bb856392
