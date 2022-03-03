import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import cv2
import numpy as np
from utils import *
from networks import *
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import disk, opening
import segmentation_models as sm
from options import Options

# def per_region_overlap(ground_truths,image_preds):
#     N=0
#     Sum=0

#     for ground_truth, image_pred in zip(ground_truths,image_preds):
        
#         #get ground_truth decomposed components
#         components=label(ground_truth)
        
#         #np.unique(components)[1:] is used to get values in components except the background (the first value that is = 0)
#         for value in np.unique(components)[1:]:
#             print(value)
#             component = np.zeros((cfg.image_size,cfg.image_size), np.float32)     
#             component[components == value ] = 1 

#             #Pi denote the set of pixels predicted as anomalous 
#             Pi=get_Pi(image_pred)
            
#             #Ci,k denote the set of pixels marked as anomalous for a connected component
#             Cik=get_Cik(component)
            
#             #result (intersection bettween Pi e Cik) / Cik
#             result=

#             Sum =+ result
#             N=+1
#     return Sum/N

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
y_true,y_pred = [],[]

for x,y in ds:
    print(tf.reduce_max(x))
    print(tf.reduce_min(x))
    break

if(cfg.patches=="True"):
    for patches,ground_truth in ds:
        ground_truth = ground_truth.numpy()
        if(not np.any(ground_truth)): continue
        
        patches_pred=[]
        
        for patch in patches:
            ssim_residual_map, l1_residual_map = get_residual_map(patch,autoencoder)
            image_pred = np.zeros((cfg.patch_size,cfg.patch_size), np.uint8)
            
            #model1 0.48397845625877345     #model2 0.1419205433130264
            image_pred[ssim_residual_map >0.49108871459960923] = 1
            image_pred = clear_border(image_pred)
            patches_pred.append(image_pred)

        patches_pred=np.array(patches_pred)
        image_pred=recon_im(patches_pred, image_shape)
        
        kernel = disk(4)
        image_pred = opening(image_pred, kernel)
        #plot_images(image_pred,ground_truth)

        y_pred.append(image_pred)
        y_true.append(ground_truth)
else:   
    autoencoder = autoencoder(image_shape, 100)
    autoencoder.load_weights("checkpoints/model5/")
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, decay=1e-5)
    autoencoder.compile(optimizer=optimizer, loss=ssim_loss, metrics=["mae"])
    
    for image,ground_truth in ds:
        ground_truth = ground_truth.numpy()
        if(not np.any(ground_truth)): continue

        residual_map = get_residual_map(image,autoencoder,cfg.loss)
        image_pred = np.zeros((cfg.image_size,cfg.image_size), np.float32)

        #model1 0.5795271706581118    #model2 0.2585686755180361
        image_pred[residual_map >  0.1241170322895051 ] = 1 
        image_pred = clear_border(image_pred)

        kernel = disk(4)
        image_pred = opening(image_pred, kernel)
        
        #plot_images(image_pred,ground_truth)

        y_pred.append(image_pred)
        y_true.append(ground_truth)

y_true=np.array(y_true)
y_pred=np.array(y_pred)


print("IOU "+ str(IOU(y_pred,y_true)))
print("FScore " + str(FScore(y_pred,y_true)))

y_pred = y_pred.flatten()
y_true = y_true.flatten()

print("AUC PR " + str(average_precision_score(y_pred,y_true)))
print("AUC ROC " + str(roc_auc_score(y_pred,y_true)))
