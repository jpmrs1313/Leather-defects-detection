import glob
import tensorflow as tf
from utils import load_image, pre_process, augment_using_ops, split_data, get_threshold
from networks import autoencoder
from options import Options

tf.random.set_seed(5)

if not Options().train_validate():
    exit()

# parse argument variables
cfg = Options().parse()

image_shape = (cfg.image_size, cfg.image_size, 3)

# read all image file paths
image_paths = []

[image_paths.extend(glob.glob("C:/Users/jpmrs/OneDrive/Desktop/Dissertação/code/Data/mvtec/leather/train" + '/**/' + '*.' + e)) for e in ['png', 'jpg']]
# create tf.Data with image paths and shuffle them
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
        func=pre_process, inp=[image], Tout=tf.float32)),
    num_parallel_calls=tf.data.AUTOTUNE,
)

if(cfg.augmentation == "True"):
    # image augmentation
    ds = ds.map(
        lambda image: (
            tf.py_function(
                func=augment_using_ops,
                inp=[image],
                Tout=tf.float32,
            )
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).repeat(cfg.augmentation_iterations)  # number of augmentation iterations

# from image, create (image,image)->(element and label)
ds = ds.map(
    lambda image: (
        image, image
    ),
    num_parallel_calls=tf.data.AUTOTUNE,
)

# split data in train, validate and test
train_dataset, val_dataset, threshold_dataset = split_data(ds,cfg.batch_size)

# construct our convolutional autoencoder
print("[INFO] building autoencoder...")
autoencoder = autoencoder(image_shape, 100)

# train the convolutional autoencoder
autoencoder.fit(train_dataset,validation_data=val_dataset,epochs=20,batch_size=cfg.batch_size)
autoencoder.save("model2")

ssim_threshold=get_threshold(threshold_dataset,autoencoder,cfg)

print(ssim_threshold)