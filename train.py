import glob
import tensorflow as tf
from tensorflow.keras import optimizers
from utils import load_image, pre_process, augment_using_ops, extract_patches, split_data
from networks import autoencoder
from options import Options

tf.random.set_seed(5)

if not Options().train_validate():
    exit()

# parse argument variables
cfg = Options().parse()

image_shape = (cfg.image_size, cfg.image_size, 3)
patch_shape = (cfg.patch_size, cfg.patch_size, 3)

# read all image file paths
image_paths = []
[image_paths.extend(glob.glob(cfg.train_data_dir + '/**/' + '*.' + e)) for e in ['png', 'jpg']]

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

if(cfg.patches == "True"):
    
    model_input_shape = patch_shape 

    # extract patches from images
    ds = ds.map(
        lambda image: (
            tf.py_function(func=extract_patches, inp=[
                image, patch_shape], Tout=tf.float32)
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # get number of patches obtained in one image
    elem = next(iter(ds))
    n_patches_per_image = len(elem)


    # remove batch -> ([245,64,32,32,3]) to [15680,32,32,3], images are saved in a list for the augment step
    ds = ds.unbatch().apply(tf.data.experimental.assert_cardinality(n_patches_per_image * n_images))
else:
    model_input_shape = image_shape   

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
train_dataset, val_dataset, test_dataset = split_data(ds,cfg.batch_size)

# construct our convolutional autoencoder
print("[INFO] building autoencoder...")
autoencoder = autoencoder(model_input_shape)

# train the convolutional autoencoder
autoencoder.fit(train_dataset,validation_data=val_dataset,epochs=20,batch_size=cfg.batch_size)

autoencoder.save("model2")
