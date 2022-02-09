import glob
import tensorflow as tf
from utils import load_image, pre_process, augment_using_ops, extract_patches
from networks import *


image_shape = (256, 256, 3)
patch_shape = (32, 32, 3)
BS = 32

path = r"C:/Users/jpmrs/OneDrive/Desktop/Dissertação/code/Unsupervised approach/Data/mvtec/leather"
train_path = path + "/train"
files = glob.glob(train_path + "/**/*.png", recursive=True)

ds = tf.data.Dataset.from_tensor_slices(files).shuffle(1024)

n_images = len(ds)

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

ds = ds.map(
    lambda image: (tf.numpy_function(
        func=pre_process, inp=[image], Tout=tf.float32)),
    num_parallel_calls=tf.data.AUTOTUNE,
)

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


ds = ds.unbatch().apply(
    tf.data.experimental.assert_cardinality(n_patches_per_image * n_images)
)
ds = ds.map(augment_using_ops,
            num_parallel_calls=tf.data.AUTOTUNE)  # .repeat(2)

train_size = int(0.7 * len(ds))
val_size = int(0.15 * len(ds))
test_size = int(0.15 * len(ds))

train_dataset = ds.take(train_size)
test_dataset = ds.skip(train_size)
val_dataset = test_dataset.skip(test_size)
test_dataset = test_dataset.take(test_size)

train_dataset = train_dataset.cache().batch(BS).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.cache().batch(BS).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.cache().batch(BS).prefetch(tf.data.AUTOTUNE)

# construct our convolutional autoencoder
print("[INFO] building autoencoder...")
autoencoder = autoencoder(patch_shape, 100)
optimizer = optimizers.Adam(learning_rate=2e-4, decay=1e-5)
autoencoder.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

# train the convolutional autoencoder
H = autoencoder.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    batch_size=BS,
)

autoencoder.save("model_test")
