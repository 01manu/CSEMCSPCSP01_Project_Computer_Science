import os
import tensorflow as tf

# image size for EfficientNet/MobileNet
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def decode_image(path, label=None, augment=False):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)  # scale 0–1

    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.15)
        img = tf.image.random_contrast(img, 0.8, 1.2)

    if label is None:
        return img
    return img, tf.one_hot(label, depth=5)

def make_dataset(df, base_dir, augment=False, shuffle=False):
    image_paths = df["image_id"].apply(lambda x: os.path.join(base_dir, x)).values
    labels = df["label"].values

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if shuffle:
        ds = ds.shuffle(len(df))

    ds = ds.map(lambda p, l: decode_image(p, l, augment),
                num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds
