import tensorflow as tf
from tensorflow.keras.applications.xception import preprocess_input

def load_and_preprocess_image(image_path, label, is_training=True):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [299, 299])
    img = preprocess_input(img)
    if is_training:
        img = tf.image.random_flip_left_right(img)
    return img, label

def data_generator(df, batch_size=32, is_training=True):
    labels = df['label'].apply(lambda x: 1 if x == 'real' else 0).values
    image_paths = df['image_path'].values
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(
        lambda x, y: load_and_preprocess_image(x, y, is_training),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    if is_training:
        dataset = dataset.shuffle(buffer_size=500)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset