import os
from functools import partial
import time

import tensorflow as tf

from image_processing import preprocess_image, resize_and_rescale_image
from tensorflow.keras import backend as K
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

NUM_DATA_WORKERS = 8


def decode_jpeg(image_buffer, scope=None):
    """Decode a JPEG string into one 3-D float image Tensor.
    Args:
        image_buffer: scalar string Tensor.
        scope: Optional scope for name_scope.
    Returns:
        3-D float Tensor with values ranging from [0, 1).
    """
    with tf.name_scope(name='decode_jpeg'):
        # Decode the string as an RGB JPEG.
        # Note that the resulting image contains an unknown height
        # and width that is set dynamically by decode_jpeg. In other
        # words, the height and width of image is unknown at compile-i
        # time.
        image = tf.image.decode_jpeg(image_buffer, channels=3)

        # After this point, all image pixels reside in [0,1)
        # until the very end, when they're rescaled to (-1, 1).
        # The various adjust_* ops all require this range for dtype
        # float.
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image


def _parse_fn(example_serialized, is_training):
    """Helper function for parse_fn_train() and parse_fn_valid()
    Each Example proto (TFRecord) contains the following fields:
    image/height: 462
    image/width: 581
    image/colorspace: 'RGB'
    image/channels: 3
    image/class/label: 615
    image/class/synset: 'n03623198'
    image/class/text: 'knee pad'
    image/format: 'JPEG'
    image/filename: 'ILSVRC2012_val_00041207.JPEG'
    image/encoded: <JPEG encoded string>
    Args:
        example_serialized: scalar Tensor tf.string containing a
        serialized Example protocol buffer.
    Returns:
        image_buffer: Tensor tf.string containing the contents of
        a JPEG file.
        label: Tensor tf.int32 containing the label.
        text: Tensor tf.string containing the human-readable label.
    """
    feature_map = {
        'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string,
                                               default_value=''),
        'image/class/label': tf.io.FixedLenFeature([], dtype=tf.int64,
                                                   default_value=-1),
        'image/class/label_text': tf.io.FixedLenFeature([], dtype=tf.string,
                                                        default_value=''),
    }
    parsed = tf.io.parse_single_example(example_serialized, feature_map)
    image = decode_jpeg(parsed['image/encoded'])
    # if config.DATA_AUGMENTATION:
    #    image = preprocess_image(image, 224, 224, is_training=is_training)
    # else:
    #    image = resize_and_rescale_image(image, 224, 224)
    image = preprocess_image(image, 96, 96, is_training=is_training, do_grayscale=True)
    #image = tf.image.rgb_to_grayscale(image)
    # The label in the tfrecords is 1~1000 (0 not used).
    # So I think the minus 1 is needed below.
    label = tf.one_hot(parsed['image/class/label'] - 1, 2, dtype=tf.float32)
    return (image, label)


def get_dataset(tfrecords_dir, subset, batch_size):
    """Read TFRecords files and turn them into a TFRecordDataset."""
    files = tf.io.matching_files(os.path.join(tfrecords_dir, '%s-*' % subset))
    print("files", files, tfrecords_dir, subset)
    shards = tf.data.Dataset.from_tensor_slices(files)
    shards = shards.shuffle(tf.cast(tf.shape(files)[0], tf.int64))
    shards = shards.repeat()
    dataset = shards.interleave(tf.data.TFRecordDataset, cycle_length=4)
    dataset = dataset.shuffle(buffer_size=8192)
    parser = partial(
        _parse_fn, is_training=True if subset == 'train' else False)
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            map_func=parser,
            batch_size=batch_size,
            num_parallel_calls=NUM_DATA_WORKERS))
    dataset = dataset.prefetch(batch_size)
    return dataset

CHANNELS = 1
vww_input = Input(shape=(96,96,CHANNELS))    # let us say this new InputLayer

# create the base pre-trained model
base_model = MobileNet(input_tensor=vww_input, alpha=0.25, classes=2, weights=None, include_top=False)

"""
# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(2, activation='softmax')(x)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional layers
# for layer in base_model.layers:
#    layer.trainable = False

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
"""
model = base_model

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# dataset

batch_size = 32
# get training and validation data
ds_train = get_dataset('dataset', 'train.record', batch_size)
ds_valid = get_dataset('dataset', 'val.record', batch_size)

# train the model on the new data for a few epochs
#model.fit_generator(, ,)

model_ckpt = tf.keras.callbacks.ModelCheckpoint(
        os.path.join("keras_birds", "mobilenetv1-kmod") + '-ckpt-{epoch:03d}.h5',
        monitor='val_loss',
        save_best_only=True)
tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir='{}/{}'.format("keras_log", time.time()))

model.fit(
        x=ds_train,
        #steps_per_epoch=1281167 // batch_size,
        steps_per_epoch=15000//batch_size,
        validation_data=ds_valid,
        #validation_steps=50000 // batch_size,
        validation_steps=1500,
        #callbacks=[lrate, model_ckpt, tensorboard],
        callbacks=[model_ckpt, tensorboard],
        # The following doesn't seem to help in terms of speed.
        # use_multiprocessing=True, workers=4,
        epochs=epochs)

save_name = "keras_birds_mobilenet_v1"
SAVE_DIR = 'keras_birds'
# training finished
model.save('{}/{}_model-final.h5'.format(SAVE_DIR, save_name))

