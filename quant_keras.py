
from image_processing import preprocess_image, resize_and_rescale_image
from tensorflow.lite.python import lite_constants as constants
import io
import os
import PIL
import numpy as np

from functools import partial
try:
    # %tensorflow_version only exists in Colab.
    #import tensorflow.compat.v2 as tf
    import tensorflow as tf

except Exception:
    pass

import tensorflow_model_optimization as tfmot


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# tf.config.gpu.set_per_process_memory_fraction(0.4)
#import inspect
#c = inspect.getfile(tf.lite.TFLiteConverter.__class__)
# print(c)

# tf.enable_v2_behavior()

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
    image = preprocess_image(
        image, 96, 96, is_training=is_training, do_grayscale=True)
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


def representative_dataset_gen_old():
    #record_iterator = tf.python_io.tf_record_iterator(path='dataset/val.record-00000-of-00010')
    record_iterator = tf.compat.v1.io.tf_record_iterator(
        path='dataset/val.record-00000-of-00010')

    #dataset = tf.data.Dataset.list_files("dataset/val.record-*")
    #record_iterator = tf.data.TFRecordDataset(dataset)

    count = 0
    for string_record in record_iterator:
        #print(f"STRING {string_record}")
        example = tf.train.Example()
        example.ParseFromString(string_record)
        image_stream = io.BytesIO(
            example.features.feature['image/encoded'].bytes_list.value[0])
        image = PIL.Image.open(image_stream)
        image = image.resize((96, 96))
        image = image.convert('L')
        array = np.array(image)
        array = np.expand_dims(array, axis=2)
        array = np.expand_dims(array, axis=0)
        array = ((array / 127.5) - 1.0).astype(np.float32)
        #array = ((array / 127.5) - 1.0).astype(np.int8)
        yield([array])
        count += 1
        if count > 300:
            break


def get_representative_dataset_gen(dataset, num_steps=2000):

    def representative_dataset_gen():
        """Generates representative dataset for quantized."""
        for image, _ in dataset.take(num_steps):
            yield [image]

    return representative_dataset_gen


"""
with tfmot.quantization.keras.quantize_scope():
    #q_aware_model = tf.keras.models.load_model('keras_birds-quantized_model-final.h5')
    q_aware_model = tf.keras.models.load_model('keras_birds-model-final.h5')
    converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
    converter.representative_dataset = representative_dataset_gen
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    quantized_tflite_model = converter.convert()

    open("keras_q1.tflite", "wb").write(quantized_tflite_model)
"""


m = "keras_birds/keras_birds_mobilenet_v2_model-final.h5"

model = tf.keras.models.load_model(m)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
#converter.representative_dataset = representative_dataset_gen
quantization_steps = 2000
batch_size = 32
ds_valid = get_dataset('dataset', 'val.record', batch_size)
converter.representative_dataset = tf.lite.RepresentativeDataset(
    get_representative_dataset_gen(ds_valid, quantization_steps))
#converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = set(
    [tf.lite.OpsSet.TFLITE_BUILTINS_INT8])
converter.target_spec.supported_types = set([constants.INT8])
converter.inference_type = constants.INT8
converter.experimental_new_converter = True

#converter.inference_input_type = tf.uint8
#converter.inference_output_type = tf.uint8
#converter.inference_type = constants.INT8
converter.inference_input_type = constants.INT8
converter.inference_output_type = constants.INT8

quantized_tflite_model = converter.convert()
open(m.replace(".h5", ".tflite"), "wb").write(quantized_tflite_model)

a = str.encode(str(converter._get_base_converter_args()))
open("qargs", "wb").write(a)
a = str.encode(str(converter._debug_info))
open("qadebuginfo", "wb").write(a)

# print(converter.experimental_new_converter)
# print(converter.experimental_new_quantizer)
