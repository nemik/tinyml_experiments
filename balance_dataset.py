import tensorflow as tf
import glob

tf.executing_eagerly()

files = glob.glob("dataset/train.record-*")
raw_image_dataset = tf.data.TFRecordDataset(files)

new_ds = tf.data.TFRecordDataset("new.tfrecord")

# Create a dictionary describing the features.
image_feature_description = {
    # 'height': tf.io.FixedLenFeature([], tf.int64),
    # 'width': tf.io.FixedLenFeature([], tf.int64),
    # 'depth': tf.io.FixedLenFeature([], tf.int64),
    # 'label': tf.io.FixedLenFeature([], tf.int64),
    # 'image_raw': tf.io.FixedLenFeature([], tf.string),
    'image/class/label': tf.io.FixedLenFeature([], tf.int64),
}


def _parse_image_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)

# print(parsed_image_dataset)


def get_parsed():
    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
    for raw_record in parsed_image_dataset.take(100):
        # example = tf.train.Example()
        # example.ParseFromString(raw_record.numpy())
        # print(example)
        print(raw_record['image/class/label'].numpy())


def get_raw():
    for raw_record in raw_image_dataset.take(10):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        print(example)

# get_parsed()

# get_raw()

can_write = True

record_file = 'dataset/balanced_birds.tfrecords'
with tf.io.TFRecordWriter(record_file) as writer:
    for raw_record in raw_image_dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        label = example.features.feature['image/class/label'].int64_list.value[0]
        if label == 0:
            #background
            if can_write:
                writer.write(raw_record)
                can_write = False
        else:
            # bird
            writer.write(raw_record)
            can_write = True
