import tensorflow as tf
import io
import PIL
import numpy as np
import os
import glob

l_counts = {}
l_counts["background"] = 0
l_counts["bird"] = 0

def r():

  print("doing")

  files = glob.glob("dataset/train.record-*")
  print(files)
  for fr in files:

    print(f"\n\tdoing {fr}")

    record_iterator = tf.python_io.tf_record_iterator(path=fr)

    count = 0

    for string_record in record_iterator:
      example = tf.train.Example()
      example.ParseFromString(string_record)
      #print(dir(example.features.feature))#.feature['label'])
      #for k,f in example.features.feature.items():
      #  print(f['image/class/label'])
      label = example.features.feature['image/class/label'].int64_list.value[0]
      if label == 0:
        l_counts["background"] += 1
      else:
        l_counts["bird"] += 1

      image_stream = io.BytesIO(example.features.feature['image/encoded'].bytes_list.value[0])
      image = PIL.Image.open(image_stream)
      image = image.resize((96, 96))
      image = image.convert('L')
      array = np.array(image)
      array = np.expand_dims(array, axis=2)
      array = np.expand_dims(array, axis=0)
      array = ((array / 127.5) - 1.0).astype(np.float32)
      yield([array])
      count += 1
      if count > 300:
          break


    print(fr)
    print(l_counts)

for v in r():
  pass
  #print(l_counts)
  
print("done")
print(l_counts)