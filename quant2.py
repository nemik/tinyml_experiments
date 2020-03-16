import tensorflow as tf
import io
import PIL
import numpy as np


def representative_dataset_gen():

  record_iterator = tf.python_io.tf_record_iterator(path='dataset/val.record-00000-of-00010')

  count = 0
  for string_record in record_iterator:
    example = tf.train.Example()
    example.ParseFromString(string_record)
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


#model = tf.keras.models.load_model(input_graph_name)
#converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter = tf.lite.TFLiteConverter.from_frozen_graph('vww_96_grayscale_frozen.pb', ['input'], ['MobilenetV1/Predictions/Reshape_1'])

quantization = 1

output_graph_name = "vww_96_grayscale_quantized.tflite"
converter.representative_dataset = representative_dataset_gen
converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
if 1== quantization:
    print("do quantization!")
    #converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    #converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
else:
    converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
    print("no quantization!")

tflite_model = converter.convert()
open(output_graph_name, "wb").write(tflite_model)