import tensorflow as tf
import io
import PIL
import numpy as np

"""
try:
  # %tensorflow_version only exists in Colab.
  import tensorflow.compat.v2 as tf
except Exception:
  pass

tf.enable_v2_behavior()
"""

def representative_dataset_gen():

  #record_iterator = tf.python_io.tf_record_iterator(path='dataset/val.record-00000-of-00010')
  record_iterator = tf.compat.v1.io.tf_record_iterator(path='dataset/val.record-00000-of-00010')
   

  #dataset = tf.data.Dataset.list_files("dataset/val.record-*")
  #record_iterator = tf.data.TFRecordDataset(dataset)
  
  count = 0
  for string_record in record_iterator:
    #print(f"STRING {string_record}")
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
    #array = ((array / 127.5) - 1.0).astype(np.int8)
    yield([array])
    count += 1
    if count > 300:
        break

#model = tf.keras.models.load_model('keras_birds-model-final.h5')
model = tf.keras.models.load_model('keras_birds-quantized_model-final.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
#converter = tf.lite.TFLiteConverter.from_frozen_graph('vww_96_grayscale_frozen.pb', ['input'], ['MobilenetV1/Predictions/Reshape_1'])
   
# tf 2.0 int8 - input is still float32 and has quantize layer that outputs int8
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
#converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.representative_dataset = representative_dataset_gen

output_graph_name = "vww_keras_96_tf2_int8_grayscale_quantized.tflite"
tflite_model = converter.convert()
open(output_graph_name, "wb").write(tflite_model)


# tf 2.0 uin8 - input is still float32 and has quantize layer that outputs int8
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.representative_dataset = representative_dataset_gen

output_graph_name = "vww_keras_96_tf2_uint8_grayscale_quantized.tflite"
tflite_model = converter.convert()
open(output_graph_name, "wb").write(tflite_model)


"""

converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file('keras_model/kmod-ckpt-001.h5')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.representative_dataset = representative_dataset_gen

output_graph_name = "vww_keras_96_tf1_int8_grayscale_quantized.tflite"
tflite_model = converter.convert()
open(output_graph_name, "wb").write(tflite_model)


converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file('keras_model/kmod-ckpt-001.h5')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.representative_dataset = representative_dataset_gen

output_graph_name = "vww_keras_96_tf1_uint8_grayscale_quantized.tflite"
tflite_model = converter.convert()
open(output_graph_name, "wb").write(tflite_model)

"""