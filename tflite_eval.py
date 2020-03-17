import tflite_runtime.interpreter as tflite
m = "keras_birds/keras_birds_mobilenet_v2_model-final.h5"

interpreter = tf.lite.Interpreter(model_path=args.model_file)

