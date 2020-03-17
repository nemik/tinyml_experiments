toco \
	--graph_def_file=vww_96_grayscale_frozen.pb --output_file=t.tflite \
	--input_shapes=96,96,1 --input_arrays='input' --output_arrays='MobilenetV1/Predictions/Reshape_1' \
	--inference_type=QUANTIZED_UINT8 --mean_values=0 --std_dev_values=9.8077
