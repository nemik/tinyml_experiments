
#bazel build /code/tf/tensorflow/tensorflow/tensorflow/python/tools:freeze_graph && \
#	bazel-bin/tensorflow/python/tools/freeze_graph \
#	--input_graph=vww_96_grayscale_graph.pb \
#	--input_checkpoint=model.ckpt-8361242 \
#	--output_graph=frozen_graph.pb --output_node_names=MobilenetV1/Predictions/Reshape_1

#exit

! python /code/tf/tensorflow/tensorflow/tensorflow/python/tools/freeze_graph.py \
	--input_graph=vww_96_grayscale_graph.pb \
	--input_checkpoint=train/model.ckpt-1000000 \
	--input_binary=true --output_graph=vww_96_grayscale_frozen.pb \
	--output_node_names=MobilenetV1/Predictions/Reshape_1
