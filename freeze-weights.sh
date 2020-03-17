
#bazel build /code/tf/tensorflow/tensorflow/tensorflow/python/tools:freeze_graph && \
#	bazel-bin/tensorflow/python/tools/freeze_graph \
#	--input_graph=vww_96_grayscale_graph.pb \
#	--input_checkpoint=model.ckpt-8361242 \
#	--output_graph=frozen_graph.pb --output_node_names=MobilenetV1/Predictions/Reshape_1

#exit

! python /home/nemik/anaconda3/envs/tf1.15/lib/python3.7/site-packages/tensorflow_core/python/tools/freeze_graph.py \
	--input_graph=vww_96_grayscale_graph.pb \
	--input_checkpoint=vww_96_grayscale/model.ckpt-16563 \
	--input_binary=true --output_graph=vww_96_grayscale_frozen.pb \
	--output_node_names=MobilenetV1/Predictions/Reshape_1
