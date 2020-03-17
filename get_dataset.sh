mkdir dataset
python /code/tf/models/research/slim/download_and_convert_data.py \
	--dataset_name=visualwakewords \
	--dataset_dir="dataset" \
	--small_object_area_threshold=0.05 \
	--foreground_class_of_interest='bird'
