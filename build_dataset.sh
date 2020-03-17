python /code/tf/models/research/slim/datasets/build_visualwakewords_data.py \
	--train_image_dir=dataset/coco_dataset/train2014 \
	--val_image_dir=dataset/coco_dataset/val2014 \
	--train_annotations_file=dataset/coco_dataset/annotations/instances_train2014.json \
	--val_annotations_file=dataset/coco_dataset/annotations/instances_val2014.json \
	--output_dir=dataset/birds \
	--small_object_area_threshold=0.05 \
	--foreground_class_of_interest='bird'
