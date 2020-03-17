python /code/tf/models/research/slim/train_image_classifier.py \
	--train_dir=vww_96_grayscale_weighted_0.05_0.95 \
	--dataset_name=visualwakewords \
	--dataset_split_name=train \
	--dataset_dir=dataset \
	--use_grayscale=True \
	--model_name=mobilenet_v1_025 \
	--preprocessing_name=mobilenet_v1 \
	--train_image_size=96 \
	--save_summaries_secs=120 \
	--learning_rate=0.045 \
	--label_smoothing=0.01 \
	--learning_rate_decay_factor=0.98 \
	--num_epochs_per_decay=2.5 \
	--moving_average_decay=0.9999 \
	--batch_size=96 \
	--max_number_of_steps=100000

#--max_number_of_steps=1000000
