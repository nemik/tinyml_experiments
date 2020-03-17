cat c_head.txt > person_detect_model_data.cc
xxd -i "$1.tflite" >> person_detect_model_data.cc
sed -i '34d' person_detect_model_data.cc
sed -i 's/unsigned char $1_tflite[\\] = {/a/g' person_detect_model_data.cc
sed -i "s/unsigned int $1_tflite_len/const int g_person_detect_model_data_len/g" person_detect_model_data.cc