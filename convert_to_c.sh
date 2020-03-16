cat c_head.txt > person_detect_model_data.cc
xxd -i vww_96_grayscale_quantized.tflite >> person_detect_model_data.cc
sed -i '34d' person_detect_model_data.cc
sed -i 's/unsigned char vww_96_grayscale_quantized_tflite[\\] = {/a/g' person_detect_model_data.cc
sed -i 's/unsigned int vww_96_grayscale_quantized_tflite_len/const int g_person_detect_model_data_len/g' person_detect_model_data.cc