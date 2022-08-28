model_path="./models/test"
output_path="./outputs/test"
log_path="./logs/test"
model_file="./models/train/model_seed_123.pkl"

CUDA_VISIBLE_DEVICES=0 python -W ignore ./main_eval.py --model_path ${model_path} --output_path ${output_path} --log_path ${log_path} --model_file ${model_file}