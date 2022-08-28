model_path="./models/train"
output_path="./outputs/train"
log_path="./logs/train"
seed=123
echo $model_path$num

CUDA_VISIBLE_DEVICES=0 python -W ignore ./main.py --model_path ${model_path} --output_path ${output_path} --log_path ${log_path} --seed ${seed}