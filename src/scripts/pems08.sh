
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi


if [ ! -d "./logs/PEMS08" ]; then
    mkdir ./logs/PEMS08
fi
model_name=TimeMixer



for seq_len in 96
do
for lr in 0.005 0.001 0.0025 
do 
for human_learning_rate in 0.001 0.01 0.005 
do 
for pred_len in 12 24 48 96 
do 
CUDA_VISIBLE_DEVICES=1 python3 -u run.py \
  --task_name 'HumanFactor' \
  --is_training 1 \
  --root_path ../dataset/\
  --data_path PEMS08.npz \
  --model_id PEMS08_$seq_len'_'$pred_len'_'$lr'_'$human_learning_rate \
  --model $model_name \
  --data human_pems \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 170 \
  --dec_in 170 \
  --c_out 170 \
  --des 'Exp' \
  --itr 1 \
  --d_model 64 \
  --d_ff 32 \
  --batch_size 32 \
  --learning_rate $lr \
  --down_sampling_layers 1 \
  --down_sampling_method avg \
  --down_sampling_window 2 \
  --human_learning_rate $human_learning_rate --human_train_epochs 20 >logs/PEMS08/$model_name'_'PEMS08_$seq_len'_'$pred_len'_'$lr'_'$human_learning_rate'_'20.log
done 
done 
done 
done 

