cd ../../
pwd

for learning_rate in 0.001 0.0001 0.00001
do
venv/bin/python train.py \
  --model_id lstm \
  --model_type lstm \
  --model_arch_desc test \
  --data_path data/forex/t_note.csv \
  --accelerator cpu \
  --num_layers 5 \
  --hidden_size 128 \
  --epoch 10 \
  --seq_len 96 \
  --pred_len 31 \
  --label_len 10 \
  --learning_rate $learning_rate \
  --weight_decay 0.0001 \
  --scale False \
  --run_tensorboard True
done

