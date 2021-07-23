MOREH_NUM_DEVICES=8 MOREH_EXEC_INTERVAL=5000 python pretrain_gpt2.py \
  --do_train \
  --model_type=gpt2 \
  --train_data_file=/home/share/dataset/GPT2_12G/tinyDedup.txt \
  --num_train_epochs=1 \
  --output_dir=extend_kogpt2_test \
  --overwrite_output_dir \
  --per_gpu_train_batch_size=32 \
  --logging_steps=500 \
  --do_eval \
  --per_gpu_eval_batch_size=32

