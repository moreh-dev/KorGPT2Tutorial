CUDA_VISIBLE_DEVICES=0 python pretrain_gpt2.py \
  --do_train \
  --model_type=gpt2 \
  --train_data_file=/nas/share/dataset/GPT2_4G/data.txt \
  --num_train_epochs=1 \
  --output_dir=extend_kogpt2_test \
  --overwrite_output_dir \
  --per_gpu_train_batch_size=2 \
  --logging_steps=1