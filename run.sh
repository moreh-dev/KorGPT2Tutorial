MOREH_NUM_DEVICES=8 MOREH_EXEC_INTERVAL=5000 python pretrain_gpt2.py --do_train --model_type=gpt2 --train_data_file=/home/share/dataset/GPT2_4G/data.txt --num_train_epochs=1 --output_dir=extend_kogpt2_test --overwrite_output_dir --per_gpu_train_batch_size=16 --logging_steps=10
