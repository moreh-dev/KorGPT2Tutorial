DATESTAMP=${DATESTAMP:-`date +'%y%m%d%H%M%S%N'`}
BENCHMARK=${BENCHMARK:-"gpt2_o2"}
LOGDIR=${LOGDIR:-"./results/$BENCHMARK"}
LOGFILE_BASE="${LOGDIR}/${DATESTAMP}"
mkdir -p $(dirname "${LOGFILE_BASE}")

CUDA_VISIBLE_DEVICES=1 python pretrain_gpt2.py \
  --do_train \
  --model_type=gpt2 \
  --train_data_file=./tiny_dedup/tinyDedup.txt \
  --num_train_epochs=3 \
  --output_dir=extend_kogpt2_test_moreh_12g \
  --overwrite_output_dir \
  --per_gpu_train_batch_size=12 \
  --per_gpu_eval_batch_size=6 \
  --save_steps=0 \
  --eval_data_file=./tiny_dedup/sample_news_data.txt \
  --logging_steps=500 \
  --evaluate_during_training \
  --fp16 \
  --fp16_opt_level=O2 |& tee ${LOGFILE_BASE}_a100.log