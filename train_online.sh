export TOKENIZERS_PARALLELISM=false
torchrun --role $(hostname -s): --tee 3 --nnodes 1 --nproc-per-node=1 \
  --rdzv-backend=c10d --rdzv-endpoint=localhost:25500 \
	train_online.py \
  --eval_every_steps 500 \
  --save_every_steps 50 \
  --max_train_steps 12000 \
  --per_device_batch_size 1 \
  --eval_batch_size 1 \
  --learning_rate 1e-5 \
  --seed 333 \
  --weight_decay 0.01 \
  --num_epochs 100 \
  --dataset "openhermes" \
  --task "online-distillation" \
  --warmup_steps 50 \
  --scheduler_type constant_with_warmup \
  --max_eval_batches 8 \
  --ddp \
  --model_name_or_path "checkpoints/philkrav/tinyllama-1.3b-draft-llama-13b-chat" \
  --gradient_checkpointing \
  --loss rkl \
  --teacher_model "checkpoints/meta-llama/Llama-2-13b-chat-hf" \
  --max_input_seq_len 1024 \
  --skip_steps 0 \
#  --use_flash_attn \
#  --compile
#  --teacher_model "checkpoints/philkrav/TinyLlama-Chat-Mistral-Tok" \
#  --compile
#  --model_name_or_path "checkpoints/mixtral-tiny" \
#  --profile \
