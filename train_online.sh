export TOKENIZERS_PARALLELISM=false
torchrun --role $(hostname -s): --tee 3 --nnodes 1 --nproc-per-node=2 \
  --rdzv-backend=c10d --rdzv-endpoint=localhost:25500 \
	train_online.py \
  --eval_every_steps 500 \
  --save_every_steps 100 \
  --max_input_seq_len 768 \
  --max_tokens 1024 \
  --max_train_steps 2500 \
  --per_device_batch_size 48 \
  --eval_batch_size 48 \
  --learning_rate 3e-4 \
  --seed 333 \
  --weight_decay 0.01 \
  --num_epochs 100 \
  --task "online-distillation" \
  --warmup_steps 100 \
  --scheduler_type constant_with_warmup \
  --max_eval_batches 16 \
  --ddp \
  --gradient_checkpointing \
  --loss jsd \
  --skip_steps 0 \
  --model_name_or_path "checkpoints/philkrav/tinyllama-1.3b-draft-llama-13b-chat" \
  --use_flash_attn \
  --teacher_model "checkpoints/meta-llama/Llama-2-13b-chat-hf" \
#  --teacher_model "checkpoints/philkrav/tinyllama-1.3b-draft-llama-13b-chat" \
#  --profile \
