export TOKENIZERS_PARALLELISM=false
torchrun --role $(hostname -s): --tee 3 --nnodes 1 --nproc-per-node=1 \
  --rdzv-backend=c10d --rdzv-endpoint=localhost:25500 \
	train_online.py \
  --model_name_or_path "checkpoints/philkrav/tinyllama-1.3b-draft-llama-13b-chat" \
  --eval_every_steps 500 \
  --save_every_steps 100 \
  --max_input_seq_len 512 \
  --max_tokens 1024 \
  --max_train_steps 12000 \
  --per_device_batch_size 32 \
  --eval_batch_size 32 \
  --learning_rate 3e-5 \
  --seed 333 \
  --weight_decay 0.01 \
  --num_epochs 100 \
  --task "online-distillation" \
  --warmup_steps 50 \
  --scheduler_type constant_with_warmup \
  --max_eval_batches 16 \
  --ddp \
  --gradient_checkpointing \
  --loss rkl \
  --teacher_model "checkpoints/meta-llama/Llama-2-13b-chat-hf" \
  --skip_steps 0 \
  --use_flash_attn \
#  --profile \
