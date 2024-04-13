export WANDB_MODE=disabled

lr=$1
OUTPUT_DIR=output/alpaca-7b-noinst-$lr

torchrun --nproc_per_node=8 --master_port=2977 train.py \
    --model_name_or_path /root/paddlejob/workspace/env_run/tangtianyi/decapoda-research/llama-7b-hf \
    --data_path data/alpaca_data.json \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 8 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 1 \
    --save_total_limit 8 \
    --learning_rate $lr \
    --weight_decay 0. \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --instruction_type "no_inst" \
    --deepspeed configs/ds_z3_bf16.json \
    --tf32 True

bash utils/convert.sh $OUTPUT_DIR

    # --warmup_ratio 0.03 \
    # --lr_scheduler_type "cosine" \