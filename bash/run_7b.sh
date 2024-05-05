export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0,1

OUTPUT_DIR=/data/model/FIGA

torchrun --nproc_per_node=2 --master_port=1234 train.py \
    --model_name_or_path /path/to/your_model \
    --data_path RUCAIBox/SPA \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 1 \
    --save_total_limit 5 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --instruction_type "inst" \
    --deepspeed configs/ds_z3_bf16.json \
    --tf32 True \
    --gradient_checkpointing \

bash utils/convert.sh $OUTPUT_DIR
