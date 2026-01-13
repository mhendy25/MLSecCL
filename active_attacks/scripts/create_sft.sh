
export ATTACKER_NAME=qwen2.5-1.5b
export SFT_SAVE_DIR=sft-$ATTACKER_NAME
python main.py \
    --mode sft \
    --model_name $ATTACKER_NAME \
    --lr 3e-5 \
    --train_steps 100 \
    --grad_acc_steps 32 \
    --batch_size 1024 \
    --few_shot_file ./prompts/sft_dataset.json \
    --exp_name attacker-$ATTACKER_NAME/sft \
    --save_dir $SFT_SAVE_DIR