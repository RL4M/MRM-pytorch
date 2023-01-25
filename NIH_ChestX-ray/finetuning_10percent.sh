CUDA_VISIBLE_DEVICES=0 python3 train.py --name mrm --stage train --model vit_base_patch16 --model_type ViT-B_16 --num_classes 14 \
    --pretrained_path "../MRM.pth" --dataset_path '/path/to/dataset/NIH_ChestX-ray/' \
    --output_dir "finetuning_outputs/" --data_volume '10' --num_steps 30000  --eval_batch_size 512 --img_size 224 \
    --learning_rate 3e-3 --warmup_steps 500 --fp16 --fp16_opt_level O2 --train_batch_size 96
