bash tools/dist_train.sh \
    configs/mrm/upernet_mrm-base_fp16_8x2_512x512_160k_siim.py 4 \
    --work-dir ./output/ --seed 0  --deterministic \
    --options model.backbone.pretrained=/path/to/pretrained/weights

