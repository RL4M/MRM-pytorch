_base_ = [
    '../_base_/models/upernet_mae.py', '../_base_/datasets/siim.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_5k.py'
]

model = dict(
    backbone=dict(
        type='MRM',
        pretrained=None,
        img_size=(512, 512),
        patch_size=16,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        init_values=1.0,
        drop_path_rate=0.1,
        out_indices=[3, 5, 7, 11]),
    neck=dict(embed_dim=768, rescales=[4, 2, 1, 0.5]),
    decode_head=dict(
        in_channels=[768, 768, 768, 768], num_classes=2, channels=768),
    auxiliary_head=dict(in_channels=768, num_classes=2),
    # test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)))
    test_cfg=dict(mode='whole'))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=2e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.65))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# mixed precision
fp16 = dict(loss_scale='dynamic')

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)
checkpoint_config = dict(by_epoch=False, interval=10000)
evaluation = dict(by_epoch=False, interval=50, metric='medDice',save_best='Dice.front',rule = 'greater')