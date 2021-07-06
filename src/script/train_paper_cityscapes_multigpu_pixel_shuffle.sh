#! /bin/bash
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 main.py --batchSize 4 --checkpoints_dir ../checkpoints --dataroot ../../datasets --gpu_ids 0,1 --n_layers_D 3 --num_D_local 1 --num_D_global 2 --name SGI_NET_cityscapes_fp16_multigpu  --nThreads 10 --niter 600 --niter_decay 0 --image_height 256 --image_width 512 --size_crop_width 256 --size_crop_height 256 --which_perceptual_loss vgg --gan_mode lsgan --lambda_rec 10 --lambda_style 500 --lambda_perceptual 10 --min_hole_size 32 --max_hole_size 128 --use_sn_discriminator --classes_of_interest 8,10 --use_bbox --label_nc 17 --n_downsample_global 4 --which_encoder ctx_label --lambda_seg_map 10 --use_spade --use_multi_scale_loss --tf_log --local_world_size=2 --use_pixel_shuffle
