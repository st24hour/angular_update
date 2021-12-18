#!/bin/bash

############################## 200527 ###########################
# 8gpu
for i in 0.125
do
	CUDA_VISIBLE_DEVICES=5 python invariant.1.0.2.3.py \
					--dataset cifar10 \
					--epochs 300 \
					--batch_size 512 \
					--weight_decay 0.0001 \
					--lr 0.8 \
					--momentum 0.9 \
					--net_type densenetBC100_GBN_invariant \
					--save_model \
					--seed 4 \
					--zero_init_residual \
					--warm_up_epoch 0 \
					--alpha_sqaure $i
done

# DGX
# for i in 0.5
# do
# 	CUDA_VISIBLE_DEVICES=5 python invariant.1.0.2.3.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.0001 \
# 					--lr 0.2 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN_invariant \
# 					--save_model \
# 					--seed 3 \
# 					--zero_init_residual \
# 					--warm_up_epoch 0 \
# 					--alpha_sqaure $i
# done

############################## 200522 ###########################
# 8gpu
# for i in 0
# do
# 	CUDA_VISIBLE_DEVICES=5 python invariant.1.0.2.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--lr 0.001 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--eps 1e-05
# done

# 8gpu
# for i in 0.001
# do
# 	CUDA_VISIBLE_DEVICES=5 python invariant.1.0.2.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--lr 0.001 \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--eps 1e-05
# done

############################## 200521 ###########################
# 8gpu
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=5 python invariant.1.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 75 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e+02 \
#  					--save_dir './logs_invariant_epoch/' \
#  					--eps 1e-05 \
#  					--epoch_step 38 56
# done

############################## 200520 ###########################
# DGX
# for i in 1e-01
# do
# 	CUDA_VISIBLE_DEVICES=5 python invariant.1.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-04  \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e+02 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e-05
# done

############################## 200515 ###########################
# # DGX
# for i in 5e-01
# do
# 	CUDA_VISIBLE_DEVICES=4 python invariant.1.0.3.py \
# 					--dataset cifar10 \
# 					--epochs 35 \
# 					--batch_size 128 \
# 					--weight_decay 2e-04 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e+02 \
#  					--save_dir './logs_invariant_faster/' \
#  					--eps 1e-05 \
#  					--epoch_step 5 10 15 20 25 30 \
#  					--lr_decay 0.2
# done

# DGX
# for i in 1e+01
# do
# 	CUDA_VISIBLE_DEVICES=5 python invariant.1.0.2.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-06 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN_invariant \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e+04 \
#  					--save_dir './logs_invariant_scaler/' \
#  					--eps 1e-03
# done

# 8gpu-h
# for i in 2e-01
# do
# 	CUDA_VISIBLE_DEVICES=5 python invariant.1.0.3.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-04 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e+02 \
#  					--save_dir './logs_invariant_faster/' \
#  					--eps 1e-05
# done

# 8gpu-h
# for i in 1e+01
# do
# 	CUDA_VISIBLE_DEVICES=5 python invariant.1.0.3.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-06 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e+04 \
#  					--save_dir './logs_invariant_scaler/' \
#  					--eps 1e-03
# done

# DGX
# for i in 1e+01
# do
# 	CUDA_VISIBLE_DEVICES=5 python variant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-06 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e+04 \
#  					--save_dir './logs_variant_same/' \
#  					--amp \
#  					--eps 1e-03
# done

# 8gpu-h
# for i in 1e+01
# do
# 	CUDA_VISIBLE_DEVICES=5 python invariant.1.0.3.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-06 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e+04 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e-03
# done

############################## 200513 ###########################
# 8gpu-h
# for i in 10
# do
# 	CUDA_VISIBLE_DEVICES=5 python variant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.000002 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 12800 \
#  					--save_dir './logs_variant_same/' \
#  					--amp \
#  					--eps 1.00E-03 \
#  					--filter_bn_bias
# done

############################## 200512 ###########################
# 8gpu-hyejin
# for i in 0.0004
# do
# 	CUDA_VISIBLE_DEVICES=5 python invariant_num.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--num_sample 25000
# done

############################## 200511 ###########################
# 8gpu-h
# for i in 1
# do
# 	CUDA_VISIBLE_DEVICES=5 python invariant.1.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-05  \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1280 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 0.0001
# done

# 8GPU-hyejin
# for i in 6250
# do
# 	CUDA_VISIBLE_DEVICES=5 python invariant_num.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.0032 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--num_sample $i
# done

############################## 200509 ###########################
# 8gpu-lee
# for i in 6250
# do
# 	CUDA_VISIBLE_DEVICES=5 python invariant_num.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.0032 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--num_sample $i
# done

############################## 200508 ###########################
# DGX
# for i in 1563 3125 6250 12500 25000 50000
# do
# 	CUDA_VISIBLE_DEVICES=5 python invariant_num.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 64 \
# 					--weight_decay 0.0002 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--num_sample $i
# done

# # DGX
# for i in 1e-06
# do
# 	CUDA_VISIBLE_DEVICES=5 python invariant.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 64 \
# 					--weight_decay $i \
# 					--lr 10 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 6400 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e-03
# done

# # 8gpu-lee
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=5 python invariant_num.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--num_sample 50000
# done

# # 8GPU-p
# for i in 0.002
# do
# 	CUDA_VISIBLE_DEVICES=5 python invariant.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--lr 0.01 \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 12.8 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e-6
# done

# # 8GPU-p
# for i in 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=5 python invariant.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 128 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e-5
# done

# # 8GPU-p
# for i in 2e-05
# do
# 	CUDA_VISIBLE_DEVICES=5 python invariant.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--lr 1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1280 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e-4
# done

############################## 200505 ###########################
# 8gpu-p
# for i in 0.0004 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=5 python invariant_num.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--num_sample 50000
# done

# # # 8gpu-lee
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=5 python invariant_num.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--num_sample 25000
# done

############################## 200504 ###########################
# 8gpu-p
# for i in 2e-09
# do
# 	CUDA_VISIBLE_DEVICES=5 python invariant.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--lr 10000 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 12800000 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1
# done

############################## 200503 ###########################
# 8gpu-p
# for i in 0.02
# do
# 	CUDA_VISIBLE_DEVICES=5 python invariant.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--lr 0.001 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e-8
# done

# 8gpu-b
# for i in 0.02
# do
# 	CUDA_VISIBLE_DEVICES=5 python invariant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--lr 0.001 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28 \
#  					--save_dir './logs_invariant_same/'
# done


############################## 200430 ###########################
# 8gpu
# for i in 3.2
# do
# 	CUDA_VISIBLE_DEVICES=5,6 python invariant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay 0.0002 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0
# done

# # 8gpu
# for i in 1.6
# do
# 	CUDA_VISIBLE_DEVICES=5,6 python invariant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 1024 \
# 					--weight_decay 0.0001 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0
# done

############################## 200429 ###########################
# # 8gpu
# for i in 0.2
# do
# 	CUDA_VISIBLE_DEVICES=5 python invariant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 256 \
# 					--weight_decay 0.0002 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0
# done

# # 8gpu
# for i in 1.6
# do
# 	CUDA_VISIBLE_DEVICES=5 python invariant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 2048 \
# 					--weight_decay 0.0002 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0
# done

############################# 200410 ###########################
# # 8gpu-p
# # num_data densenet
# for i in 1.6 3.2 6.4
# do
# 	CUDA_VISIBLE_DEVICES=5 python num_data.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 64 \
# 					--weight_decay 0.0001 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 15 \
#  					--num_sample 50000
# done

############################## 200331 ###########################
# 8gpu
# for i in 0.1 0.2 0.4
# do
# 	CUDA_VISIBLE_DEVICES=5 python num_data.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.0002 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 15 \
#  					--num_sample 50000
# done

############################## 200306 ###########################
# 8gpu-b
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=5 python wd_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 2048 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual
# done

# 8gpu-h
# for i in 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=5 python wd_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 2048 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual
# done

############################## 200301 ###########################
# 8gpu-b
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=5 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--dataroot /home/user/ssd1/dataset/ \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--save_dir ./logs/ \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual
# done

############################## 200228 ###########################
# 8gpu-b
# for i in 0.0128
# do
# 	CUDA_VISIBLE_DEVICES=5 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--dataroot /home/user/ssd1/dataset/ \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--save_dir ./logs/ \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual
# done

# 8gpu-b
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=5 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--dataroot /home/user/ssd1/dataset/ \
# 					--epochs 300 \
# 					--batch_size 512 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--save_dir ./logs/ \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual
# done

############################## 200227 ###########################
# 8gpu-b
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=5 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--dataroot /home/user/ssd1/dataset/ \
# 					--epochs 300 \
# 					--batch_size 1024 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--save_dir ./logs/ \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual
# done

############################## 200114 ###########################
# 8GPU
# for i in 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=5 python wd_lr_test.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs/ \
# 					--seed 0 1 2
# done

####################.########## 200111 ###########################
# 8GPU
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=5 python critical.0.0.1.py \
# 					--dataset cifar100 \
# 					--epochs 300 \
# 					--batch_size 512 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs/
# done

# 8GPU
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=5 python critical.0.0.1.py \
# 					--dataset cifar100 \
# 					--epochs 300 \
# 					--batch_size 256 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs/
# done

# # 8GPU
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=5 python critical.0.0.1.py \
# 					--dataset cifar100 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs/
# done

####################.########## 200108 ###########################
# # 8GPU
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=5 python critical.0.0.1.py \
# 					--dataset cifar100 \
# 					--epochs 300 \
# 					--batch_size 64 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs/
# done

############################## 200103 ###########################
# 8GPU
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=5 python critical.0.0.1.py \
# 					--epochs 300 \
# 					--batch_size 256 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs/
# done

############################## 201225 ###########################
# # 8GPU
# # for i in 0.0001
# for i in 0.0006 0.0007 0.0008 0.0009 0.001
# do
# 	CUDA_VISIBLE_DEVICES=5 python critical.0.0.1.py \
# 					--epochs 300 \
# 					--batch_size 256 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs/
# done

############################## 201125 ###########################
# 8GPU
# for i in 125 
# do
# 	CUDA_VISIBLE_DEVICES=5 python critical.0.0.1.py \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.0001 \
# 					--wd_off_epoch $i
# done


############################## 201124 ###########################
# 8GPU
# for i in $(seq 225 25 300)
# for i in 125
# do
# 	CUDA_VISIBLE_DEVICES=5 python critical.0.0.1.py \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.0005 \
# 					--wd_off_epoch $i
# done
