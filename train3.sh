#!/bin/bash

############################## 200527 ###########################
# DGX
for i in 0.25
do
	CUDA_VISIBLE_DEVICES=3 python invariant.1.0.2.3.py \
					--dataset cifar10 \
					--epochs 300 \
					--batch_size 256 \
					--weight_decay 0.0001 \
					--lr 0.4 \
					--momentum 0.9 \
					--net_type densenetBC100_GBN_invariant \
					--save_model \
					--seed 2 3 \
					--zero_init_residual \
					--warm_up_epoch 0 \
					--alpha_sqaure $i
done

############################## 200525 ###########################
# # DGX
	# CUDA_VISIBLE_DEVICES=3 python invariant.1.0.2.py \
	# 				--dataset cifar10 \
	# 				--epochs 300 \
	# 				--batch_size 128 \
	# 				--weight_decay 0.0001 \
	# 				--lr 0.2 \
	# 				--momentum 0.9 \
	# 				--net_type densenetBC100_GBN_invariant \
	# 				--save_model \
	# 				--seed 0 1 2 3 4 \
	# 				--zero_init_residual \
	# 				--warm_up_epoch 15 

############################## 200523 ###########################
# # DGX
# for i in 0.125
# do
# 	CUDA_VISIBLE_DEVICES=3 python invariant.1.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 512 \
# 					--weight_decay 0.0001 \
# 					--lr 0.8 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
# 					--warm_up_epoch 15 \
# 					--alpha_sqaure $i
# done

# DGX
	# CUDA_VISIBLE_DEVICES=3 python invariant.1.0.2.py \
	# 				--dataset cifar10 \
	# 				--epochs 300 \
	# 				--batch_size 2048 \
	# 				--weight_decay 0.0002 \
	# 				--lr 1.6 \
	# 				--momentum 0.9 \
	# 				--net_type resnet18_GBN_invariant2 \
	# 				--save_model \
	# 				--seed 0 1 2 3 4 \
	# 				--zero_init_residual \
 # 					--warm_up_epoch 15
 					
# 8gpu
# for i in 0.0625
# do
# 	CUDA_VISIBLE_DEVICES=3 python invariant.1.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 2048 \
# 					--weight_decay 0.0002 \
# 					--lr 1.6 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 15 \
#  					--alpha_sqaure $i
# done

############################## 200522 ###########################
# 8gpu
# for i in 0.00008
# do
# 	CUDA_VISIBLE_DEVICES=3 python invariant.1.0.2.2.py \
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

############################## 200521 ###########################
# 8gpu
# for i in 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=3 python invariant.1.0.2.py \
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

# 8gpu
# for i in 12500
# do
# 	CUDA_VISIBLE_DEVICES=3 python invariant_num.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.0002 \
# 					--lr 0.4 \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--num_sample $i \
#  					--base_norm 512 \
#  					--eps 0.00004
# done

############################## 200520 ###########################
# DGX
# for i in 1e+16
# do
# 	CUDA_VISIBLE_DEVICES=3 python invariant.1.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-21  \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e+19 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e+12
# done

############################## 200515 ###########################
# # DGX
# for i in 1e+00
# do
# 	CUDA_VISIBLE_DEVICES=3 python invariant.1.0.3.1.py \
# 					--dataset cifar10 \
# 					--epochs 100 \
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
#  					--epoch_step 10 30 50 70 \
#  					--lr_decay 0.1 \
#  					--wd_linear 0.0005
# done

# # DGX
# for i in 5e-01
# do
# 	CUDA_VISIBLE_DEVICES=2 python invariant.1.0.3.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 1e-03 \
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
#  					--epoch_step 3 6 9 12 \
#  					--lr_decay 0.2
# done

# DGX
# for i in 1e-01
# do
# 	CUDA_VISIBLE_DEVICES=3 python invariant.1.0.2.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-04 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN_invariant \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e+02 \
#  					--save_dir './logs_invariant_scaler/' \
#  					--eps 1e-05
# done

# 8gpu-h
# for i in 2e-01
# do
# 	CUDA_VISIBLE_DEVICES=3 python invariant.1.0.3.py \
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

# 8gpu-chan
# for i in 1e-01
# do
# 	CUDA_VISIBLE_DEVICES=3 python invariant.1.0.3.py \
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
#  					--save_dir './logs_invariant_scaler/' \
#  					--eps 1e-05
# done

# DGX
# for i in 1e-01
# do
# 	CUDA_VISIBLE_DEVICES=3 python variant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-04 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e+02 \
#  					--save_dir './logs_variant_same/' \
#  					--amp \
#  					--eps 1e-05
# done

# samsung
# for i in 1e+12
# do
# 	CUDA_VISIBLE_DEVICES=3 python invariant.1.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-17  \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e+15 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e+08
# done

# 8gpu-h
# for i in 1e-01
# do
# 	CUDA_VISIBLE_DEVICES=3 python invariant.1.0.3.py \
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
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e-05
# done

############################## 200513 ###########################
# samsung2
# for i in 1e+07
# do
# 	CUDA_VISIBLE_DEVICES=3 python invariant.1.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-12  \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e+10 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1000
# done

# 8gpu-h
# for i in 0.1
# do
# 	CUDA_VISIBLE_DEVICES=3 python variant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.0002 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 128 \
#  					--save_dir './logs_variant_same/' \
#  					--amp \
#  					--eps 1.00E-05 \
#  					--filter_bn_bias
# done

############################## 200512 ###########################
# samsung
# for i in 0.4
# do
# 	CUDA_VISIBLE_DEVICES=3 python invariant_num.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.0002 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--num_sample 12500 \
#  					--base_norm 512 \
#  					--eps 0.00004
# done

# # samsung
# for i in 0.2
# do
# 	CUDA_VISIBLE_DEVICES=3 python invariant_num.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.0002 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--num_sample 25000 \
#  					--base_norm 256 \
#  					--eps 0.00002
# done

# # samsung
# for i in 0.1
# do
# 	CUDA_VISIBLE_DEVICES=3 python invariant_num.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.0002 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--num_sample 50000 \
#  					--base_norm 128 \
#  					--eps 0.00001
# done

# samsung
# for i in 0.8
# do
# 	CUDA_VISIBLE_DEVICES=3 python invariant_num.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.0002 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--num_sample 6250 \
#  					--base_norm 1024 \
#  					--eps 0.00008
# done

# 8gpu-hyejin
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=3 python invariant_num.1.0.1.py \
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
# for i in 100
# do
# 	CUDA_VISIBLE_DEVICES=3 python invariant.1.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-07  \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 128000 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 0.01
# done

# samsung
# for i in 0.0002 0.0004 0.0016
# do
# 	CUDA_VISIBLE_DEVICES=3 python invariant_num.1.0.1.py \
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
#  					--num_sample 12500
# done

# samsung
# for i in 3125
# do
# 	CUDA_VISIBLE_DEVICES=3 python invariant_num.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.0001 \
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
# 	CUDA_VISIBLE_DEVICES=3 python invariant_num.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.0008 \
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
# 	CUDA_VISIBLE_DEVICES=3 python invariant_num.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 64 \
# 					--weight_decay 0.0008 \
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
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=3 python invariant.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 64 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 64 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e-05
# done

# # 8gpu-lee
# for i in 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=3 python invariant_num.1.0.0.py \
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

# # # group4
# for i in 0.0008 0.0004 0.0002 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=3 python invariant_num.1.0.0.py \
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
#  					--num_sample 1563
# done

############################## 200507 ###########################
# # group4
# for i in 0.0032 0.0002 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=3 python invariant_num.1.0.0.py \
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
#  					--num_sample 3125
# done

############################## 200505 ###########################
# # # 8gpu-lee
# for i in 0.0008 0.0004
# do
# 	CUDA_VISIBLE_DEVICES=3 python invariant_num.1.0.0.py \
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
#  					--num_sample 12500
# done

# # group4
# for i in 0.0128 0.0064 0.0032 0.0016 0.0008 0.0004 0.0002 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=3 python invariant_num.1.0.0.py \
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
#  					--num_sample 3120
# done

############################## 200504 ###########################
# 8gpu-p
# for i in 0.000002
# do
# 	CUDA_VISIBLE_DEVICES=3 python invariant.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--lr 10 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 12800 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e-3
# done

# # 8gpu-p
# for i in 2e-07
# do
# 	CUDA_VISIBLE_DEVICES=3 python invariant.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--lr 100 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 128000 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e-2
# done

############################## 200503 ###########################
# 8gpu-p
# for i in 0.00000002
# do
# 	CUDA_VISIBLE_DEVICES=3 python invariant.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--lr 1000 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1280000 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp
# done

# 8gpu-b
# for i in 0.00000002
# do
# 	CUDA_VISIBLE_DEVICES=3 python invariant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--lr 1000 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1280000 \
#  					--save_dir './logs_invariant_same/'
# done

############################## 200501 ###########################
# # 8gpu-b
# for i in 0.02
# do
# 	CUDA_VISIBLE_DEVICES=3 python invariant.1.0.0.py \
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

############################## 200429 ###########################
# # 8gpu
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=3,4 python invariant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 128
# done

# # 8gpu
# for i in 0.0016
# do
# 	CUDA_VISIBLE_DEVICES=3,4 python invariant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 1024 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 64
# done

############################## 200410 ###########################
# # 8gpu-p
# # num_data densenet
# for i in 0.05 0.1 0.2 0.4 0.8 1.6 3.2 6.4
# do
# 	CUDA_VISIBLE_DEVICES=3 python num_data.1.0.0.py \
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
#  					--num_sample 25000
# done

# # # group4
# for i in 12.8
# do
# 	CUDA_VISIBLE_DEVICES=3 python num_data.1.0.0.py \
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
#  					--warm_up_epoch 0 \
#  					--num_sample 50000
# done

# # group4
# for i in 0.2
# do
# 	CUDA_VISIBLE_DEVICES=3 python num_data.1.1.0.py \
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
#  					--warm_up_epoch 0 \
#  					--num_sample 25000 \
#  					--base_lr 0.1
# done

# # group4
# for i in 0.4
# do
# 	CUDA_VISIBLE_DEVICES=3 python num_data.1.1.0.py \
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
#  					--warm_up_epoch 0 \
#  					--num_sample 12500 \
#  					--base_lr 0.1
# done

# # group4
# for i in 0.8
# do
# 	CUDA_VISIBLE_DEVICES=3 python num_data.1.1.0.py \
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
#  					--warm_up_epoch 0 \
#  					--num_sample 6250 \
#  					--base_lr 0.1
# done

# # group4
# for i in 1.6
# do
# 	CUDA_VISIBLE_DEVICES=3 python num_data.1.1.0.py \
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
#  					--warm_up_epoch 0 \
#  					--num_sample 3125 \
#  					--base_lr 0.1
# done

# # group4
# for i in 3.2
# do
# 	CUDA_VISIBLE_DEVICES=3 python num_data.1.1.0.py \
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
#  					--warm_up_epoch 0 \
#  					--num_sample 1563 \
#  					--base_lr 0.1
# done

############################## 200403 ###########################
# group4
# for i in 0.05 0.1 0.2 0.4 0.8 1.6 3.2 6.4 12.8
# do
# 	CUDA_VISIBLE_DEVICES=3 python num_data.1.0.0.py \
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
#  					--warm_up_epoch 0 \
#  					--num_sample 50000
# done

############################## 200401 ###########################
# group4
# for i in 1.6
# do
# 	CUDA_VISIBLE_DEVICES=3 python lr_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 2048 \
# 					--weight_decay 0.0002 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0
# done

# # group4
# for i in 0.8
# do
# 	CUDA_VISIBLE_DEVICES=3 python lr_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 1024 \
# 					--weight_decay 0.0002 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0
# done

# # group4
# for i in 0.4
# do
# 	CUDA_VISIBLE_DEVICES=3 python lr_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 512 \
# 					--weight_decay 0.0002 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0
# done

# group4
# for i in 0.0128
# do
# 	CUDA_VISIBLE_DEVICES=3 python num_data.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--num_sample 25000
# done

############################## 200331 ###########################
# 8gpu
# for i in 0.1 0.2 0.4 0.8
# do
# 	CUDA_VISIBLE_DEVICES=3 python num_data.1.0.0.py \
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
#  					--num_sample 25000
# done

# IITP
# for i in 0.0064 0.0128
# do
# 	CUDA_VISIBLE_DEVICES=3 python num_data.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--num_sample 12500
# done

############################## 200329 ###########################
# 8gpu-o
# for i in 0.0001 0.0002 0.0004 0.0008 0.0016 0.0032 0.0064 0.0128
# do
# 	CUDA_VISIBLE_DEVICES=3 python num_data.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--num_sample 1563
# done

# IITP
# for i in 0.0001 0.0002 0.0004 0.0008 0.0016 0.0032 0.0064 0.0128
# do
# 	CUDA_VISIBLE_DEVICES=3 python num_data.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--num_sample 12500
# done

# group4
# for i in 0.0001 0.0002 0.0004 0.0008 0.0016 0.0032 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=3 python num_data.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--num_sample 25000
# done

############################## 200321 ###########################
# IITP
# for i in 0.0064 0.0128
# do
# 	CUDA_VISIBLE_DEVICES=3 python wd_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 64 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0
# done

# group4
# for i in 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=3 python scaler.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 15 \
#  					--base_norm 256
# done

# group4
# for i in 0.0004
# do
# 	CUDA_VISIBLE_DEVICES=3 python scaler.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 256 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 15 \
#  					--base_norm 256
# done

# # group4
# for i in 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=3 python scaler.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 512 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 15 \
#  					--base_norm 256
# done

############################## 200321 ###########################
# IITP
# for i in 0.8
# do
# 	CUDA_VISIBLE_DEVICES=3 python lr_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 512 \
# 					--weight_decay 0.0001 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 15
# done

############################## 200311 ###########################
# group4
# for i in 0.0002 0.0004 0.0008 0.0016 0.0032 0.0064 0.0128 0.0256
# do
# 	CUDA_VISIBLE_DEVICES=3 python wd_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 256 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual
# done

############################## 200312 ###########################
# IITP
# for i in 0.0001 0.0002 0.0004 0.0008 0.0016 0.0032 0.0064 0.0128 0.0256
# do
# 	CUDA_VISIBLE_DEVICES=3 python wd_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 512 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual
# done

############################## 200311 ###########################
# group4
# for i in 0.0001 0.0002 0.0004 0.0008 0.0016 0.0032 0.0064 0.0128 0.0256
# do
# 	CUDA_VISIBLE_DEVICES=3 python wd_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 256 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual
# done

############################## 200307 ###########################
# samsung
# for i in 0.4
# do
# 	CUDA_VISIBLE_DEVICES=3 python lr_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 512 \
# 					--weight_decay 0.0002 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 15 \
# 					--filter_bn \
# 					--filter_bias	
# done

############################## 200305 ###########################
# group4
# for i in 0.0001 0.0002 0.0004 0.0008 0.0016 0.0032 0.0064 0.0128 0.0256
# do
# 	CUDA_VISIBLE_DEVICES=3 python wd_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 1024 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--save_dir ./logs/ \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual
# done

# IITP
# for i in 0.0001 0.0002 0.0004 0.0008 0.0016 0.0032 0.0064 0.0128 0.0256
# do
# 	CUDA_VISIBLE_DEVICES=3 python wd_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 256 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual			 
# done

# 8GPU
# for i in 3.2
# do
# 	CUDA_VISIBLE_DEVICES=6,7 python main.0.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay 0.0002 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
# 					--filter_bn \
# 					--filter_bias\
# 					--warm_up_epoch 15 
# done


# IITP
# for i in 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=3 python main.0.0.2.py \
# 					--dataset cifar10 \
# 					--dataroot /home/user/ssd1/dataset/ \
# 					--epochs 300 \
# 					--batch_size 1024 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.8 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
# 					--warm_up_epoch 15 
# done

############################## 200301 ###########################
# group4
# for i in 0.0001 0.0002 0.0004 0.0008 0.0016 0.0032 0.0064 0.0128 0.0256
# do
# 	CUDA_VISIBLE_DEVICES=3 python critical.0.0.1.py \
# 					--dataset cifar10 \
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

# 8gpu-b
# for i in 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=3 python critical.0.0.1.py \
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
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=3 python critical.0.0.1.py \
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
# for i in 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=3 python critical.0.0.1.py \
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
# for i in 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=3 python critical.0.0.1.py \
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

# IITP
# for i in 0.0002 0.0008 0.0032 0.0128
# do
# 	CUDA_VISIBLE_DEVICES=3 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 256 \
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

############################## 200127 ###########################
# IITP
# for i in 0.2 0.4 0.8
# do
# 	CUDA_VISIBLE_DEVICES=3 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 1024 \
# 					--weight_decay 0.0001 \
# 					--wd_off_epoch 300 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--warm_up_epoch 5
# done

############################## 200119 ###########################
# group4
# for i in 0.0128
# do
# 	CUDA_VISIBLE_DEVICES=3 python critical.0.0.1.py \
# 					--dataset cifar100 \
# 					--epochs 300 \
# 					--batch_size 32 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs/
# done

############################## 200114 ###########################
# 8GPU
# for i in 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=3 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 6000 \
# 					--epoch_step 2000 4000 \
# 					--batch_size 512 \
# 					--weight_decay $i \
# 					--wd_off_epoch 3000 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs/ \
# 					--seed 0 1 2
# done


# group4
# for i in 0.0128 0.0256
# do
# 	CUDA_VISIBLE_DEVICES=3 python critical.0.0.1.py \
# 					--dataset cifar100 \
# 					--epochs 300 \
# 					--batch_size 32 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 1.6 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs/
# done

############################## 200112 ###########################
# IITP
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=3 python critical.0.0.1.py \
# 					--dataset cifar100 \
# 					--epochs 300 \
# 					--batch_size 32 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs/
# done

####################.########## 200111 ###########################
# group4
# for i in 0.0256
# do
# 	CUDA_VISIBLE_DEVICES=3 python critical.0.0.1.py \
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

# # group4
# for i in 0.0256
# do
# 	CUDA_VISIBLE_DEVICES=3 python critical.0.0.1.py \
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

# # group4
# for i in 0.0256
# do
# 	CUDA_VISIBLE_DEVICES=3 python critical.0.0.1.py \
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

# # 8GPU
# for i in 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=3 python critical.0.0.1.py \
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

# # 8GPU
# for i in 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=3 python critical.0.0.1.py \
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
# for i in 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=3 python critical.0.0.1.py \
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

# IITP
# for i in 0.00005
# do
# 	CUDA_VISIBLE_DEVICES=3 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 1024 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs/
# done

# # IITP
# for i in 0.00005
# do
# 	CUDA_VISIBLE_DEVICES=3 python critical.0.0.1.py \
# 					--dataset cifar10 \
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

####################.########## 200108 ###########################
# group4
# for i in 0.0016 0.0032 0.0064 0.0128
# do
# 	CUDA_VISIBLE_DEVICES=3 python critical.0.0.1.py \
# 					--dataset cifar100 \
# 					--epochs 300 \
# 					--batch_size 1024 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs/
# done

# 8GPU
# for i in 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=3 python critical.0.0.1.py \
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

############################## 200106 ###########################
# IITP
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=3 python critical.0.0.1.py \
# 					--epochs 300 \
# 					--batch_size 32 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs/
# done

############################## 200104 ###########################
# # IITP
# for i in 0.
# do
# 	CUDA_VISIBLE_DEVICES=3 python critical.0.0.1.py \
# 					--epochs 300 \
# 					--batch_size 32 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs/
# done

############################## 200103 ###########################
# # 8GPU
# for i in 0.0256
# do
# 	CUDA_VISIBLE_DEVICES=3 python critical.0.0.1.py \
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
# # 8GPU - J
# for i in 0.002 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=6,7 python critical.0.0.1.py \
# 					--epochs 300 \
# 					--batch_size 1024 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs/
# done

# 8GPU
# # for i in 0.0001
# for i in 0.0006 0.0007 0.0008 0.0009 0.001
# do
# 	CUDA_VISIBLE_DEVICES=3 python critical.0.0.1.py \
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

############################## 201124 ###########################
# IITP
# for i in 75 175 275
# do
# 	CUDA_VISIBLE_DEVICES=3 python critical.0.0.1.py \
# 					--epochs 300 \
# 					--batch_size 1024 \
# 					--weight_decay 0.0005 \
# 					--wd_off_epoch $i
# done


# 8GPU
# # for i in $(seq 0 25 100)
# for i in 75 275
# do
# 	CUDA_VISIBLE_DEVICES=3 python critical.0.0.1.py \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.0005 \
# 					--wd_off_epoch $i
# done
