#!/bin/bash

############################## 200603 ###########################
# DGX
for i in 1
do
	CUDA_VISIBLE_DEVICES=4,5,6,7 python invariant.1.0.2.3.py \
					--dataset imagenet \
					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
					--num_workers 40 \
					--epochs 90 \
					--epoch_step 30 60 80 \
					--batch_size 256 \
					--weight_decay 0.0001 \
					--lr 0.1 \
					--nesterov \
					--momentum 0.9 \
					--net_type resnet50_GBN_invariant2 \
					--save_model \
					--seed 0 \
					--zero_init_residual \
					--warm_up_epoch 5 \
					--alpha_sqaure $i \
					--load_dir logs_invariant_no_scale_w/imagenet/resnet50_GBN_invariant2/batch_256/invariant_normx_alpha_sq_1.0_WD_0.0001_lr0.1_eps_1e-05_zero_init_True_warm_iter_5_filter_bn_bias_False_momentum0.9_nester_True_epoch90_amp_True_seed0_save0/0/
done



############################## 200602 ###########################
# 8gpu - h
# for i in 0.5
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 python invariant.1.0.2.3.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 40 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 512 \
# 					--weight_decay 0.0001 \
# 					--lr 0.2 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--alpha_sqaure $i \
# 					--load_dir logs_invariant_no_scale_w/imagenet/resnet50_GBN_invariant2/batch_512/invariant_normx_alpha_sq_0.5_WD_0.0002_lr0.1_eps_5e-06_zero_init_True_warm_iter_0_filter_bn_bias_False_momentum0.9_nester_True_epoch10_amp_True_seed0_save0/0/
# done


# 8gpu - hong
# for i in 1
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 python invariant.1.0.2.3.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 256 \
# 					--weight_decay 0.0001 \
# 					--lr 0.1 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--alpha_sqaure $i
# done

############################## 200528 ###########################
# DGX
# for i in 1
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 python invariant.1.0.2.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay 0.0001 \
# 					--lr 6.4 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--save_dir './logs_invariant_scaler/' \
#  					--alpha_sqaure $i
# done

############################## 200527 ###########################
# # 8gpu
# for i in 0.03125
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 python invariant.1.0.2.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 2048 \
# 					--weight_decay 0.0001 \
# 					--lr 3.2 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--save_dir './logs_invariant_scaler/' \
#  					--alpha_sqaure $i
# done

############################## 200527 ###########################
# DGX
# for i in 0.25
# do
# 	CUDA_VISIBLE_DEVICES=4 python invariant.1.0.2.3.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 256 \
# 					--weight_decay 0.0001 \
# 					--lr 0.4 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN_invariant \
# 					--save_model \
# 					--seed 4 \
# 					--zero_init_residual \
# 					--warm_up_epoch 0 \
# 					--alpha_sqaure $i
# done

# DGX
# for i in 0.015625
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 python invariant.1.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay 0.0001 \
# 					--lr 6.4 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN_bn_fix \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--zero_init_residual \
# 					--warm_up_epoch 0 \
# 					--alpha_sqaure $i
# done

# 8gpu
# for i in 0.03125
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 python invariant.1.0.2.3.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 2048 \
# 					--weight_decay 0.0001 \
# 					--lr 3.2 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
# 					--warm_up_epoch 0 \
# 					--alpha_sqaure $i
# done

############################## 200526 ###########################
# 8gpu
# for i in 0.015625
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 python invariant.1.0.2.3.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay 0.0001 \
# 					--lr 6.4 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
# 					--warm_up_epoch 0 \
# 					--alpha_sqaure $i
# done

############################## 200526 ###########################
# 8gpu
# for i in 0.03125
# do
# 	CUDA_VISIBLE_DEVICES=4,5 python invariant.1.0.2.3.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay 0.0002 \
# 					--lr 3.2 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2_avg \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
# 					--warm_up_epoch 0 \
# 					--alpha_sqaure $i
# done

############################## 200525 ###########################
	# CUDA_VISIBLE_DEVICES=4,5 python invariant.1.0.2.py \
	# 				--dataset cifar10 \
	# 				--epochs 300 \
	# 				--batch_size 2048 \
	# 				--weight_decay 0.0001 \
	# 				--lr 3.2 \
	# 				--momentum 0.9 \
	# 				--net_type densenetBC100_GBN_invariant \
	# 				--save_model \
	# 				--seed 0 1 2 3 4 \
	# 				--zero_init_residual \
	# 				--warm_up_epoch 15
					
############################## 200524 ###########################
# # DGX
# for i in 0.03125
# do
# 	CUDA_VISIBLE_DEVICES=4,5 python invariant.1.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 2048 \
# 					--weight_decay 0.0001 \
# 					--lr 3.2 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
# 					--warm_up_epoch 15 \
# 					--alpha_sqaure $i
# done

############################## 200523 ###########################
# # DGX
# for i in 0.03125
# do
# 	CUDA_VISIBLE_DEVICES=4,5 python invariant.1.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay 0.0002 \
# 					--lr 3.2 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 15 \
#  					--alpha_sqaure $i
# done

# 	CUDA_VISIBLE_DEVICES=4,5 python invariant.1.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay 0.0002 \
# 					--lr 3.2 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 15
				
# 8gpu
# for i in 0.015625
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 python invariant.1.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 8192 \
# 					--weight_decay 0.0002 \
# 					--lr 6.4 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 15 \
#  					--alpha_sqaure $i
# done

# # 8gpu
# CUDA_VISIBLE_DEVICES=4,5,6,7 python invariant.1.0.2.py \
# 				--dataset cifar10 \
# 				--epochs 300 \
# 				--batch_size 8192 \
# 				--weight_decay 0.0002 \
# 				--lr 6.4 \
# 				--momentum 0.9 \
# 				--net_type resnet18_GBN_invariant2 \
# 				--save_model \
# 				--seed 0 1 2 3 4 \
# 				--zero_init_residual \
# 				--warm_up_epoch 15


############################## 200522 ###########################
# 8gpu
# for i in 0.001
# do
# 	CUDA_VISIBLE_DEVICES=4 python invariant.1.0.2.2.py \
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
# for i in 0.0016
# do
# 	CUDA_VISIBLE_DEVICES=4 python invariant.1.0.2.py \
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
# for i in 25000
# do
# 	CUDA_VISIBLE_DEVICES=4 python invariant_num.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.0002 \
# 					--lr 0.2 \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--num_sample $i \
#  					--base_norm 256 \
#  					--eps 0.00002
# done

############################## 200520 ###########################
# DGX
# for i in 1e-01
# do
# 	CUDA_VISIBLE_DEVICES=4 python invariant.1.0.2.py \
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

# DGX
# for i in 0.2
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 python invariant.1.0.2.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 512 \
# 					--weight_decay 0.0001 \
# 					--lr $i \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN_invariant2_noBN \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5
# done

############################## 200515 ###########################
# # DGX
# for i in 1e+00
# do
# 	CUDA_VISIBLE_DEVICES=4 python invariant.1.0.3.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
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
#  					--epoch_step 80 160 240 \
#  					--lr_decay 0.1 \
#  					--wd_linear 0.0005
# done

# DGX
# for i in 1e-00
# do
# 	CUDA_VISIBLE_DEVICES=4 python invariant.1.0.2.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-05 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN_invariant \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e+03 \
#  					--save_dir './logs_invariant_scaler/' \
#  					--eps 1e-04
# done

# 8gpu-h
# for i in 5e-01
# do
# 	CUDA_VISIBLE_DEVICES=4 python invariant.1.0.3.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 1e-03 \
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
# for i in 1e-00
# do
# 	CUDA_VISIBLE_DEVICES=4 python invariant.1.0.3.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-05 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e+03 \
#  					--save_dir './logs_invariant_scaler/' \
#  					--eps 1e-04
# done

# DGX
# for i in 1e-00
# do
# 	CUDA_VISIBLE_DEVICES=4 python variant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-05 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e+03 \
#  					--save_dir './logs_variant_same/' \
#  					--amp \
#  					--eps 1e-04
# done

# 8gpu-h
# for i in 1e-00
# do
# 	CUDA_VISIBLE_DEVICES=4 python invariant.1.0.3.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-05 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e+03 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e-04
# done

############################## 200513 ###########################
# 8gpu-h
# for i in 1
# do
# 	CUDA_VISIBLE_DEVICES=4 python variant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.00002 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1280 \
#  					--save_dir './logs_variant_same/' \
#  					--amp \
#  					--eps 1.00E-04 \
#  					--filter_bn_bias
# done

############################## 200512 ###########################
# 8gpu-hyejin
# for i in 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=4 python invariant_num.1.0.1.py \
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
# for i in 1e-06
# do
# 	CUDA_VISIBLE_DEVICES=4 python invariant.1.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 20  \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 0.00128 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e-10
# done

# 8GPU-hyejin
# for i in 6250
# do
# 	CUDA_VISIBLE_DEVICES=4 python invariant_num.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.0064 \
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
# 	CUDA_VISIBLE_DEVICES=4 python invariant_num.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.0016 \
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
# 	CUDA_VISIBLE_DEVICES=4 python invariant_num.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 64 \
# 					--weight_decay 0.0004 \
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
# for i in 1e-05
# do
# 	CUDA_VISIBLE_DEVICES=4 python invariant.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 64 \
# 					--weight_decay $i \
# 					--lr 1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 640 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e-04
# done

# # 8gpu-lee
# for i in 0.0016
# do
# 	CUDA_VISIBLE_DEVICES=4 python invariant_num.1.0.0.py \
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
# for i in 2
# do
# 	CUDA_VISIBLE_DEVICES=4 python invariant.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--lr 1e-05 \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 0.0128 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e-9
# done

# # 8GPU-p
# for i in 0.2
# do
# 	CUDA_VISIBLE_DEVICES=4 python invariant.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--lr 1e-04 \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 0.128 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e-8
# done

# # 8GPU-p
# for i in 0.02
# do
# 	CUDA_VISIBLE_DEVICES=4 python invariant.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--lr 0.001 \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e-7
# done

############################## 200505 ###########################
# 8gpu-p
# for i in 0.0016 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=4 python invariant_num.1.0.0.py \
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

# # 8gpu-p
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=4 python invariant_num.1.0.0.py \
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
# for i in 0.0002 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=4 python invariant_num.1.0.0.py \
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

############################## 200504 ###########################
# 8gpu-p
# for i in 2e-08
# do
# 	CUDA_VISIBLE_DEVICES=4 python invariant.1.0.1.py \
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
#  					--amp \
#  					--eps 0.1
# done

############################## 200503 ###########################
# 8gpu-p
# for i in 0.002
# do
# 	CUDA_VISIBLE_DEVICES=4 python invariant.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--lr 0.01 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 12.8 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp
# done

# # 8gpu-b
# for i in 0.002
# do
# 	CUDA_VISIBLE_DEVICES=4 python invariant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--lr 0.01 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 12.8 \
#  					--save_dir './logs_invariant_same/'
# done

############################## 200430 ###########################
# 8gpu
# for i in 0.2
# do
# 	CUDA_VISIBLE_DEVICES=4 python invariant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.0001 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0
# done

############################## 200410 ###########################
# 8gpu-p
# num_data densenet
# for i in 0.05 0.2 0.4 0.8 
# do
# 	CUDA_VISIBLE_DEVICES=4 python num_data.1.0.0.py \
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
# for i in 1.6 3.2 6.4 12.8
# do
# 	CUDA_VISIBLE_DEVICES=4 python num_data.1.0.0.py \
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

############################## 200306 ###########################
# 8gpu-b
# for i in 0.0016 0.0008 0.0004
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 python wd_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual
# done

############################## 200306 ###########################
# 8gpu-b
# for i in 0.0016
# do
# 	CUDA_VISIBLE_DEVICES=4 python wd_scaler.py \
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
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=4 python wd_scaler.py \
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
# for i in 0.0016
# do
# 	CUDA_VISIBLE_DEVICES=4 python critical.0.0.1.py \
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
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=4 python critical.0.0.1.py \
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
# for i in 0.0016
# do
# 	CUDA_VISIBLE_DEVICES=4 python critical.0.0.1.py \
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
# # 8gpu-b
# for i in 0.0016
# do
# 	CUDA_VISIBLE_DEVICES=4 python critical.0.0.1.py \
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
# for i in 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=4 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 3000 \
# 					--epoch_step 1000 2000 \
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

####################.########## 200111 ###########################
# 8GPU
# for i in 0.0016
# do
# 	CUDA_VISIBLE_DEVICES=4 python critical.0.0.1.py \
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
# for i in 0.0016
# do
# 	CUDA_VISIBLE_DEVICES=4 python critical.0.0.1.py \
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
# for i in 0.0016
# do
# 	CUDA_VISIBLE_DEVICES=4 python critical.0.0.1.py \
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
# for i in 0.0016
# do
# 	CUDA_VISIBLE_DEVICES=4 python critical.0.0.1.py \
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
# for i in 0.0128
# do
# 	CUDA_VISIBLE_DEVICES=4 python critical.0.0.1.py \
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
# for i in 0.0001 0.0002 0.0003 0.0004 0.0005
# do
# 	CUDA_VISIBLE_DEVICES=4 python critical.0.0.1.py \
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

############################## 201201 ###########################
# 8GPU
# for i in 300
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 python critical.0.0.1.py \
# 					--epochs 300 \
# 					--batch_size 8192 \
# 					--weight_decay 0.0128 \
# 					--wd_off_epoch $i \
# 					--lr 0.1 \
# 					--momentum 0.8 \
# 					--std_weight 1
# done

# # 8GPU
# for i in 300
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 python critical.0.0.1.py \
# 					--epochs 300 \
# 					--batch_size 8192 \
# 					--weight_decay 0.0128 \
# 					--wd_off_epoch $i \
# 					--lr 0.1 \
# 					--momentum 0.7 \
# 					--std_weight 1
# done

# 8GPU
# for i in 300
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 python critical.0.0.1.py \
# 					--epochs 300 \
# 					--batch_size 8192 \
# 					--weight_decay 0.0128 \
# 					--wd_off_epoch $i \
# 					--lr 0.1

# done


############################## 201124 ###########################
# 8GPU
# for i in $(seq 125 25 200)
# for i in 100
# do
# 	CUDA_VISIBLE_DEVICES=4 python critical.0.0.1.py \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.0005 \
# 					--wd_off_epoch $i
# done
