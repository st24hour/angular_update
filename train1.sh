#!/bin/bash

############################## 200527 ###########################
# # 8gpu
for i in 1
do
	CUDA_VISIBLE_DEVICES=2,3 python invariant.1.0.2.1.py \
					--dataset cifar10 \
					--epochs 300 \
					--batch_size 4096 \
					--weight_decay 0.0002 \
					--lr 6.4 \
					--momentum 0.9 \
					--net_type resnet18_GBN_invariant2 \
					--save_model \
					--seed 0 1 2 3 4 \
					--zero_init_residual \
 					--warm_up_epoch 0 \
 					--save_dir './logs_invariant_scaler/' \
 					--alpha_sqaure $i
done

# DGX
# for i in 0.125
# do
# 	CUDA_VISIBLE_DEVICES=1 python invariant.1.0.2.3.py \
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
# 					--warm_up_epoch 0 \
# 					--alpha_sqaure $i
# done

############################## 200525 ###########################
# # DGX
	# CUDA_VISIBLE_DEVICES=1 python invariant.1.0.2.py \
	# 				--dataset cifar10 \
	# 				--epochs 300 \
	# 				--batch_size 256 \
	# 				--weight_decay 0.0001 \
	# 				--lr 0.4 \
	# 				--momentum 0.9 \
	# 				--net_type densenetBC100_GBN_invariant \
	# 				--save_model \
	# 				--seed 0 1 2 3 4 \
	# 				--zero_init_residual \
	# 				--warm_up_epoch 15 
					
############################## 200523 ###########################
# # DGX
# for i in 0.5
# do
# 	CUDA_VISIBLE_DEVICES=1 python invariant.1.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.0001 \
# 					--lr 0.2 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
# 					--warm_up_epoch 15 \
# 					--alpha_sqaure $i
# done

# DGX
	# CUDA_VISIBLE_DEVICES=1 python invariant.1.0.2.py \
	# 				--dataset cifar10 \
	# 				--epochs 300 \
	# 				--batch_size 512 \
	# 				--weight_decay 0.0002 \
	# 				--lr 0.4 \
	# 				--momentum 0.9 \
	# 				--net_type resnet18_GBN_invariant2 \
	# 				--save_model \
	# 				--seed 0 1 2 3 4 \
	# 				--zero_init_residual \
 # 					--warm_up_epoch 15
 					
# 8gpu
# for i in 0.25
# do
# 	CUDA_VISIBLE_DEVICES=1 python invariant.1.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 512 \
# 					--weight_decay 0.0002 \
# 					--lr 0.4 \
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
# for i in 0.004
# do
# 	CUDA_VISIBLE_DEVICES=1 python invariant.1.0.2.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 2048 \
# 					--weight_decay 0.0001 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0
# done

# 8gpu
# for i in 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=1 python invariant.1.0.2.2.py \
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
# for i in 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=1 python invariant.1.0.2.py \
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
# for i in 3125
# do
# 	CUDA_VISIBLE_DEVICES=1 python invariant_num.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.0002 \
# 					--lr 1.6 \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--num_sample $i \
#  					--base_norm 2048 \
#  					--eps 0.00016
# done

############################## 200520 ###########################
# DGX
# for i in 1e+14
# do
# 	CUDA_VISIBLE_DEVICES=1 python invariant.1.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-19  \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e+17 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e+10
# done

# 8gpu
# for i in 1e-01
# do
# 	CUDA_VISIBLE_DEVICES=1 python invariant.1.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-04  \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant_noBN \
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
# 					--net_type resnet50_GBN_invariant2 \
# 					--save_model \
# 					--seed 1 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5
# done

############################## 200517 ###########################
# # 8gpu
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=2,3 python invariant.1.0.2.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--save_dir './logs_invariant_scaler/' \
#  					--eps 3.125e-07 \
#  					--base_norm 128
# done

############################## 200516 ###########################
# 8gpu-h
# for i in 1e-01
# do
# 	CUDA_VISIBLE_DEVICES=1 python invariant.1.0.3.1.py \
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
#  					--save_dir './logs_invariant_tuning/' \
#  					--eps 1e-05 \
#  					--epoch_step 150 225 \
#  					--lr_decay 0.1 \
#  					--wd_linear 0.0002
# done

############################## 200515 ###########################
# # DGX
# for i in 1e-02
# do
# 	CUDA_VISIBLE_DEVICES=1 python invariant.1.0.3.1.py \
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
#  					--epoch_step 225 \
#  					--lr_decay 0.1 \
#  					--wd_linear 0.0005
# done



# # DGX
# for i in 1e-01
# do
# 	CUDA_VISIBLE_DEVICES=1 python invariant.1.0.3.py \
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
#  					--epoch_step 20 40 60 80 \
#  					--lr_decay 0.2
# done

# 8gpu-h
# for i in 1e-00
# do
# 	CUDA_VISIBLE_DEVICES=1 python invariant.1.0.3.py \
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
# for i in 1e-03
# do
# 	CUDA_VISIBLE_DEVICES=1 python invariant.1.0.3.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-02 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e-00 \
#  					--save_dir './logs_invariant_scaler/' \
#  					--eps 1e-07
# done

# DGX
# for i in 1e-03
# do
# 	CUDA_VISIBLE_DEVICES=1 python variant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-02 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e-00 \
#  					--save_dir './logs_variant_same/' \
#  					--amp \
#  					--eps 1e-07
# done

# group4
# for i in 1e+09
# do
# 	CUDA_VISIBLE_DEVICES=1 python invariant.1.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-14  \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e+12 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e+05
# done

# 8gpu-h
# for i in 1e-03
# do
# 	CUDA_VISIBLE_DEVICES=1 python invariant.1.0.3.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-02 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e-00 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e-07
# done

############################## 200513 ###########################
# 8gpu-h
# for i in 0.001
# do
# 	CUDA_VISIBLE_DEVICES=1 python variant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.02 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28 \
#  					--save_dir './logs_variant_same/' \
#  					--amp \
#  					--eps 1.00E-07 \
#  					--filter_bn_bias
# done

############################## 200511 ###########################
# 8gpu-h
# for i in 1e-04
# do
# 	CUDA_VISIBLE_DEVICES=1 python invariant.1.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.2 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 0.128 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e-08
# done

############################## 200510 ###########################
# 8gpu-p
# for i in 1
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 python invariant.1.0.1.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 512 \
# 					--weight_decay 0.0002 \
# 					--lr 0.1 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN_invariant2 \
# 					--save_model \
# 					--seed $i \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--base_norm 256 \
# 					--eps 5e-06
# done

# 8gpu
# for i in 1
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 python invariant.1.0.0.py \
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
# 					--seed $i \
# 					--zero_init_residual \
# 					--warm_up_epoch 0
# done

############################## 200509 ###########################
# 8gpu-lee
# for i in 6250
# do
# 	CUDA_VISIBLE_DEVICES=1 python invariant_num.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.0002 \
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
# # DGX
# for i in 1563 3125 6250 12500 25000 50000
# do
# 	CUDA_VISIBLE_DEVICES=1 python invariant_num.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 64 \
# 					--weight_decay 0.0032 \
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
# for i in 0.01
# do
# 	CUDA_VISIBLE_DEVICES=1 python invariant.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 64 \
# 					--weight_decay $i \
# 					--lr 0.001 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 0.64 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e-07
# done

# # 8gpu-lee
# for i in 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=1 python invariant_num.1.0.0.py \
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

############################## 200505 ###########################
# group2
# for i in 0.0128 0.0064 0.0032 
# do
# 	CUDA_VISIBLE_DEVICES=1 python invariant_num.1.0.0.py \
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

# # # 8gpu-lee
# for i in 0.0128 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=1 python invariant_num.1.0.0.py \
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

# 8gpu-p
# for i in 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 python invariant.1.0.1.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 512 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--base_norm 256 \
# 					--eps 5e-06
# done

############################## 200504 ###########################
# # 8gpu-p
# for i in 20
# do
# 	CUDA_VISIBLE_DEVICES=1 python invariant.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--lr 0.000001 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 0.00128 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e-10
# done

# # 8gpu-p
# for i in 0.02
# do
# 	CUDA_VISIBLE_DEVICES=1 python invariant.1.0.1.py \
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
#  					--eps 1e-7
# done

############################## 200503 ###########################
# 8gpu-p
# for i in 0.000002
# do
# 	CUDA_VISIBLE_DEVICES=1 python invariant.1.0.1.py \
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
#  					--amp
# done

# 8gpu-b
# for i in 0.000002
# do
# 	CUDA_VISIBLE_DEVICES=1 python invariant.1.0.0.py \
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
#  					--save_dir './logs_invariant_same/'
# done


############################## 200501 ###########################
# 8gpu
# for i in 0.2
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 python invariant.1.0.0.py \
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
# 					--net_type resnet50_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5
# done

# # 8gpu-b
# for i in 3.2
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 python invariant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 2048 \
# 					--weight_decay 0.0001 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0
# done

# 8gpu-p
# for i in 6.4
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 python invariant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay 0.0001 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0
# done

# # 8gpu-p
# for i in 12.8
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python invariant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 8192 \
# 					--weight_decay 0.0001 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0
# done

# # group2
# for i in 0.8
# do
# 	CUDA_VISIBLE_DEVICES=1 python invariant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 512 \
# 					--weight_decay 0.0001 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0
# done

# # group2
# for i in 0.2
# do
# 	CUDA_VISIBLE_DEVICES=1 python invariant.1.0.0.py \
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

# 8gpu-b
# for i in 0.002
# do
# 	CUDA_VISIBLE_DEVICES=1 python invariant.1.0.0.py \
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
# 8gpu-p
# for i in 6.4
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 python invariant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
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
# 8gpu-b
# for i in 6.4
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 python invariant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 8192 \
# 					--weight_decay 0.0002 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0
# done

# group2
# for i in 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=1 python invariant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
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

# # 8gpu
# for i in 0.0004
# do
# 	CUDA_VISIBLE_DEVICES=1 python invariant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 256 \
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
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=1 python invariant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 2048 \
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

# # group2
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=1 python invariant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 1 \
# 					--batch_size 1024 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 128 \
#  					--epoch_step 1
# done

############################## 200419 ###########################
# # 8gpu
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 python scaler.1.3.1.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 8192 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN_invariant \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--base_norm 256 \
# 					--warm_up_epoch 0
# done

############################## 200419 ###########################
# # 8gpu
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 python scaler.1.3.1.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 8192 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN_invariant \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--base_norm 256 \
# 					--warm_up_epoch 0 \
# 					--filter_bn_bias
# done

############################## 200409 ###########################
# # # 8gpu
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 python scaler.1.3.1.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 8192 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN_invariant \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 0 \
# 					--filter_bn_bias
# done

############################## 200414 ###########################
# 8gpu-p
# num_data densenet
# for i in 3.2 6.4
# do
# 	CUDA_VISIBLE_DEVICES=1 python num_data.1.0.0.py \
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

############################## 200410 ###########################
# 8gpu-p
# for i in 0.2
# do
# 	CUDA_VISIBLE_DEVICES=1 python scaler.1.2.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 256 \
# 					--weight_decay 0.0002 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_lr 0.1
# done

# # 8gpu-p
# for i in 0.4
# do
# 	CUDA_VISIBLE_DEVICES=1 python scaler.1.2.0.py \
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
#  					--warm_up_epoch 0 \
#  					--base_lr 0.1
# done

# # 8gpu-p
# for i in 0.8
# do
# 	CUDA_VISIBLE_DEVICES=1 python scaler.1.2.0.py \
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
#  					--warm_up_epoch 0 \
#  					--base_lr 0.1
# done

# # 8gpu-p
# for i in 1.6
# do
# 	CUDA_VISIBLE_DEVICES=1 python scaler.1.2.0.py \
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
#  					--warm_up_epoch 0 \
#  					--base_lr 0.1
# done
############################## 200409 ###########################
# # 8gpu
# for i in 0.2
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 python scaler.1.0.0.py \
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
# 					--net_type resnet50_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--filter_bn \
# 					--filter_bias
# done

############################## 200407 ###########################
# group2-1
# for i in 0.05 0.2 0.4 0.8 1.6 3.2 6.4 12.8
# do
# 	CUDA_VISIBLE_DEVICES=1 python num_data.1.0.0.py \
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
#  					--num_sample 25000
# done

############################## 200405 ###########################
# # group2
# for i in 0.0004
# do
# 	CUDA_VISIBLE_DEVICES=1 python scaler.1.0.0.py \
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
#  					--warm_up_epoch 0 \
#  					--base_norm 64
# done

# # group2
# for i in 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=1 python scaler.1.0.0.py \
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
#  					--warm_up_epoch 0 \
#  					--base_norm 64
# done

############################## 200403 ###########################
# group2
# for i in 0.0004
# do
# 	CUDA_VISIBLE_DEVICES=1 python num_data.1.0.0.py \
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

# # group2
# for i in 0.4
# do
# 	CUDA_VISIBLE_DEVICES=1 python lr_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 256 \
# 					--weight_decay 0.0001 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0
# done

# # group2
# for i in 0.8
# do
# 	CUDA_VISIBLE_DEVICES=1 python lr_scaler.py \
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
#  					--warm_up_epoch 0
# done

############################## 200331 ###########################
# group2
# for i in 0.0004 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=1 python num_data.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 64 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--num_sample 12500
# done

# # group2
# for i in 0.0004 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=1 python num_data.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 64 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--num_sample 6250
# done

############################## 200331 ###########################
# 8gpu
# for i in 0.1 0.2 0.4 0.8 1.6 3.2 6.4 12.8
# do
# 	CUDA_VISIBLE_DEVICES=1 python num_data.1.0.0.py \
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
#  					--num_sample 6250
# done

############################## 200326 ###########################
# group2
# for i in 0.0001 0.0002 0.004 0.008 0.0016 0.0032 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=1 python num_data.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 64 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--num_sample 12500
# done

# # group2
# for i in 0.0001 0.0002 0.004 0.008 0.0016 0.0032 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=1 python num_data.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 64 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--num_sample 6250
# done

############################## 200325 ###########################
# # IITP
# for i in 0.0004 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=1 python wd_scaler.py \
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

# group2
# for i in 0.0016
# do
# 	CUDA_VISIBLE_DEVICES=1 python scaler.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 1024 \
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

############################## 200324 ###########################
# group2
# for i in 0.2
# do
# 	CUDA_VISIBLE_DEVICES=1 python lr_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.0001 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 15
# done

############################## 200321 ###########################
# group2
# for i in 0.2
# do
# 	CUDA_VISIBLE_DEVICES=1 python lr_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 256 \
# 					--weight_decay 0.0002 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 15
# done

# # group2
# for i in 0.1
# do
# 	CUDA_VISIBLE_DEVICES=1 python lr_scaler.py \
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
# 					--filter_bn \
# 					--filter_bias	
# done

# IITP
# for i in 0.8
# do
# 	CUDA_VISIBLE_DEVICES=1 python lr_scaler.py \
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
#  					--warm_up_epoch 15
# done

############################## 200320 ###########################
# IITP
# for i in 1.6
# do
# 	CUDA_VISIBLE_DEVICES=1 python lr_scaler.py \
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
#  					--warm_up_epoch 15
# done

# group2
# for i in 0.2
# do
# 	CUDA_VISIBLE_DEVICES=1 python lr_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.0001 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 15 \
# 					--filter_bn \
# 					--filter_bias	
# done

############################## 200319 ###########################
# 8gpu
# for i in 6.4
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 python lr_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 8192 \
# 					--weight_decay 0.0002 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 15
# done

# # 8gpu
# for i in 3.2
# do
# 	CUDA_VISIBLE_DEVICES=4,5 python lr_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay 0.0002 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 15
# done

# IITP
# for i in 0.8
# do
# 	CUDA_VISIBLE_DEVICES=1 python lr_scaler.py \
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
#  					--warm_up_epoch 15 \
# 					--filter_bn \
# 					--filter_bias	
# done

############################## 200319 ###########################
# 8gpu
# for i in 0.0004 0.0002 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 python wd_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 2048 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual
# done

############################## 200310 ###########################
# group2
# for i in 0.0001 0.0002 0.0004 0.0008 0.0016 0.0032 0.0064 0.0128 0.0256
# do
# 	CUDA_VISIBLE_DEVICES=1 python wd_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual
# done

############################## 200308 ###########################
# group2
# for i in 0.1
# do
# 	CUDA_VISIBLE_DEVICES=1 python lr_scaler.py \
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
# 					--filter_bn \
# 					--filter_bias	
# done

############################## 200306 ###########################
# 8gpu-ha
# for i in 0.0002 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 python wd_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 8192 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--save_dir ./logs/ \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual
# done

############################## 200305 ###########################
# 8gpu
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=2,3 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--save_dir ./logs/ \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
# 					--filter_bn
# done

############################## 200304 ###########################
# 8gpu
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=2,3 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--save_dir ./logs/ \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
# 					--filter_bn
# done

# group2
# for i in 0.0001 0.0002 0.0004 0.0008 0.0016 0.0032 0.0064 0.0128 0.0256
# do
# 	CUDA_VISIBLE_DEVICES=1 python critical.0.0.1.py \
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

############################## 200301 ###########################
# 8gpu-b
# for i in 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=1 python critical.0.0.1.py \
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
# # 8gpu-o
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--dataroot /home/user/ssd1/dataset/ \
# 					--epochs 300 \
# 					--batch_size 4096 \
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

# # 8gpu-o
# for i in 0.0256
# do
# 	CUDA_VISIBLE_DEVICES=7 python critical.0.0.1.py \
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

# 8gpu-b
# for i in 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=1 python critical.0.0.1.py \
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
# for i in 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=1 python critical.0.0.1.py \
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
# for i in 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=1 python critical.0.0.1.py \
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


# 8gpu-o
# for i in 0.0032 0.0016 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--dataroot /home/user/ssd1/dataset/ \
# 					--epochs 300 \
# 					--batch_size 4096 \
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

############################## 200226 ###########################
# IITP
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=2,3 python main.0.0.2.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/data/ILSVRC2012/ \
# 					--num_workers 16 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 256 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 0 
# done

############################## 200224 ###########################
# 8GPU
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
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
# for i in 1.6 0.8
# do
# 	CUDA_VISIBLE_DEVICES=1 python main.0.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 2048 \
# 					--weight_decay 0.0002 \
# 					--wd_off_epoch 300 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--seed 0 \
# 					--warm_up_epoch 15 \
# 					--zero_init_residual
# done
		
############################## 200223 ###########################
# 8GPU
# for i in 3.2 1.6 0.8
# do
# 	CUDA_VISIBLE_DEVICES=2,3 python main.0.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay 0.0002 \
# 					--wd_off_epoch 300 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--seed 0 \
# 					--warm_up_epoch 15
# done

############################## 200223 ###########################
# # IITP
# for i in 3.2 1.6 0.8
# do
# 	CUDA_VISIBLE_DEVICES=2,3 python main.0.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 2048 \
# 					--weight_decay 0.0001 \
# 					--wd_off_epoch 300 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--seed 0 \
# 					--warm_up_epoch 15 \
# 					--filter_bias_and_bn
# done

############################## 200221 ###########################
# 8GPU
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=1 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 64 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs/ \
# 					--weight_multiplier 1
# done

############################## 200127 ###########################
# # IITP
# for i in 0.2 0.4 0.8
# do
# 	CUDA_VISIBLE_DEVICES=1 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 256 \
# 					--weight_decay 0.0001 \
# 					--wd_off_epoch 300 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--warm_up_epoch 5
# done

############################## 200126 ###########################
# 8GPU
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=2,3,4,5 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 2048 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 3.2 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--warm_up_epoch 5
# done

############################## 200121 ###########################
# # 8GPU
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=2 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 512 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.8 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs/
# done

############################## 200119 ###########################
# group4
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=2 python critical.0.0.1.py \
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
# 	CUDA_VISIBLE_DEVICES=1 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 3000 \
# 					--epoch_step 1000 2000 \
# 					--batch_size 256 \
# 					--weight_decay $i \
# 					--wd_off_epoch 3000 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs/ \
# 					--seed 0 1 2
# done

############################## 200112 ###########################

# group2
# for i in 0.0002 0.0004
# do
# 	CUDA_VISIBLE_DEVICES=1 python critical.0.0.1.py \
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

# IITP
# for i in 0.00005
# do
# 	CUDA_VISIBLE_DEVICES=1 python critical.0.0.1.py \
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
# 8GPU
# for i in 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=1 python critical.0.0.1.py \
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
# for i in 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=1 python critical.0.0.1.py \
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
# for i in 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=1 python critical.0.0.1.py \
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
# 	CUDA_VISIBLE_DEVICES=1 python critical.0.0.1.py \
# 					--dataset cifar10 \
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

############################## 200110 ###########################
# # group2
# for i in 0.0002 0.00005 0.0256
# do
# 	CUDA_VISIBLE_DEVICES=1 python critical.0.0.1.py \
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

############################## 200109 ###########################
# IITP
# for i in 0.0128
# do
# 	CUDA_VISIBLE_DEVICES=1 python critical.0.0.1.py \
# 					--dataset cifar10 \
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

####################.########## 200108 ###########################
# group4
# for i in 0.0256
# do
# 	CUDA_VISIBLE_DEVICES=1 python critical.0.0.1.py \
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
# for i in 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=1 python critical.0.0.1.py \
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
# group2
# for i in 0.0256
# do
# 	CUDA_VISIBLE_DEVICES=1 python critical.0.0.1.py \
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

# # group4
# for i in 0.0064 0.0128 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=1,2 python critical.0.0.1.py \
# 					--dataset cifar100 \
# 					--epochs 300 \
# 					--batch_size 2048 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs/
# done

# # IITP
# for i in 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=1 python critical.0.0.1.py \
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
# for i in 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=1 python critical.0.0.1.py \
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
# # group4
# for i in 0.0064 0.0128
# do
# 	CUDA_VISIBLE_DEVICES=1 python critical.0.0.1.py \
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

# # group2
# for i in 0.0128 0.0256 
# do
# 	CUDA_VISIBLE_DEVICES=1 python critical.0.0.1.py \
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


############################## 201225 ###########################
# # 8GPU - J
# for i in 0.0004 0.0008 
# do
# 	CUDA_VISIBLE_DEVICES=2,3 python critical.0.0.1.py \
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
# 	CUDA_VISIBLE_DEVICES=1 python critical.0.0.1.py \
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


############################## 201215 ###########################
# 8GPU
# for i in 0.02 0.025
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 python critical.0.0.1.py \
# 					--epochs 300 \
# 					--batch_size 8192 \
# 					--weight_decay 0.0128 \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--std_weight 1 \
# 					--norm_grad_ratio $i \
# 					--weight_multiplier 0.1
# done

# for i in 0.02
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 python critical.0.0.1.py \
# 					--epochs 300 \
# 					--batch_size 8192 \
# 					--weight_decay 0.0128 \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--std_weight 1 \
# 					--norm_grad_ratio $i \
# 					--weight_multiplier 0.2
# done

############################## 201126 ###########################
# 8GPU
# for i in 50
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 python critical.0.0.1.py \
# 					--epochs 300 \
# 					--batch_size 8192 \
# 					--weight_decay 0.0128 \
# 					--wd_off_epoch $i
# done


############################## 201125 ###########################
# 8GPU
# for i in 25 225
# do
# 	CUDA_VISIBLE_DEVICES=1 python critical.0.0.1.py \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.0001 \
# 					--wd_off_epoch $i
# done


############################## 201124 ###########################
# IITP
# for i in 25 125 225
# do
# 	CUDA_VISIBLE_DEVICES=1 python critical.0.0.1.py \
# 					--epochs 300 \
# 					--batch_size 1024 \
# 					--weight_decay 0.0005 \
# 					--wd_off_epoch $i
# done


# 8GPU
# # for i in $(seq 125 25 200)
# for i in 25 225 
# do
# 	CUDA_VISIBLE_DEVICES=1 python critical.0.0.1.py \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.0005 \
# 					--wd_off_epoch $i
# done
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         