#!/bin/bash

############################## 220127 ###########################
# # 8gpu-ICML3
for i in 3125 6250 12500 25000 0
do
	CUDA_VISIBLE_DEVICES=6 python direction.0.0.2.py \
					--dataset cifar100 \
					--num_sample $i \
					--epochs 300 \
					--epoch_step 150 225 \
					--batch_size 128 \
					--lr 0.5657 \
					--weight_decay 0.0005657 \
					--momentum 0.9 \
					--net_type densenetBC100_GBN_invariant \
					--save_model \
					--seed 0 1 \
					--warm_up_epoch 0 \
					--save_dir './logs_inv/'
done

############################## 220126 ###########################
# 8gpu-ICML1
# for i in 12500
# do
# 	CUDA_VISIBLE_DEVICES=6 python direction.0.0.2.py \
# 					--dataset tinyimagenet \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 120 \
# 					--epoch_step 60 90 \
# 					--batch_size 256 \
# 					--lr 12.8 \
# 					--weight_decay 0.0001 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_inv_stl \
# 					--save_model \
# 					--seed 0 1 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_inv/'
# done

# # 8gpu-ICML2
# for i in 3125 6250 12500 25000 0
# do
# 	CUDA_VISIBLE_DEVICES=6 python direction.0.0.2.py \
# 					--dataset cifar100 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size 128 \
# 					--lr 0.1414 \
# 					--weight_decay 0.0001414 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN_invariant \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_inv/'
# done

# 8gpu-ICML1
# for i in 0
# do
# 	CUDA_VISIBLE_DEVICES=6 python direction.0.0.2.py \
# 					--dataset tinyimagenet \
# 					--test_freq 1 \
# 					--num_sample $i \
# 					--epochs 120 \
# 					--epoch_step 60 90 \
# 					--batch_size 256 \
# 					--lr 0.1 \
# 					--weight_decay 0.0004 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_inv_stl \
# 					--save_model \
# 					--seed 0 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_inv/'
# done

############################## 220125 ###########################
# # 8gpu-ICML3
# for i in 3125 6250 12500 25000 0
# do
# 	CUDA_VISIBLE_DEVICES=6 python direction.0.0.2.py \
# 					--dataset cifar100 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size 128 \
# 					--lr 6.4 \
# 					--weight_decay 0.0001 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN_invariant \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_inv/'
# done

# 8gpu-ICML1
# for i in 12500
# do
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.2.test.py \
# 					--dataset tinyimagenet \
# 					--test_freq 1 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size 256 \
# 					--lr 0.1 \
# 					--weight_decay 0.0001 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_inv_stl \
# 					--save_model \
# 					--seed 0 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_trash/'
# done

# 8gpu-ICML2
# for i in 3125 6250 12500 25000 50000
# do
# 	CUDA_VISIBLE_DEVICES=6 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size 128 \
# 					--lr 6.4 \
# 					--weight_decay 0.0001 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_var/'
# done

# # 8gpu-ICML2
# for i in 3125 6250 12500 25000 50000
# do
# 	CUDA_VISIBLE_DEVICES=6 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size 128 \
# 					--lr 0.1 \
# 					--weight_decay 0.0064 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_var/'
# done

# # 8gpu-ICML2
# for i in 3125 6250 12500 25000 50000
# do
# 	CUDA_VISIBLE_DEVICES=6 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size 128 \
# 					--lr 0.8 \
# 					--weight_decay 0.0008 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_var/'
# done
############################## 220124 ###########################
# 8gpu - ICML3
# for i in 3125 6250 12500 25000 50000
# do
# 	CUDA_VISIBLE_DEVICES=6 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size 128 \
# 					--lr 1.414 \
# 					--weight_decay 0.0001414 \
# 					--momentum 0 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_inv_wo_moment/'
# done

# # 8gpu - ICML3
# for i in 3125 6250 12500 25000 50000
# do
# 	CUDA_VISIBLE_DEVICES=6 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size 128 \
# 					--lr 5.657 \
# 					--weight_decay 0.0005657 \
# 					--momentum 0 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_inv_wo_moment/'
# done

# 8gpu-ICML1
# for i in 3125 6250 12500 25000 50000
# do
# 	CUDA_VISIBLE_DEVICES=6 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size 128 \
# 					--lr 16 \
# 					--weight_decay 0.0001 \
# 					--momentum 0 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_inv_wo_moment/'
# done

# 8gpu-ICML3
# for i in 4
# do
# 	CUDA_VISIBLE_DEVICES=6 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample 0 \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size 128 \
# 					--lr $i \
# 					--weight_decay 0.0001 \
# 					--momentum 0 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_inv_wo_moment_trash/'
# done

# 8gpu-ICML2
# for i in 2048
# do
# 	CUDA_VISIBLE_DEVICES=6 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample 0 \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size $i \
# 					--lr 6.4 \
# 					--weight_decay 0.0001 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_inv/'
# done

# # 8gpu-ICML2
# for i in 2048
# do
# 	CUDA_VISIBLE_DEVICES=6 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample 0 \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size $i \
# 					--lr 0.1 \
# 					--weight_decay 0.0064 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_inv/'
# done

# # 8gpu-ICML2
# for i in 2048
# do
# 	CUDA_VISIBLE_DEVICES=6 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample 0 \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size $i \
# 					--lr 0.8 \
# 					--weight_decay 0.0008 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_inv/'
# done
############################## 220124 ###########################
# 8gpu-ICML3
# for i in 3125 6250 12500 25000 
# do
# 	CUDA_VISIBLE_DEVICES=6 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size 128 \
# 					--lr 0.5657 \
# 					--weight_decay 0.0005657 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_inv/'
# done

############################## 220123 ###########################
# 8gpu-ICML3
# for i in 3125 6250 12500 25000 
# do
# 	CUDA_VISIBLE_DEVICES=6 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size 128 \
# 					--lr 0.2 \
# 					--weight_decay 0.0002 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_inv/'
# done

# 8gpu-JDG
# for i in 0.2 0.4 0.8 1.6 3.2
# do
# 	CUDA_VISIBLE_DEVICES=6 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample 6250 \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size 256 \
# 					--lr $i \
# 					--weight_decay 0.0001 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_inv/'
# done

# 8gpu-ICML
# for i in 0.0001 0.0002 0.0004 0.0008 0.0016 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=6 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample 0 \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size 512 \
# 					--lr 0.1 \
# 					--weight_decay $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_inv/'
# done

