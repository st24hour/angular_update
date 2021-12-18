# #!/bin/bash

############################## 200527 ###########################
# 8gpu
for i in 0.125
do
	CUDA_VISIBLE_DEVICES=6 python invariant.1.0.2.3.py \
					--dataset cifar10 \
					--epochs 300 \
					--batch_size 512 \
					--weight_decay 0.0001 \
					--lr 0.8 \
					--momentum 0.9 \
					--net_type densenetBC100_GBN_invariant \
					--save_model \
					--seed 3 \
					--zero_init_residual \
					--warm_up_epoch 0 \
					--alpha_sqaure $i
done

# DGX
# for i in 0.5
# do
# 	CUDA_VISIBLE_DEVICES=6 python invariant.1.0.2.3.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.0001 \
# 					--lr 0.2 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN_invariant \
# 					--save_model \
# 					--seed 4 \
# 					--zero_init_residual \
# 					--warm_up_epoch 0 \
# 					--alpha_sqaure $i
# done

############################## 200526 ###########################
# 8gpu
# for i in 0.0625
# do
# 	CUDA_VISIBLE_DEVICES=6 python invariant.1.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 2048 \
# 					--weight_decay 0.0002 \
# 					--lr 1.6 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2_avg \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
# 					--warm_up_epoch 0 \
# 					--alpha_sqaure $i
# done

############################## 200525 ###########################
# # DGX
	# CUDA_VISIBLE_DEVICES=6 python invariant.1.0.2.py \
	# 				--dataset cifar10 \
	# 				--epochs 300 \
	# 				--batch_size 512 \
	# 				--weight_decay 0.0001 \
	# 				--lr 0.8 \
	# 				--momentum 0.9 \
	# 				--net_type densenetBC100_GBN_invariant \
	# 				--save_model \
	# 				--seed 0 1 2 3 4 \
	# 				--zero_init_residual \
	# 				--warm_up_epoch 15 
					
############################## 200522 ###########################
# 8gpu
# for i in 0.002
# do
# 	CUDA_VISIBLE_DEVICES=6 python invariant.1.0.2.2.py \
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

# ############################## 200515 ###########################
# DGX
# for i in 1e+02
# do
# 	CUDA_VISIBLE_DEVICES=6 python invariant.1.0.2.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-07 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN_invariant \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e+05 \
#  					--save_dir './logs_invariant_scaler/' \
#  					--eps 1e-02
# done

# 8gpu-h
# for i in 2e-01
# do
# 	CUDA_VISIBLE_DEVICES=6 python invariant.1.0.3.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 4e-04 \
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
# for i in 1e+02
# do
# 	CUDA_VISIBLE_DEVICES=6 python invariant.1.0.2.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-07 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e+05 \
#  					--save_dir './logs_invariant_scaler/' \
#  					--eps 1e-02
# done

# # DGX
# for i in 1e+02
# do
# 	CUDA_VISIBLE_DEVICES=6 python variant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-07 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e+05 \
#  					--save_dir './logs_variant_same/' \
#  					--amp \
#  					--eps 1e-02
# done


# 8gpu-h
# for i in 1e+02
# do
# 	CUDA_VISIBLE_DEVICES=6 python invariant.1.0.3.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-07 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e+05 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e-02
# done

############################## 200513 ###########################
# 8gpu-h
# for i in 100
# do
# 	CUDA_VISIBLE_DEVICES=6 python variant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-07 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 128000 \
#  					--save_dir './logs_variant_same/' \
#  					--amp \
#  					--eps 1.00E-02 \
#  					--filter_bn_bias
# done

############################## 200512 ###########################
# 8gpu-hyejin
# for i in 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=6 python invariant_num.1.0.1.py \
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
# for i in 0.1
# do
# 	CUDA_VISIBLE_DEVICES=6 python invariant.1.0.2.py \
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
#  					--base_norm 128 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 0.00001
# done

# 8GPU-hyejin
# for i in 6250
# do
# 	CUDA_VISIBLE_DEVICES=6 python invariant_num.1.0.1.py \
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

############################## 200509 ###########################
# 8gpu-lee
# for i in 6250
# do
# 	CUDA_VISIBLE_DEVICES=6 python invariant_num.1.0.0.py \
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

############################## 200508 ###########################
# DGX
# for i in 1563 3125 6250 12500 25000 50000
# do
# 	CUDA_VISIBLE_DEVICES=6 python invariant_num.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 64 \
# 					--weight_decay 0.0001 \
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
# for i in 1e-07
# do
# 	CUDA_VISIBLE_DEVICES=6 python invariant.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 64 \
# 					--weight_decay $i \
# 					--lr 100 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 64000 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e-02
# done

# # 8gpu-lee
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=6 python invariant_num.1.0.0.py \
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
# for i in 2e-06
# do
# 	CUDA_VISIBLE_DEVICES=6 python invariant.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--lr 10 \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 12800 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e-3
# done

# # 8GPU-p
# for i in 2e-07
# do
# 	CUDA_VISIBLE_DEVICES=6 python invariant.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--lr 100 \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 128000 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e-2
# done

# # 8GPU-p
# for i in 2e-08
# do
# 	CUDA_VISIBLE_DEVICES=6 python invariant.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--lr 1000 \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1280000 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e-1
# done

############################## 200505 ###########################
# 8gpu-p
# for i in 0.0016 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=6 python invariant_num.1.0.0.py \
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
# for i in 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=6 python invariant_num.1.0.0.py \
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
# for i in 2e-10
# do
# 	CUDA_VISIBLE_DEVICES=6 python invariant.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--lr 100000 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 128000000 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 10
# done

############################## 200503 ###########################
# 8gpu-p
# for i in 0.2
# do
# 	CUDA_VISIBLE_DEVICES=6 python invariant.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--lr 0.0001 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 0.128 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e-8
# done

# # 8gpu-b
# for i in 0.2
# do
# 	CUDA_VISIBLE_DEVICES=6 python invariant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--lr 0.0001 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 0.128 \
#  					--save_dir './logs_invariant_same/'
# done

############################## 200429 ###########################
# # 8gpu
# for i in 0.4
# do
# 	CUDA_VISIBLE_DEVICES=6 python invariant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 512 \
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
# for i in 0.8
# do
# 	CUDA_VISIBLE_DEVICES=6 python invariant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 1024 \
# 					--weight_decay 0.0002 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0
# done

############################## 200414 ###########################
# # 8gpu-p
# # num_data densenet
# for i in 0.8 
# do
# 	CUDA_VISIBLE_DEVICES=6 python num_data.1.0.0.py \
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

############################## 200410 ###########################
# 8gpu-p
# for i in 3.2
# do
# 	CUDA_VISIBLE_DEVICES=6,7 python scaler.1.2.0.py \
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
#  					--warm_up_epoch 0 \
#  					--base_lr 0.1
# done

############################## 200331 ###########################
# 8gpu
# for i in 0.8 1.6 3.2
# do
# 	CUDA_VISIBLE_DEVICES=6 python num_data.1.0.0.py \
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

############################## 200327 ###########################
# 8gpu-b
# for i in 0.0004 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=6 python num_data.1.0.0.py \
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
#  					--num_sample 1563
# done

############################## 200307 ###########################
# 8gpu-o
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=6,7 python wd_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4
# done

# 8gpu-h
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=6,7 python wd_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
# 					--weight_multiplier 1
# done


############################## 200306 ###########################
# 8gpu-o
# for i in 1.6
# do
# 	CUDA_VISIBLE_DEVICES=6 python lr_scaler.py \
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
#  					--warm_up_epoch 15 \
# 					--filter_bn \
# 					--filter_bias	
# done

# 8gpu-b
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=6 python wd_scaler.py \
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
# for i in 0.0004
# do
# 	CUDA_VISIBLE_DEVICES=6 python wd_scaler.py \
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
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=6 python critical.0.0.1.py \
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
# for i in 0.0256
# do
# 	CUDA_VISIBLE_DEVICES=6 python critical.0.0.1.py \
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
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=6 python critical.0.0.1.py \
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
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=6 python critical.0.0.1.py \
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

####################.########## 200111 ###########################
# 8GPU
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=6 python critical.0.0.1.py \
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
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=6 python critical.0.0.1.py \
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
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=6 python critical.0.0.1.py \
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
# 8GPU
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=6 python critical.0.0.1.py \
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
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=6 python critical.0.0.1.py \
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
# for i in 0.0001 0.0002 0.0004 0.0008 0.001
# do
# 	CUDA_VISIBLE_DEVICES=6 python critical.0.0.1.py \
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

############################## 201125 ###########################
# 8GPU
# for i in 150 
# do
# 	CUDA_VISIBLE_DEVICES=6 python critical.0.0.1.py \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.0001 \
# 					--wd_off_epoch $i
# done


############################## 201124 ###########################
# 8GPU
# for i in $(seq 0 25 150)
# for i in 150
# do
# 	CUDA_VISIBLE_DEVICES=6 python critical.0.0.1.py \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.0005 \
# 					--wd_off_epoch $i
# done
