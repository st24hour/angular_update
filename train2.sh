# #!/bin/bash

############################## 200601 ###########################
# group4
for i in 0.5
do
	CUDA_VISIBLE_DEVICES=2,3 python invariant.1.0.2.3.py \
					--dataset imagenet \
					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
					--num_workers 40 \
					--epochs 90 \
					--epoch_step 30 60 80 \
					--batch_size 512 \
					--weight_decay 0.0001 \
					--lr 0.2 \
					--nesterov \
					--momentum 0.9 \
					--net_type resnet50_GBN_invariant2 \
					--save_model \
					--seed 0 \
					--zero_init_residual \
					--warm_up_epoch 0 \
					--alpha_sqaure $i
done

############################## 200527 ###########################
# DGX
# for i in 0.5
# do
# 	CUDA_VISIBLE_DEVICES=2 python invariant.1.0.2.3.py \
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
# 					--warm_up_epoch 0 \
# 					--alpha_sqaure $i
# done

# 8gpu
# for i in 0.0625
# do
# 	CUDA_VISIBLE_DEVICES=2,3 python invariant.1.0.2.3.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 1024 \
# 					--weight_decay 0.0001 \
# 					--lr 1.6 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
# 					--warm_up_epoch 0 \
# 					--alpha_sqaure $i
# done

############################## 200525 ###########################
# DGX
	# CUDA_VISIBLE_DEVICES=2 python invariant.1.0.2.py \
	# 				--dataset cifar10 \
	# 				--epochs 300 \
	# 				--batch_size 1024 \
	# 				--weight_decay 0.0001 \
	# 				--lr 1.6 \
	# 				--momentum 0.9 \
	# 				--net_type densenetBC100_GBN_invariant \
	# 				--save_model \
	# 				--seed 0 1 2 3 4 \
	# 				--zero_init_residual \
	# 				--warm_up_epoch 15

############################## 200523 ###########################
# # DGX
# for i in 0.25
# do
# 	CUDA_VISIBLE_DEVICES=2 python invariant.1.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 256 \
# 					--weight_decay 0.0001 \
# 					--lr 0.4 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
# 					--warm_up_epoch 15 \
# 					--alpha_sqaure $i
# done

# DGX
	# CUDA_VISIBLE_DEVICES=2 python invariant.1.0.2.py \
	# 				--dataset cifar10 \
	# 				--epochs 300 \
	# 				--batch_size 1024 \
	# 				--weight_decay 0.0002 \
	# 				--lr 0.8 \
	# 				--momentum 0.9 \
	# 				--net_type resnet18_GBN_invariant2 \
	# 				--save_model \
	# 				--seed 0 1 2 3 4 \
	# 				--zero_init_residual \
 # 					--warm_up_epoch 15
 					
# 8gpu
# for i in 0.125
# do
# 	CUDA_VISIBLE_DEVICES=2 python invariant.1.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 1024 \
# 					--weight_decay 0.0002 \
# 					--lr 0.8 \
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
# for i in 0.00004
# do
# 	CUDA_VISIBLE_DEVICES=2 python invariant.1.0.2.2.py \
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
# for i in 0.0004
# do
# 	CUDA_VISIBLE_DEVICES=2 python invariant.1.0.2.py \
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
# for i in 6250
# do
# 	CUDA_VISIBLE_DEVICES=2 python invariant_num.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.0002 \
# 					--lr 0.8 \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--num_sample $i \
#  					--base_norm 1024 \
#  					--eps 0.00008
# done

############################## 200520 ###########################
# DGX
# for i in 1e+15
# do
# 	CUDA_VISIBLE_DEVICES=2 python invariant.1.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-20  \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e+18 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e+11
# done

# 8gpu
# for i in 1e+01
# do
# 	CUDA_VISIBLE_DEVICES=2 python invariant.1.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-06  \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant_noBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e+04 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e-03
# done

############################## 200516 ###########################
# 8gpu-h
# for i in 1e-01
# do
# 	CUDA_VISIBLE_DEVICES=2 python invariant.1.0.3.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-04 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant_avg \
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

# ############################## 200515 ###########################
# 8gpu-h
# for i in 1e-01
# do
# 	CUDA_VISIBLE_DEVICES=2 python invariant.1.0.3.1.py \
# 					--dataset cifar100 \
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
#  					--epoch_step 50 150 250 \
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
#  					--epoch_step 20 40 60 80 \
#  					--lr_decay 0.2
# done

# # DGX
# for i in 1e-02
# do
# 	CUDA_VISIBLE_DEVICES=2 python invariant.1.0.2.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-03 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN_invariant \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e+01 \
#  					--save_dir './logs_invariant_scaler/' \
#  					--eps 1e-06
# done

# 8gpu-h
# for i in 5e-01
# do
# 	CUDA_VISIBLE_DEVICES=2 python invariant.1.0.3.py \
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
# for i in 1e-02
# do
# 	CUDA_VISIBLE_DEVICES=2 python invariant.1.0.3.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-03 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e+01 \
#  					--save_dir './logs_invariant_scaler/' \
#  					--eps 1e-06
# done

# DGX
# for i in 1e-02
# do
# 	CUDA_VISIBLE_DEVICES=2 python variant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-03 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e+01 \
#  					--save_dir './logs_variant_same/' \
#  					--amp \
#  					--eps 1e-06
# done

# samsung
# for i in 1e+11
# do
# 	CUDA_VISIBLE_DEVICES=2 python invariant.1.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-16  \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e+14 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e+07
# done

# group4
# for i in 1e+10
# do
# 	CUDA_VISIBLE_DEVICES=2 python invariant.1.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-15  \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e+13 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e+06
# done

# 8gpu-h
# for i in 1e-02
# do
# 	CUDA_VISIBLE_DEVICES=2 python invariant.1.0.3.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-03 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e+01 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e-06
# done

############################## 200514 ###########################
# samsung
# for i in 1e-01
# do
# 	CUDA_VISIBLE_DEVICES=2 python invariant.1.0.3.py \
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
#  					--eps 1e-05 \
#  					--wd_linear 0.0005
# done

# 8gpu-h
# for i in 1e-02
# do
# 	CUDA_VISIBLE_DEVICES=2 python invariant.1.0.3.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-03 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e+01 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e-06
# done

############################## 200513 ###########################
# 8gpu-h
# for i in 0.01
# do
# 	CUDA_VISIBLE_DEVICES=2 python variant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.002 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 12.8 \
#  					--save_dir './logs_variant_same/' \
#  					--amp \
#  					--eps 1.00E-06 \
#  					--filter_bn_bias
# done

############################## 200512 ###########################


############################## 200511 ###########################
# 8gpu-h
# for i in 1000
# do
# 	CUDA_VISIBLE_DEVICES=2 python invariant.1.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-08  \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1280000 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 0.1
# done

# samsung
# for i in 0.0064 0.0128
# do
# 	CUDA_VISIBLE_DEVICES=2 python invariant_num.1.0.1.py \
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
# for i in 6250
# do
# 	CUDA_VISIBLE_DEVICES=2 python invariant_num.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.0128 \
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
# 	CUDA_VISIBLE_DEVICES=2 python invariant_num.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.0004 \
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
# 	CUDA_VISIBLE_DEVICES=2 python invariant_num.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 64 \
# 					--weight_decay 0.0016 \
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
# for i in 0.001
# do
# 	CUDA_VISIBLE_DEVICES=2 python invariant.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 64 \
# 					--weight_decay $i \
# 					--lr 0.01 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 6.4 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e-06
# done

# # 8gpu-lee
# for i in 0.0004
# do
# 	CUDA_VISIBLE_DEVICES=2 python invariant_num.1.0.0.py \
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

# # group4
# for i in 0.0128 0.0064 0.0032 0.0016
# do
# 	CUDA_VISIBLE_DEVICES=2 python invariant_num.1.0.0.py \
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
# for i in 0.0128 0.0064 0.0032 0.0016
# do
# 	CUDA_VISIBLE_DEVICES=2 python invariant_num.1.0.0.py \
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
# for i in 0.0032 0.0016
# do
# 	CUDA_VISIBLE_DEVICES=2 python invariant_num.1.0.0.py \
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
# 	CUDA_VISIBLE_DEVICES=2 python invariant_num.1.0.0.py \
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
#  					--num_sample 1563
# done

############################## 200504 ###########################
# 8gpu-p
# for i in 0.002
# do
# 	CUDA_VISIBLE_DEVICES=2 python invariant.1.0.1.py \
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
#  					--amp \
#  					--eps 1e-6
# done

# # 8gpu-p
# for i in 0.00002
# do
# 	CUDA_VISIBLE_DEVICES=2 python invariant.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--lr 1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1280 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e-4
# done

############################## 200503 ###########################
# 8gpu-p
# for i in 0.0000002
# do
# 	CUDA_VISIBLE_DEVICES=2 python invariant.1.0.1.py \
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
#  					--amp
# done

# group4
# for i in 0.00002
# do
# 	CUDA_VISIBLE_DEVICES=2 python invariant.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--lr 1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1280 \
#  					--save_dir './logs_invariant_same/' \
#  					--eps 1e-7
# done

# 8gpu-b
# for i in 0.0000002
# do
# 	CUDA_VISIBLE_DEVICES=2 python invariant.1.0.0.py \
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
#  					--save_dir './logs_invariant_same/'
# done



############################## 200429 ###########################
# # 8gpu
# for i in 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=2 python invariant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 512 \
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
# 	CUDA_VISIBLE_DEVICES=2 python invariant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 1024 \
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

############################## 200429 ###########################
# # group2
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=2 python invariant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 6 \
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
#  					--epoch_step 6
# done

############################## 200410 ###########################
# # 8gpu-p
# # num_data densenet
# for i in 0.05 0.1 0.2 0.4 0.8 1.6 3.2 6.4
# do
# 	CUDA_VISIBLE_DEVICES=2 python num_data.1.0.0.py \
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
#  					--num_sample 1563
# done

# # 8gpu-p
# # num_data densenet
# for i in 0.05 0.1 0.2 0.4 0.8 1.6 3.2 6.4
# do
# 	CUDA_VISIBLE_DEVICES=2 python num_data.1.0.0.py \
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
#  					--num_sample 12500
# done

############################## 200403 ###########################
# group2
# for i in 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=2 python num_data.1.0.0.py \
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
# for i in 0.1 0.2 0.4 0.8 1.6 3.2 6.4 12.8
# do
# 	CUDA_VISIBLE_DEVICES=2 python num_data.1.0.0.py \
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
#  					--num_sample 12500
# done


# IITP
# for i in 0.0008 0.0016 0.0032 0.0064 0.0128
# do
# 	CUDA_VISIBLE_DEVICES=2 python num_data.1.0.0.py \
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
#  					--num_sample 6250
# done

############################## 200329 ###########################
# IITP
# for i in 0.0004 0.0008 0.0016 0.0032 0.0064 0.0128 
# do
# 	CUDA_VISIBLE_DEVICES=2 python num_data.1.0.0.py \
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
#  					--num_sample 3125
# done

# # IITP
# for i in 0.0001 0.0002 0.0004 0.0008 0.0016 0.0032 0.0064 0.0128
# do
# 	CUDA_VISIBLE_DEVICES=2 python num_data.1.0.0.py \
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
#  					--num_sample 6250
# done

############################## 200325 ###########################
# group4
# for i in 0.0256
# do
# 	CUDA_VISIBLE_DEVICES=2 python wd_scaler.py \
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

# IITP
# for i in 0.0016 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=2 python wd_scaler.py \
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

############################## 200324 ###########################
# group4
# for i in 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=2 python wd_scaler.py \
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

############################## 200321 ###########################
# IITP
# for i in 0.4
# do
# 	CUDA_VISIBLE_DEVICES=2 python lr_scaler.py \
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
#  					--warm_up_epoch 15
# done


############################## 200312 ###########################
# # IITP
# for i in 0.0001 0.0002 0.0004 0.0008 0.0016 0.0032 0.0064 0.0128 0.0256
# do
# 	CUDA_VISIBLE_DEVICES=2 python wd_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 1024 \
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
# for i in 0.8
# do
# 	CUDA_VISIBLE_DEVICES=2 python lr_scaler.py \
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
#  					--warm_up_epoch 15 \
# 					--filter_bn \
# 					--filter_bias	
# done

############################## 200305 ###########################


# group4
# for i in 0.0001 0.0002 0.0004 0.0008 0.0016 0.0032 0.0064 0.0128 0.0256
# do
# 	CUDA_VISIBLE_DEVICES=2 python wd_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 512 \
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
# 	CUDA_VISIBLE_DEVICES=2 python wd_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
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
# 	CUDA_VISIBLE_DEVICES=4,5 python main.0.0.2.py \
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
# 					--filter_bn\
# 					--warm_up_epoch 15 
# done


############################## 200304 ###########################
# # 8gpu
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=4,5 python critical.0.0.1.py \
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

# # 8GPU
# for i in 3.2
# do
# 	CUDA_VISIBLE_DEVICES=4,5 python main.0.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay 0.0002 \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual\
# 					--warm_up_epoch 15 
# done

# IITP
# for i in 0.1 0.2 0.4 0.8 3.2 6.4 12.8 
# do
# 	CUDA_VISIBLE_DEVICES=2 python main.0.0.2.py \
# 					--dataset cifar10 \
# 					--dataroot /home/user/ssd1/dataset/ \
# 					--epochs 300 \
# 					--batch_size 2048 \
# 					--weight_decay 0.0002 \
# 					--wd_off_epoch 300 \
# 					--lr 1.6 \
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
# 	CUDA_VISIBLE_DEVICES=2 python critical.0.0.1.py \
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
# for i in 0.0004
# do
# 	CUDA_VISIBLE_DEVICES=2 python critical.0.0.1.py \
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
# for i in 0.0016
# do
# 	CUDA_VISIBLE_DEVICES=2 python critical.0.0.1.py \
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
# for i in 0.0004
# do
# 	CUDA_VISIBLE_DEVICES=2 python critical.0.0.1.py \
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
# for i in 0.0004
# do
# 	CUDA_VISIBLE_DEVICES=2 python critical.0.0.1.py \
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
# for i in 0.0001 0.0004 0.0016 0.0064 0.0256
# do
# 	CUDA_VISIBLE_DEVICES=2 python critical.0.0.1.py \
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

# IITP
# for i in 15 
# do
# 	CUDA_VISIBLE_DEVICES=2,3 python main.0.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 2048 \
# 					--weight_decay 0.0002 \
# 					--wd_off_epoch 300 \
# 					--lr 1.6 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--seed 0 1 2 3 4 \
# 					--warm_up_epoch $i \
# 					--zero_init_residual
# done

############################## 200226 ###########################
# # IITP
# for i in 15 
# do
# 	CUDA_VISIBLE_DEVICES=2,3 python main.0.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 2048 \
# 					--weight_decay 0.0002 \
# 					--wd_off_epoch 300 \
# 					--lr 3.2 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--seed 0 1 2 3 4 \
# 					--warm_up_epoch $i \
# 					--zero_init_residual
# done

############################## 200224 ###########################
# # IITP
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
# 					--warm_up_epoch 15 \
# 					--zero_init_residual
# done

############################## 200221 ###########################
# # 8GPU
# for i in 15 30 5 
# do
# 	CUDA_VISIBLE_DEVICES=2,3,4,5 python main.0.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 2048 \
# 					--weight_decay 0.0001 \
# 					--wd_off_epoch 300 \
# 					--lr 3.2 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--seed 0 1 2 \
# 					--warm_up_epoch $i
# done

############################## 200127 ###########################
# # IITP
# for i in 0.2 0.4 0.8
# do
# 	CUDA_VISIBLE_DEVICES=2 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 512 \
# 					--weight_decay 0.0001 \
# 					--wd_off_epoch 300 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--warm_up_epoch 5
# done

############################## 200121 ###########################
# 8GPU
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=3 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 256 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.4 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs/
# done
############################## 200119 ###########################
# group4
# for i in 0.0064
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
# 	CUDA_VISIBLE_DEVICES=2 python critical.0.0.1.py \
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

# group4
# for i in 0.0032 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=2 python critical.0.0.1.py \
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
# for i in 0.0128
# do
# 	CUDA_VISIBLE_DEVICES=2 python critical.0.0.1.py \
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

####################.########## 200111 ###########################
# # group4
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=2 python critical.0.0.1.py \
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

# # group4
# for i in 0.00005
# do
# 	CUDA_VISIBLE_DEVICES=2 python critical.0.0.1.py \
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
# for i in 0.00005
# do
# 	CUDA_VISIBLE_DEVICES=2 python critical.0.0.1.py \
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
# for i in 0.0004
# do
# 	CUDA_VISIBLE_DEVICES=2 python critical.0.0.1.py \
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
# for i in 0.0004
# do
# 	CUDA_VISIBLE_DEVICES=2 python critical.0.0.1.py \
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
# for i in 0.0004
# do
# 	CUDA_VISIBLE_DEVICES=2 python critical.0.0.1.py \
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
# 	CUDA_VISIBLE_DEVICES=2 python critical.0.0.1.py \
# 					--dataset cifar10 \
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

############################## 200109 ###########################
# # IITP
# for i in 0.0256
# do
# 	CUDA_VISIBLE_DEVICES=2 python critical.0.0.1.py \
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
# for i in 0.0001 0.0002 0.0004 0.0008 0.0016 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=2,3 python critical.0.0.1.py \
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
# for i in 0.0004
# do
# 	CUDA_VISIBLE_DEVICES=2 python critical.0.0.1.py \
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
# for i in 0.0016
# do
# 	CUDA_VISIBLE_DEVICES=2 python critical.0.0.1.py \
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
# for i in 0.00005
# do
# 	CUDA_VISIBLE_DEVICES=2 python critical.0.0.1.py \
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
# for i in 0.0128 0.0256
# do
# 	CUDA_VISIBLE_DEVICES=2 python critical.0.0.1.py \
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
# 	CUDA_VISIBLE_DEVICES=2 python critical.0.0.1.py \
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

############################## 201225 ###########################
# # 8GPU - J
# for i in 0.001 0.0016 
# do
# 	CUDA_VISIBLE_DEVICES=4,5 python critical.0.0.1.py \
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
# for i in 0.0001 0.0002 0.0003 0.0004 0.0005
# do
# 	CUDA_VISIBLE_DEVICES=2 python critical.0.0.1.py \
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

############################## 201215 ###########################
# 8GPU
# for i in 300
# do
# 	CUDA_VISIBLE_DEVICES=2 python critical.0.0.1.py \
# 					--epochs 300 \
# 					--batch_size 1024 \
# 					--weight_decay 0.0016 \
# 					--wd_off_epoch $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--std_weight 1
# done




############################## 201124 ###########################
# IITP
# for i in 50 150 250
# do
# 	CUDA_VISIBLE_DEVICES=2 python critical.0.0.1.py \
# 					--epochs 300 \
# 					--batch_size 1024 \
# 					--weight_decay 0.0005 \
# 					--wd_off_epoch $i
# done


# 8GPU
# # for i in $(seq 225 25 300)
# for i in 50 250 
# do
# 	CUDA_VISIBLE_DEVICES=2 python critical.0.0.1.py \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.0005 \
# 					--wd_off_epoch $i
# done
