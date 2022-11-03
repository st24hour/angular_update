#!/bin/bash

############################## 220610 ###########################
# ECCV
for i in 0
do
	CUDA_VISIBLE_DEVICES=1 python direction.0.0.6.py \
					--dataset cifar10 \
					--num_workers 2 \
					--num_sample $i \
					--epochs 300 \
					--grad_test_epoch 99 199 299 \
					--decay_epoch 150 225 \
					--batch_size 1024 \
					--lr 0.8 \
					--weight_decay 0.0004 \
					--momentum 0.9 \
					--schedule_type 'step' \
					--net_type resnet18_GBN_invariant2 \
					--save_model \
					--seed 0 \
					--warmup_epochs 0 \
					--save_dir './logs_trash/'
done

############################## 220529 ###########################
# 한상
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 ./segmentation/train.py \
# 		--lr 0.02 \
# 		--schedule_type step \
# 		--decay_epoch 45 55 \
# 		--weight_decay $i \
# 		--num_sample 40000 \
# 		--test_freq 10 \
# 		--dataset coco \
# 		--batch_size 32 \
# 		--model fcn_resnet50 \
# 		--data-path=/home/user/dataset/COCO \
# 		--epochs 60 \
# 		--output-dir ./logs_coco
# done

# 재성
# for i in 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 ./segmentation/train.py \
# 		--lr 0.02 \
# 		--schedule_type step \
# 		--decay_epoch 45 55 \
# 		--weight_decay $i \
# 		--num_sample 40000 \
# 		--test_freq 10 \
# 		--dataset coco \
# 		--batch_size 32 \
# 		--model fcn_resnet50 \
# 		--data-path=/home/user/dataset/COCO \
# 		--epochs 60 \
# 		--output-dir ./logs_coco
# done

# 형권
# for i in 0.0016
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 ./segmentation/train.py \
# 		--lr 0.02 \
# 		--schedule_type step \
# 		--decay_epoch 45 55 \
# 		--weight_decay $i \
# 		--num_sample 40000 \
# 		--test_freq 10 \
# 		--dataset coco \
# 		--batch_size 32 \
# 		--model fcn_resnet50 \
# 		--data-path=/home/user/dataset/COCO \
# 		--epochs 60 \
# 		--output-dir ./logs_coco
# done

# 주승
# for i in 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 ./segmentation/train.py \
# 		--lr 0.02 \
# 		--schedule_type step \
# 		--decay_epoch 45 55 \
# 		--weight_decay $i \
# 		--num_sample 10000 \
# 		--test_freq 10 \
# 		--dataset coco \
# 		--batch_size 32 \
# 		--model fcn_resnet50 \
# 		--data-path=/home/user/dataset/COCO \
# 		--epochs 60 \
# 		--output-dir ./logs_coco
# done

# 한상
# for i in 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 ./segmentation/train.py \
# 		--lr 0.02 \
# 		--schedule_type step \
# 		--decay_epoch 45 55 \
# 		--weight_decay $i \
# 		--num_sample 10000 \
# 		--test_freq 10 \
# 		--dataset coco \
# 		--batch_size 32 \
# 		--model fcn_resnet50 \
# 		--data-path=/home/user/dataset/COCO \
# 		--epochs 60 \
# 		--output-dir ./logs_coco
# done

# 형권
# for i in 0.0016
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 ./segmentation/train.py \
# 		--lr 0.02 \
# 		--schedule_type step \
# 		--decay_epoch 45 55 \
# 		--weight_decay $i \
# 		--num_sample 10000 \
# 		--test_freq 10 \
# 		--dataset coco \
# 		--batch_size 32 \
# 		--model fcn_resnet50 \
# 		--data-path=/home/user/dataset/COCO \
# 		--epochs 60 \
# 		--output-dir ./logs_coco
# done

# 형권4
# for i in 0.0016 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 ./segmentation/train.py \
# 		--lr 0.02 \
# 		--schedule_type step \
# 		--decay_epoch 45 55 \
# 		--weight_decay $i \
# 		--num_sample 20000 \
# 		--test_freq 10 \
# 		--dataset coco \
# 		--batch_size 32 \
# 		--model fcn_resnet50 \
# 		--data-path=/home/user/dataset/COCO \
# 		--epochs 60 \
# 		--output-dir ./logs_coco
# done

# 한상
# for i in 0.0002 0.0004
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 ./segmentation/train.py \
# 		--lr 0.02 \
# 		--schedule_type step \
# 		--decay_epoch 45 55 \
# 		--weight_decay $i \
# 		--num_sample 20000 \
# 		--test_freq 10 \
# 		--dataset coco \
# 		--batch_size 32 \
# 		--model fcn_resnet50 \
# 		--data-path=/home/user/dataset/COCO \
# 		--epochs 60 \
# 		--output-dir ./logs_coco
# done

# 한상
# for i in 0.0128 
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 ./segmentation/train.py \
# 		--lr 0.02 \
# 		--schedule_type step \
# 		--decay_epoch 45 55 \
# 		--weight_decay $i \
# 		--num_sample 5000 \
# 		--test_freq 10 \
# 		--dataset coco \
# 		--batch_size 32 \
# 		--model fcn_resnet50 \
# 		--data-path=/home/user/dataset/COCO \
# 		--epochs 60 \
# 		--output-dir ./logs_coco
# done

# 형권4
# for i in 0.0032 
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 ./segmentation/train.py \
# 		--lr 0.02 \
# 		--schedule_type step \
# 		--decay_epoch 45 55 \
# 		--weight_decay $i \
# 		--num_sample 5000 \
# 		--test_freq 10 \
# 		--dataset coco \
# 		--batch_size 32 \
# 		--model fcn_resnet50 \
# 		--data-path=/home/user/dataset/COCO \
# 		--epochs 60 \
# 		--output-dir ./logs_coco
# done

# # 형권4
# for i in 0.0004 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 ./segmentation/train.py \
# 		--lr 0.02 \
# 		--schedule_type step \
# 		--decay_epoch 45 55 \
# 		--weight_decay $i \
# 		--num_sample 5000 \
# 		--test_freq 10 \
# 		--dataset coco \
# 		--batch_size 32 \
# 		--model fcn_resnet50 \
# 		--data-path=/home/user/dataset/COCO \
# 		--epochs 60 \
# 		--output-dir ./logs_coco
# done

# 재성
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 ./segmentation/train.py \
# 		--lr 0.02 \
# 		--schedule_type step \
# 		--decay_epoch 45 55 \
# 		--weight_decay $i \
# 		--num_sample 80000 \
# 		--test_freq 10 \
# 		--dataset coco \
# 		--batch_size 32 \
# 		--model fcn_resnet50 \
# 		--data-path=/home/user/dataset/COCO \
# 		--epochs 60 \
# 		--output-dir ./logs_coco
# done

############################## 220528 ###########################
# 주승
# for i in 0.0016
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 ./segmentation/train.py \
# 		--lr 0.02 \
# 		--schedule_type step \
# 		--decay_epoch 45 55 \
# 		--weight_decay $i \
# 		--num_sample 80000 \
# 		--test_freq 10 \
# 		--dataset coco \
# 		--batch_size 32 \
# 		--model fcn_resnet50 \
# 		--data-path=/home/user/dataset/COCO \
# 		--epochs 60 \
# 		--output-dir ./logs_coco
# done

# 형권
# for i in 0.0004
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 ./segmentation/train.py \
# 		--lr 0.02 \
# 		--schedule_type step \
# 		--decay_epoch 45 55 \
# 		--weight_decay $i \
# 		--num_sample 80000 \
# 		--test_freq 10 \
# 		--dataset coco \
# 		--batch_size 32 \
# 		--model fcn_resnet50 \
# 		--data-path=/home/user/dataset/COCO \
# 		--epochs 60 \
# 		--output-dir ./logs_coco
# done

# 한상
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 ./segmentation/train.py \
# 		--lr 0.02 \
# 		--schedule_type step \
# 		--decay_epoch 45 55 \
# 		--weight_decay $i \
# 		--num_sample 40000 \
# 		--test_freq 10 \
# 		--dataset coco \
# 		--batch_size 32 \
# 		--model fcn_resnet50 \
# 		--aux-loss \
# 		--data-path=/home/user/dataset/COCO \
# 		--epochs 60 \
# 		--output-dir ./logs_coco
# done

# 형권2
# for i in 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 ./segmentation/train.py \
# 		--lr 0.02 \
# 		--schedule_type step \
# 		--decay_epoch 45 55 \
# 		--weight_decay $i \
# 		--num_sample 40000 \
# 		--test_freq 10 \
# 		--dataset coco \
# 		--batch_size 32 \
# 		--model fcn_resnet50 \
# 		--aux-loss \
# 		--data-path=/home/user/dataset/COCO \
# 		--epochs 60 \
# 		--output-dir ./logs_coco
# done

# 형권2
# for i in 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 ./segmentation/train.py \
# 		--lr 0.02 \
# 		--schedule_type step \
# 		--decay_epoch 45 55 \
# 		--weight_decay $i \
# 		--num_sample 40000 \
# 		--test_freq 10 \
# 		--dataset coco \
# 		--batch_size 32 \
# 		--model fcn_resnet50 \
# 		--aux-loss \
# 		--data-path=/home/user/dataset/COCO \
# 		--epochs 60 \
# 		--output-dir ./logs_coco
# done

# 형권
# for i in 0.0016 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 ./segmentation/train.py \
# 		--lr 0.02 \
# 		--schedule_type step \
# 		--decay_epoch 45 55 \
# 		--weight_decay $i \
# 		--num_sample 20000 \
# 		--test_freq 10 \
# 		--dataset coco \
# 		--batch_size 32 \
# 		--model fcn_resnet50 \
# 		--aux-loss \
# 		--data-path=/home/user/dataset/COCO \
# 		--epochs 60 \
# 		--output-dir ./logs_coco
# done

# 형권2
# for i in 0.0004
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 ./segmentation/train.py \
# 		--lr 0.02 \
# 		--schedule_type step \
# 		--decay_epoch 45 55 \
# 		--weight_decay $i \
# 		--num_sample 20000 \
# 		--test_freq 10 \
# 		--dataset coco \
# 		--batch_size 32 \
# 		--model fcn_resnet50 \
# 		--aux-loss \
# 		--data-path=/home/user/dataset/COCO \
# 		--epochs 60 \
# 		--output-dir ./logs_coco
# done

# 재성
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 ./segmentation/train.py \
# 		--lr 0.02 \
# 		--schedule_type step \
# 		--decay_epoch 45 55 \
# 		--weight_decay $i \
# 		--num_sample 20000 \
# 		--test_freq 10 \
# 		--dataset coco \
# 		--batch_size 32 \
# 		--model fcn_resnet50 \
# 		--aux-loss \
# 		--data-path=/home/user/dataset/COCO \
# 		--epochs 60 \
# 		--output-dir ./logs_coco
# done

# 형권2 
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 ./segmentation/train.py \
# 		--lr 0.02 \
# 		--schedule_type step \
# 		--decay_epoch 45 55 \
# 		--weight_decay $i \
# 		--num_sample 10000 \
# 		--dataset coco \
# 		--batch_size 32 \
# 		--model fcn_resnet50 \
# 		--aux-loss \
# 		--data-path=/home/user/dataset/COCO \
# 		--epochs 60 \
# 		--output-dir ./logs_coco
# done

# 형권
# for i in 0.0008 0.0016
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 ./segmentation/train.py \
# 		--lr 0.02 \
# 		--schedule_type step \
# 		--decay_epoch 45 55 \
# 		--weight_decay $i \
# 		--num_sample 10000 \
# 		--dataset coco \
# 		--batch_size 32 \
# 		--model fcn_resnet50 \
# 		--aux-loss \
# 		--data-path=/home/user/dataset/COCO \
# 		--epochs 60 \
# 		--output-dir ./logs_coco
# done

# 재성
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 ./segmentation/train.py \
# 		--lr 0.02 \
# 		--schedule_type step \
# 		--decay_epoch 45 55 \
# 		--weight_decay $i \
# 		--num_sample 10000 \
# 		--dataset coco \
# 		--batch_size 32 \
# 		--model fcn_resnet50 \
# 		--aux-loss \
# 		--data-path=/home/user/dataset/COCO \
# 		--epochs 60 \
# 		--output-dir ./logs_coco
# done

# 주승
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 ./segmentation/train.py \
# 		--lr 0.02 \
# 		--schedule_type step \
# 		--decay_epoch 45 55 \
# 		--weight_decay $i \
# 		--num_sample 80000 \
# 		--dataset coco \
# 		--batch_size 32 \
# 		--model fcn_resnet50 \
# 		--aux-loss \
# 		--data-path=/home/user/dataset/COCO \
# 		--epochs 60 \
# 		--output-dir ./logs_coco
# done

# 한상
# for i in 0.0004
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 ./segmentation/train.py \
# 		--lr 0.02 \
# 		--schedule_type step \
# 		--decay_epoch 45 55 \
# 		--weight_decay $i \
# 		--num_sample 80000 \
# 		--dataset coco \
# 		--batch_size 32 \
# 		--model fcn_resnet50 \
# 		--aux-loss \
# 		--data-path=/home/user/dataset/COCO \
# 		--epochs 60 \
# 		--output-dir ./logs_coco
# done

# 형권2
# for i in 0.0128
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 ./segmentation/train.py \
# 		--lr 0.02 \
# 		--schedule_type exp \
# 		--decay_epoch 45 55 \
# 		--weight_decay $i \
# 		--num_sample 5000 \
# 		--test_freq 10 \
# 		--dataset coco \
# 		--batch_size 32 \
# 		--model fcn_resnet50 \
# 		--aux-loss \
# 		--data-path=/home/user/dataset/COCO \
# 		--epochs 60 \
# 		--output-dir ./logs_coco
# done

# 형권
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 ./segmentation/train.py \
# 		--lr 0.02 \
# 		--schedule_type exp \
# 		--decay_epoch 45 55 \
# 		--weight_decay $i \
# 		--num_sample 5000 \
# 		--test_freq 10 \
# 		--dataset coco \
# 		--batch_size 32 \
# 		--model fcn_resnet50 \
# 		--aux-loss \
# 		--data-path=/home/user/dataset/COCO \
# 		--epochs 60 \
# 		--output-dir ./logs_coco
# done

# 재성
# for i in 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 ./segmentation/train.py \
# 		--lr 0.02 \
# 		--schedule_type exp \
# 		--decay_epoch 45 55 \
# 		--weight_decay $i \
# 		--num_sample 5000 \
# 		--test_freq 10 \
# 		--dataset coco \
# 		--batch_size 32 \
# 		--model fcn_resnet50 \
# 		--aux-loss \
# 		--data-path=/home/user/dataset/COCO \
# 		--epochs 60 \
# 		--output-dir ./logs_coco
# done

# ECCV
# for i in 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 ./segmentation/train.py \
# 		--lr 0.02 \
# 		--schedule_type exp \
# 		--decay_epoch 45 55 \
# 		--weight_decay $i \
# 		--num_sample 5000 \
# 		--test_freq 10 \
# 		--dataset coco \
# 		--batch_size 32 \
# 		--model fcn_resnet50 \
# 		--aux-loss \
# 		--data-path=/home/user/dataset/COCO \
# 		--epochs 60 \
# 		--output-dir ./logs_coco
# done

# 재성
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 ./segmentation/train.py \
# 		--lr 0.02 \
# 		--dataset coco \
# 		--batch_size 32 \
# 		--model fcn_resnet50 \
# 		--aux-loss \
# 		--data-path=/home/user/dataset/COCO \
# 		--epochs 50 \
# 		--weight_decay $i \
# 		--num_sample 80000
# done

# # ECCV
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 ./segmentation/train.py \
# 		--lr 0.02 \
# 		--dataset coco \
# 		--batch_size 32 \
# 		--model fcn_resnet50 \
# 		--aux-loss \
# 		--data-path=/home/user/dataset/COCO \
# 		--epochs 30 \
# 		--weight_decay $i \
# 		--num_sample 80000
# done
# # ECCV
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 ./segmentation/train.py \
# 		--lr 0.02 \
# 		--dataset coco \
# 		--batch_size 32 \
# 		--model fcn_resnet50 \
# 		--aux-loss \
# 		--data-path=/home/user/dataset/COCO \
# 		--epochs 20 \
# 		--weight_decay $i \
# 		--num_sample 80000
# done

############################## 220525 ###########################
# jaeyoung
# for i in 25000 12500
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.5.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--decay_epoch 150 225 \
# 					--batch_size 128 \
# 					--lr 0.1 \
# 					--weight_decay 0.0002 \
# 					--momentum 0.9 \
# 					--schedule_type 'step' \
# 					--net_type resnet18_LN_invariant \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warmup_epochs 0 \
# 					--save_dir './logs_inv/'
# done

# group4
# for i in 0.0002 0.0016
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.5.py \
# 					--amp \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample 6250 \
# 					--epochs 300 \
# 					--decay_epoch 150 225 \
# 					--batch_size 128 \
# 					--lr 0.1 \
# 					--weight_decay $i \
# 					--momentum 0.9 \
# 					--schedule_type 'step' \
# 					--net_type resnet18_LN_invariant \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warmup_epochs 0 \
# 					--save_dir './logs_inv/'
# done

# ECCV
# for i in 0
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.5.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--decay_epoch 150 225 \
# 					--batch_size 128 \
# 					--lr 0.1 \
# 					--weight_decay 0.0002 \
# 					--momentum 0.9 \
# 					--schedule_type 'step' \
# 					--net_type resnet18_LN_invariant \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warmup_epochs 0 \
# 					--save_dir './logs_inv/'
# done

# group4
# for i in 3125
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.5.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--decay_epoch 150 225 \
# 					--batch_size 128 \
# 					--lr 0.1 \
# 					--weight_decay 0.0002 \
# 					--momentum 0.9 \
# 					--schedule_type 'step' \
# 					--net_type resnet18_LN_invariant \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warmup_epochs 0 \
# 					--save_dir './logs_inv/'
# done

############################# 220414 ###########################
# for i in -9
# do	
# 	wd=$(echo 2^$i | bc -l)
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.5.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample 0 \
# 					--epochs 300 \
# 					--decay_epoch 150 225 \
# 					--batch_size 128 \
# 					--optim SGD \
# 					--lr 0.1 \
# 					--weight_decay $wd \
# 					--momentum 0.9 \
# 					--schedule_type 'step' \
# 					--net_type resnet18_GBN_invariant2 \
# 					--width 1 \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warmup_epochs 0 \
# 					--save_dir './logs_width_tuning/'
# done

# ############################# 220411 ###########################
# for i in 0.0003 0.0004
# do
# 	CUDA_VISIBLE_DEVICES=2,3 python direction.0.0.5.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample 0 \
# 					--epochs 300 \
# 					--decay_epoch 150 225 \
# 					--batch_size 128 \
# 					--optim SGD \
# 					--lr 0.1 \
# 					--weight_decay $i \
# 					--momentum 0.9 \
# 					--schedule_type 'step' \
# 					--net_type resnet18_GBN_invariant2 \
# 					--width 4 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--warmup_epochs 0 \
# 					--save_dir './logs_width_tuning/'
# done

############################# 220407 ###########################
# for i in 0.0006 0.0005
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.5.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample 12500 \
# 					--epochs 300 \
# 					--decay_epoch 150 225 \
# 					--batch_size 128 \
# 					--optim SGD \
# 					--lr 0.1 \
# 					--weight_decay $i \
# 					--momentum 0.9 \
# 					--schedule_type 'step' \
# 					--net_type resnet18_GBN_invariant2 \
# 					--width 0.25 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--warmup_epochs 0 \
# 					--save_dir './logs_width_tuning/'
# done

############################# 220406 ###########################
# for i in 0.0006 0.0005
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.5.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample 12500 \
# 					--epochs 300 \
# 					--decay_epoch 150 225 \
# 					--batch_size 128 \
# 					--optim SGD \
# 					--lr 0.1 \
# 					--weight_decay $i \
# 					--momentum 0.9 \
# 					--schedule_type 'step' \
# 					--net_type resnet18_GBN_invariant2 \
# 					--width 0.5 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--warmup_epochs 0 \
# 					--save_dir './logs_width_tuning/'
# done

# for i in 0.0012 0.0010 0.0036
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.5.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample 6250 \
# 					--epochs 300 \
# 					--decay_epoch 150 225 \
# 					--batch_size 128 \
# 					--optim SGD \
# 					--lr 0.1 \
# 					--weight_decay $i \
# 					--momentum 0.9 \
# 					--schedule_type 'step' \
# 					--net_type resnet18_GBN_invariant2 \
# 					--width 0.5 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--warmup_epochs 0 \
# 					--save_dir './logs_width_tuning/'
# done

############################# 220405 ###########################
# for i in 0.0006 0.0005
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.5.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample 12500 \
# 					--epochs 300 \
# 					--decay_epoch 150 225 \
# 					--batch_size 128 \
# 					--optim SGD \
# 					--lr 0.1 \
# 					--weight_decay $i \
# 					--momentum 0.9 \
# 					--schedule_type 'step' \
# 					--net_type resnet18_GBN_invariant2 \
# 					--width 0.5 \
# 					--save_model \
# 					--seed 0 \
# 					--warmup_epochs 0 \
# 					--save_dir './logs_width_tuning/'
# done

# for i in 0.0012 0.0010
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.5.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample 6250 \
# 					--epochs 300 \
# 					--decay_epoch 150 225 \
# 					--batch_size 128 \
# 					--optim SGD \
# 					--lr 0.1 \
# 					--weight_decay $i \
# 					--momentum 0.9 \
# 					--schedule_type 'step' \
# 					--net_type resnet18_GBN_invariant2 \
# 					--width 0.5 \
# 					--save_model \
# 					--seed 0 \
# 					--warmup_epochs 0 \
# 					--save_dir './logs_width_tuning/'
# done

############################# 220404 ###########################
# for i in 0 25000 12500 6250 3125
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.3.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--decay_epoch 150 225 \
# 					--batch_size 128 \
# 					--optim AdamW \
# 					--lr 0.0016 \
# 					--weight_decay 0.1 \
# 					--momentum 0.9 \
# 					--schedule_type 'step' \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warmup_epochs 0 \
# 					--save_dir './logs_AdamW/'
# done

# for i in 0 25000 12500 6250 3125
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.3.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--decay_epoch 150 225 \
# 					--batch_size 128 \
# 					--optim AdamW \
# 					--lr 0.001 \
# 					--weight_decay 0.16 \
# 					--momentum 0.9 \
# 					--schedule_type 'step' \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warmup_epochs 0 \
# 					--save_dir './logs_AdamW/'
# done

# for i in 3125
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.3.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--decay_epoch 150 225 \
# 					--batch_size 128 \
# 					--optim AdamW \
# 					--lr 0.021 \
# 					--weight_decay 0.1 \
# 					--momentum 0.9 \
# 					--schedule_type 'step' \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--warmup_epochs 0 \
# 					--save_dir './logs_AdamW_tuning/'
# done

# for i in 6250
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.3.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--decay_epoch 150 225 \
# 					--batch_size 128 \
# 					--optim AdamW \
# 					--lr 0.011 \
# 					--weight_decay 0.1 \
# 					--momentum 0.9 \
# 					--schedule_type 'step' \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--warmup_epochs 0 \
# 					--save_dir './logs_AdamW_tuning/'
# done

############################# 220401 ###########################
# for i in 3125
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.3.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--decay_epoch 150 225 \
# 					--batch_size 128 \
# 					--optim AdamW \
# 					--lr 0.016 \
# 					--weight_decay 0.1 \
# 					--momentum 0.9 \
# 					--schedule_type 'step' \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warmup_epochs 0 \
# 					--save_dir './logs_AdamW_tuning/'
# done

# for i in 6250
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.3.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--decay_epoch 150 225 \
# 					--batch_size 128 \
# 					--optim AdamW \
# 					--lr 0.008 \
# 					--weight_decay 0.1 \
# 					--momentum 0.9 \
# 					--schedule_type 'step' \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warmup_epochs 0 \
# 					--save_dir './logs_AdamW_tuning/'
# done

# for i in 3125
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.3.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--decay_epoch 150 225 \
# 					--batch_size 128 \
# 					--optim AdamW \
# 					--lr 0.007 \
# 					--weight_decay 0.1 \
# 					--momentum 0.9 \
# 					--schedule_type 'step' \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warmup_epochs 0 \
# 					--save_dir './logs_AdamW_tuning/'
# done

############################# 220330 ###########################
# for i in 12500 6250 3125
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.3.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--decay_epoch 150 225 \
# 					--batch_size 128 \
# 					--optim AdamW \
# 					--lr 0.001414 \
# 					--weight_decay 0.1 \
# 					--momentum 0.9 \
# 					--schedule_type 'step' \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warmup_epochs 0 \
# 					--save_dir './logs_AdamW/'
# done

############################# 220329 ###########################
# for i in 0 25000 12500 6250 3125
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.3.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--decay_epoch 150 225 \
# 					--batch_size 128 \
# 					--optim AdamW \
# 					--lr 0.001 \
# 					--weight_decay 0.2 \
# 					--momentum 0.9 \
# 					--schedule_type 'step' \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warmup_epochs 0 \
# 					--save_dir './logs_AdamW/'
# done

############################# 220316 ###########################
# for i in 0 25000 12500 6250 3125
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.4.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 600 \
# 					--decay_epoch 300 450 \
# 					--batch_size 128 \
# 					--lr 0.1 \
# 					--weight_decay 0.0001 \
# 					--momentum 0.9 \
# 					--schedule_type 'step' \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warmup_epochs 0 \
# 					--save_dir './logs_epoch/'
# done

# for i in 0 25000 12500 6250 3125
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.4.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 600 \
# 					--decay_epoch 300 450 \
# 					--batch_size 128 \
# 					--lr 0.4 \
# 					--weight_decay 0.0004 \
# 					--momentum 0.9 \
# 					--schedule_type 'step' \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warmup_epochs 0 \
# 					--save_dir './logs_epoch/'
# done

# for i in 0 25000 6250 3125
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.4.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 600 \
# 					--decay_epoch 300 450 \
# 					--batch_size 128 \
# 					--lr 0.5657 \
# 					--weight_decay 0.0005657 \
# 					--momentum 0.9 \
# 					--schedule_type 'step' \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warmup_epochs 0 \
# 					--save_dir './logs_epoch/'
# done

# for i in 25000
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.4.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 150 \
# 					--angle_freq 195 \
# 					--decay_epoch 75 112.5 \
# 					--batch_size 128 \
# 					--lr 0.5657 \
# 					--weight_decay 0.0005657 \
# 					--momentum 0.9 \
# 					--schedule_type 'step' \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 \
# 					--warmup_epochs 0 \
# 					--save_dir './logs_epoch/'
# done

############################## 220306 ###########################
# GCP
# for i in 2048 1024 512 256
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.3.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample 0 \
# 					--epochs 300 \
# 					--decay_epoch 150 225 \
# 					--batch_size $i \
# 					--lr 2 \
# 					--weight_decay 0.0002 \
# 					--momentum 0 \
# 					--schedule_type 'step' \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warmup_epochs 0 \
# 					--save_dir './logs_inv_wo_moment/'
# done

# ############################## 220302 ###########################
# # # GCP
# # for i in 0 25000 12500 6250 3125
# # do
# # 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.3.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--decay_epoch 150 225 \
# 					--batch_size 128 \
# 					--lr 0.2 \
# 					--weight_decay 0.0002 \
# 					--momentum 0.9 \
# 					--schedule_type 'cosine' \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warmup_epochs 0 \
# 					--save_dir './logs_inv_cosine/'
# done

############################## 220226 ###########################
# 8gpu-ICML2
# for i in 256 512 1024 
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample 12500 \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size $i \
# 					--lr 0.1 \
# 					--weight_decay 0.0016 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_inv/'
# done

# 8gpu-ICML3
# for i in 0 
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size 64 \
# 					--lr 0.1 \
# 					--weight_decay 0.0016 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_inv/'
# done

############################## 220224 ###########################
# ICML2
# for i in 0.0002 0.000025
# do
# 	CUDA_VISIBLE_DEVICES=2,3 python direction.0.0.2.py \
# 					--dataset imagenet \
# 					--num_workers 10 \
# 					--num_sample 40036 \
# 					--epochs 120 \
# 					--epoch_step 60 90 110 \
# 					--batch_size 256 \
# 					--lr 0.1 \
# 					--weight_decay $i \
# 					--momentum 0.9 \
# 					--net_type efficientnet_b0 \
# 					--save_model \
# 					--seed 0 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_inv/'
# done

############################## 220223 ###########################
# ICML2
# for i in 0.000025 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=2,3 python direction.0.0.2.py \
# 					--dataset imagenet \
# 					--num_workers 10 \
# 					--num_sample 80072 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 256 \
# 					--lr 0.1 \
# 					--weight_decay $i \
# 					--momentum 0.9 \
# 					--net_type efficientnet_b0_inv \
# 					--save_model \
# 					--seed 0 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_inv/'
# done

############################## 220222 ###########################
# GCP
# for i in 0
# do
# 	CUDA_VISIBLE_DEVICES=2,3 python direction.0.0.2.py \
# 					--GCP \
# 					--dataset imagenet \
# 					--num_workers 12 \
# 					--num_sample $i \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 256 \
# 					--lr 0.1 \
# 					--weight_decay 0.0002 \
# 					--momentum 0.9 \
# 					--net_type efficientnet_b0_inv \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_inv/'
# done

# GCP
# for i in 0
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.2.py \
# 					--GCP \
# 					--dataset imagenet \
# 					--num_workers 16 \
# 					--num_sample $i \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 256 \
# 					--lr 0.1 \
# 					--weight_decay 0.0002 \
# 					--momentum 0.9 \
# 					--net_type efficientnet_b0_inv \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_inv/'
# done

############################## 220221 ###########################
# GCP
# for i in 0
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.2.py \
# 					--GCP \
# 					--dataset imagenet \
# 					--num_workers 32 \
# 					--num_sample $i \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 512 \
# 					--lr 0.1 \
# 					--weight_decay 0.0001 \
# 					--momentum 0.9 \
# 					--net_type efficientnet_b0_inv \
# 					--save_model \
# 					--seed 0 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_trash/'
# done

############################## 220218 ###########################
# GCP
# for i in 0
# do
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.2.py \
# 					--GCP \
# 					--eps 1e-05 \
# 					--alpha_sqaure 1000 \
# 					--amp \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size 128 \
# 					--lr 0.1 \
# 					--weight_decay 0.0001 \
# 					--momentum 0.9 \
# 					--net_type efficientnet_b0_inv \
# 					--save_model \
# 					--seed 0 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_trash/'
# done

############################## 220127 ###########################
# group4
# for i in 12500
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.2.py \
# 					--dataset tinyimagenet \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 120 \
# 					--epoch_step 60 90 \
# 					--batch_size 256 \
# 					--lr 0.8 \
# 					--weight_decay 0.0008 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_inv_stl \
# 					--save_model \
# 					--seed 0 1 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_inv/'
# done

# 8gpu-ICML1
# for i in 12500 25000 50000 0
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.2.py \
# 					--dataset tinyimagenet \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 120 \
# 					--epoch_step 60 90 \
# 					--batch_size 256 \
# 					--lr 0.1 \
# 					--weight_decay 0.0032 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_inv_stl \
# 					--save_model \
# 					--seed 0 1 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_inv/'
# done

############################## 220126 ###########################
# 8gpu-ICML1
# for i in 25000 50000 0
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.2.py \
# 					--dataset tinyimagenet \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 120 \
# 					--epoch_step 60 90 \
# 					--batch_size 256 \
# 					--lr 6.4 \
# 					--weight_decay 0.0001 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_inv_stl \
# 					--save_model \
# 					--seed 0 1 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_inv/'
# done

# group4
# for i in 6250
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.2.py \
# 					--dataset tinyimagenet \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 120 \
# 					--epoch_step 60 90 \
# 					--batch_size 256 \
# 					--lr 6.4 \
# 					--weight_decay 0.0001 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_inv_stl \
# 					--save_model \
# 					--seed 0 1 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_inv/'
# done

# for i in 6250
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.2.py \
# 					--dataset tinyimagenet \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 120 \
# 					--epoch_step 60 90 \
# 					--batch_size 256 \
# 					--lr 0.4 \
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
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.2.py \
# 					--dataset cifar100 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size 128 \
# 					--lr 0.1 \
# 					--weight_decay 0.0004 \
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
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.2.py \
# 					--dataset tinyimagenet \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 120 \
# 					--epoch_step 60 90 \
# 					--batch_size 256 \
# 					--lr 0.1 \
# 					--weight_decay 0.0002 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_inv_stl \
# 					--save_model \
# 					--seed 0 1 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_inv/'
# done

############################## 220125 ###########################
# # 8gpu-ICML3
# for i in 3125 6250 12500 25000 0
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.2.py \
# 					--dataset cifar100 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size 128 \
# 					--lr 0.2 \
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
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.2.test.py \
# 					--dataset tinyimagenet \
# 					--test_freq 1 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size 256 \
# 					--lr 0.1 \
# 					--weight_decay 0.0004 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_inv_stl \
# 					--save_model \
# 					--seed 0 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_trash/'
# done

# 8gpu-ICML3
# for i in 6250 12500 25000 0
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size 128 \
# 					--lr 0.1 \
# 					--weight_decay 0.0004 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_inv/'
# done

# 8gpu-ICML2
# for i in 3125 6250 12500 25000 50000
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size 128 \
# 					--lr 0.2 \
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
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size 128 \
# 					--lr 0.1 \
# 					--weight_decay 0.0002 \
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
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size 128 \
# 					--lr 0.1414 \
# 					--weight_decay 0.0001414 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_var/'
# done
############################## 220124 ###########################
# 8gpu-group4
# for i in 3125 6250 12500 25000 50000
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size 128 \
# 					--lr 1 \
# 					--weight_decay 0.0004 \
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
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size 128 \
# 					--lr 1 \
# 					--weight_decay 0.0001 \
# 					--momentum 0 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_inv_wo_moment/'
# done

# 8gpu-ICML3
# for i in 1024 512 256 128
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.2.py \
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

# 8gpu-ICML2
# for i in 2048
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample 0 \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size $i \
# 					--lr 0.2 \
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
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample 0 \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size $i \
# 					--lr 0.1 \
# 					--weight_decay 0.0002 \
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
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample 0 \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size $i \
# 					--lr 0.1414 \
# 					--weight_decay 0.0001414 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_inv/'
# done

############################## 220123 ###########################
# 8gpu-ICML3
# for i in 1024 512 256 128
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample 0 \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size $i \
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
# for i in 0.1 0.2 0.4 0.8 1.6 3.2
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample 12500 \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size 128 \
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
# for i in 0.1 0.2 0.4 0.8 1.6 3.2
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample 0 \
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
# for i in 0.25
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample 12500 \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size 256 \
# 					--weight_decay 0.0008 \
# 					--lr 0.2 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_observation/'
# done

# 8gpu-ICML
# for i in 0.25
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample 12500 \
# 					--epochs 3000 \
# 					--epoch_step 300 2000 \
# 					--batch_size 128 \
# 					--weight_decay 0.0004 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_observation/'
# done

############################## 220122 ###########################
# 8gpu-ICML
# for i in 0.25
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample 25000 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.0002 \
# 					--lr 0.2 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_observation/'
# done

# group4
# for i in 0.25
# do
# 	CUDA_VISIBLE_DEVICES=1 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample 0 \
# 					--epochs 300 \
# 					--batch_size 1024 \
# 					--weight_decay 0.0002 \
# 					--lr 0.8 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_observation/'
# done

############################## 200527 ###########################
# # 8gpu
# for i in 1
# do
# 	CUDA_VISIBLE_DEVICES=2,3 python invariant.1.0.2.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay 0.0002 \
# 					--lr 6.4 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--save_dir './logs_invariant_scaler/' \
#  					--alpha_sqaure $i
# done

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
