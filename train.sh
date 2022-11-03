#!/bin/bash

############################## 220621 ###########################
# ECCV
for i in 0.0016 
do
	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 seg_train.0.0.1.py \
		--lr 0.02 \
		--schedule_type step \
		--decay_epoch 45 55 \
		--weight_decay $i \
		--num_sample 40000 \
		--test_freq 10 \
		--dataset coco \
		--batch_size 32 \
		--model fcn_resnet50_invariant \
		--data-path=/home/user/dataset/COCO \
		--epochs 60 \
		--output-dir ./segmentation/logs_coco_invariant_test
done

############################## 220624 ###########################
# ECCV
# for i in 4096
# do
# 	CUDA_VISIBLE_DEVICES=0,1 python direction.0.0.6.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--grad_test_epoch 99 199 299 \
# 					--decay_epoch 150 225 \
# 					--batch_size 4096 \
# 					--lr 0.1 \
# 					--weight_decay 0.0004 \
# 					--momentum 0.9 \
# 					--schedule_type 'step' \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 \
# 					--warmup_epochs 0 \
# 					--save_dir './logs_grad_constant/'
# done

############################## 220621 ###########################
# ECCV
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./segmentation/train.py \
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
# 		--epochs 1 \
# 		--output-dir ./segmentation/logs_coco_save_dir_test
# done

############################## 220610 ###########################
# ECCV
# for i in 0
# do
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.6.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--grad_test_epoch 99 199 299 \
# 					--decay_epoch 150 225 \
# 					--batch_size 128 \
# 					--lr 0.1 \
# 					--weight_decay 0.0004 \
# 					--momentum 0.9 \
# 					--schedule_type 'step' \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 \
# 					--warmup_epochs 0 \
# 					--save_dir './logs_trash/'
# done

############################## 220528 ###########################
# 재성4
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./segmentation/train.py \
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

# # 재성4
# for i in 0.0004
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./segmentation/train.py \
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
# for i in 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./segmentation/train.py \
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
# for i in 0.0004
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./segmentation/train.py \
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
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./segmentation/train.py \
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

# 재성
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./segmentation/train.py \
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
# for i in 0.0008 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./segmentation/train.py \
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
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./segmentation/train.py \
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
# for i in 0.0016 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./segmentation/train.py \
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
# for i in 0.0001 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./segmentation/train.py \
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

############################## 220528 ###########################
# 재성4
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./segmentation/train.py \
# 		--lr 0.02 \
# 		--schedule_type step \
# 		--decay_epoch 45 55 \
# 		--weight_decay $i \
# 		--num_sample 40000 \
# 		--test_freq 10 \
# 		--dataset coco \
# 		--batch_size 32 \
# 		--model fcn_resnet50 \
# 		--aux_loss \
# 		--data-path=/home/user/dataset/COCO \
# 		--epochs 60 \
# 		--output-dir ./logs_coco
# done

# 재성
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./segmentation/train.py \
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

# 주승
# for i in 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./segmentation/train.py \
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
# for i in 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./segmentation/train.py \
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
# for i in 0.0016
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./segmentation/train.py \
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
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./segmentation/train.py \
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

# 재성4
# for i in 0.0004 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./segmentation/train.py \
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
# for i in 0.0008 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./segmentation/train.py \
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
# for i in 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./segmentation/train.py \
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
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./segmentation/train.py \
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
# for i in 0.0002 0.0004
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./segmentation/train.py \
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
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./segmentation/train.py \
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

# 주승
# for i in 0.0016
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./segmentation/train.py \
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

# 재영
# for i in 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=4,1,2,3 torchrun --nproc_per_node=4 ./segmentation/train.py \
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
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./segmentation/train.py \
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

# 재성4
# for i in 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./segmentation/train.py \
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
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./segmentation/train.py \
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
# for i in 0.0016
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./segmentation/train.py \
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
# for i in 0.0004
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./segmentation/train.py \
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

# # # ECCV
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./segmentation/train.py \
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

# Jaeyoung
# for i in 0.0001 0.0004 0.0016
# do
# 	CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 ./segmentation/train.py \
# 		--lr 0.02 \
# 		--dataset coco \
# 		--batch_size 32 \
# 		--model fcn_resnet50 \
# 		--aux-loss \
# 		--data-path=/home/user/dataset/COCO \
# 		--epochs 30 \
# 		--weight_decay 0.0001 \
# 		--num_sample 5000
# done

# 재성
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./segmentation/train.py \
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

# 형권2
# for i in 0.0016 0.0032 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./segmentation/train.py \
# 		--lr 0.02 \
# 		--dataset coco \
# 		--batch_size 32 \
# 		--model fcn_resnet50 \
# 		--aux-loss \
# 		--data-path=/home/user/dataset/COCO \
# 		--epochs 30 \
# 		--weight_decay $i \
# 		--num_sample 5000
# done

# Hansang
# for i in 0.0002 0.0004 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./segmentation/train.py \
# 		--lr 0.02 \
# 		--dataset coco \
# 		--batch_size 32 \
# 		--model fcn_resnet50 \
# 		--aux-loss \
# 		--data-path=/home/user/dataset/COCO \
# 		--epochs 30 \
# 		--weight_decay $i \
# 		--num_sample 5000
# done

# Jaeyoung
# for i in 10 20 30
# do
# 	CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 ./segmentation/train.py \
# 		--lr 0.02 \
# 		--dataset coco \
# 		--batch_size 32 \
# 		--model fcn_resnet50 \
# 		--aux-loss \
# 		--data-path=/home/user/dataset/COCO \
# 		--epochs $i \
# 		--weight_decay 0.0001 \
# 		--num_sample 5000
# done

# ECCV
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./segmentation/train.py \
# 		--lr 0.02 \
# 		--dataset coco \
# 		--batch_size 32 \
# 		--model fcn_resnet50 \
# 		--aux-loss \
# 		--data-path=/home/user/dataset/COCO \
# 		--epochs 10 \
# 		--weight_decay $i \
# 		--num_sample 80000
# done
# # ECCV
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./segmentation/train.py \
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

############################## 220525 ###########################
# jaeyoung
# for i in 25000 12500
# do
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.5.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--decay_epoch 150 225 \
# 					--batch_size 128 \
# 					--lr 0.1 \
# 					--weight_decay 0.0001 \
# 					--momentum 0.9 \
# 					--schedule_type 'step' \
# 					--net_type resnet18_LN_invariant \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warmup_epochs 0 \
# 					--save_dir './logs_inv/'
# done

# group4
# for i in 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.5.py \
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
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.5.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--decay_epoch 150 225 \
# 					--batch_size 128 \
# 					--lr 0.1 \
# 					--weight_decay 0.0001 \
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
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.5.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--decay_epoch 150 225 \
# 					--batch_size 128 \
# 					--lr 0.1 \
# 					--weight_decay 0.0001 \
# 					--momentum 0.9 \
# 					--schedule_type 'step' \
# 					--net_type resnet18_LN_invariant \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warmup_epochs 0 \
# 					--save_dir './logs_inv/'
# done

############################# 220414 ###########################
# for i in -8
# do	
# 	wd=$(echo 2^$i | bc -l)
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.5.py \
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

############################# 220411 ###########################
# for i in 0.0001 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=0,1 python direction.0.0.5.py \
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
# for i in 0.0008 0.0007
# do
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.5.py \
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
# for i in 0.0008 0.0007
# do
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.5.py \
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

# for i in 0.0016 0.0014 0.0034
# do
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.5.py \
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
# for i in 0.0008 0.0007
# do
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.5.py \
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

# for i in 0.0016 0.0014
# do
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.5.py \
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
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.3.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--decay_epoch 150 225 \
# 					--batch_size 128 \
# 					--optim AdamW \
# 					--lr 0.0008 \
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
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.3.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--decay_epoch 150 225 \
# 					--batch_size 128 \
# 					--optim AdamW \
# 					--lr 0.001 \
# 					--weight_decay 0.08 \
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
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.3.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--decay_epoch 150 225 \
# 					--batch_size 128 \
# 					--optim AdamW \
# 					--lr 0.020 \
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
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.3.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--decay_epoch 150 225 \
# 					--batch_size 128 \
# 					--optim AdamW \
# 					--lr 0.010 \
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
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.3.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--decay_epoch 150 225 \
# 					--batch_size 128 \
# 					--optim AdamW \
# 					--lr 0.014 \
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
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.3.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--decay_epoch 150 225 \
# 					--batch_size 128 \
# 					--optim AdamW \
# 					--lr 0.006 \
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
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.3.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--decay_epoch 150 225 \
# 					--batch_size 128 \
# 					--optim AdamW \
# 					--lr 0.006 \
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
# for i in 0 25000 12500 6250 3125
# do
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.3.py \
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
# 					--save_dir './logs_AdamW/'
# done

############################# 220330 ###########################
# for i in 0 25000 12500 6250 3125
# do
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.3.py \
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
# 					--save_dir './logs_AdamW/'
# done

# for i in 0 25000 12500 6250 3125
# do
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.3.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--decay_epoch 150 225 \
# 					--batch_size 128 \
# 					--optim AdamW \
# 					--lr 0.001 \
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
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.3.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--decay_epoch 150 225 \
# 					--batch_size 128 \
# 					--optim AdamW \
# 					--lr 0.001 \
# 					--weight_decay 0.1 \
# 					--momentum 0.9 \
# 					--schedule_type 'step' \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warmup_epochs 0 \
# 					--save_dir './logs_AdamW/'
# done

############################# 220318 ###########################
# for i in 0 25000 12500 6250 3125
# do
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.4.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 600 \
# 					--decay_epoch 300 450 \
# 					--batch_size 128 \
# 					--lr 0.1414 \
# 					--weight_decay 0.0001414 \
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
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.4.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 600 \
# 					--decay_epoch 300 450 \
# 					--batch_size 128 \
# 					--lr 0.2 \
# 					--weight_decay 0.0002 \
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
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.4.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 600 \
# 					--decay_epoch 300 450 \
# 					--batch_size 128 \
# 					--lr 0.2828 \
# 					--weight_decay 0.0002828 \
# 					--momentum 0.9 \
# 					--schedule_type 'step' \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warmup_epochs 0 \
# 					--save_dir './logs_epoch/'
# done

# ############################# 220316 ###########################
# for i in 0 25000 12500 6250 3125
# do
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.4.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 150 \
# 					--decay_epoch 75 112.5 \
# 					--batch_size 128 \
# 					--lr 0.1414 \
# 					--weight_decay 0.0001414 \
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
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.4.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 150 \
# 					--decay_epoch 75 112.5 \
# 					--batch_size 128 \
# 					--lr 0.2 \
# 					--weight_decay 0.0002 \
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
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.4.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 150 \
# 					--decay_epoch 75 112.5 \
# 					--batch_size 128 \
# 					--lr 0.2828 \
# 					--weight_decay 0.0002828 \
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
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.4.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 150 \
# 					--decay_epoch 75 112.5 \
# 					--batch_size 128 \
# 					--lr 0.4 \
# 					--weight_decay 0.0004 \
# 					--momentum 0.9 \
# 					--schedule_type 'step' \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 \
# 					--warmup_epochs 0 \
# 					--save_dir './logs_epoch/'
# done

############################# 220313 ###########################
# GCP
# for i in 0.000025 0.00005 0.0001 0.0002 0.0004 0.0008 0.0016
# do
# 	CUDA_VISIBLE_DEVICES=8 python direction.0.0.4.py \
# 					--GCP \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample 0 \
# 					--epochs 150 \
# 					--angle_freq 195 \
# 					--decay_epoch 75 112.5 \
# 					--batch_size 128 \
# 					--lr 0.1 \
# 					--weight_decay $i \
# 					--momentum 0.9 \
# 					--schedule_type 'step' \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warmup_epochs 0 \
# 					--save_dir './logs_epoch/'
# done

############################## 220306 ###########################
# GCP
# for i in 2048 1024 512 256
# do
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.3.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample 0 \
# 					--epochs 300 \
# 					--decay_epoch 150 225 \
# 					--batch_size $i \
# 					--lr 1.414 \
# 					--weight_decay 0.0001414 \
# 					--momentum 0 \
# 					--schedule_type 'step' \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warmup_epochs 0 \
# 					--save_dir './logs_inv_wo_moment/'
# done

############################## 220302 ###########################
# GCP
# for i in 0 25000 12500 6250 3125
# do
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.3.py \
# 					--dataset cifar10 \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--decay_epoch 150 225 \
# 					--batch_size 128 \
# 					--lr 0.1414 \
# 					--weight_decay 0.0001414 \
# 					--momentum 0.9 \
# 					--schedule_type 'cosine' \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warmup_epochs 0 \
# 					--save_dir './logs_inv_cosine/'
# done

############################## 220226 ###########################
# 8gpu-ICML3
# for i in 0 
# do
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size 32 \
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
# for i in 0.0001 0.00005 0.000025 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python direction.0.0.2.py \
# 					--dataset imagenet \
# 					--num_workers 40 \
# 					--num_sample 320291 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 1024 \
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
# for i in 0.0000125 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0,1 python direction.0.0.2.py \
# 					--dataset imagenet \
# 					--num_workers 10 \
# 					--num_sample 40036 \
# 					--epochs 120 \
# 					--epoch_step 60 90 110 \
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

# GCP
# for i in 0
# do
# 	CUDA_VISIBLE_DEVICES=0,1 python direction.0.0.2.py \
# 					--GCP \
# 					--dataset imagenet \
# 					--num_workers 12 \
# 					--num_sample $i \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 256 \
# 					--lr 0.1 \
# 					--weight_decay 0.0001 \
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
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.2.py \
# 					--GCP \
# 					--dataset imagenet \
# 					--num_workers 16 \
# 					--num_sample $i \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 256 \
# 					--lr 0.1 \
# 					--weight_decay 0.0001 \
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
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.2.py \
# 					--GCP \
# 					--dataset imagenet \
# 					--num_workers 16 \
# 					--num_sample $i \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 256 \
# 					--lr 0.1 \
# 					--weight_decay 0.0001 \
# 					--momentum 0.9 \
# 					--net_type efficientnet_b0_inv \
# 					--save_model \
# 					--seed 0 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_trash/'
# done

############################## 220220 ###########################
# GCP
# CUDA_VISIBLE_DEVICES=0 python invariant_test_v2.py \
# 				--GCP \
# 				--amp \
# 				--dataset imagenet \
# 				--epochs 2 \
# 				--batch_size 128 \
# 				--weight_decay1 0.0002 \
# 				--lr1 0.1 \
# 				--alpha_sqaure 100 \
# 				--momentum 0.9 \
# 				--net_type efficientnet_b0_inv \
# 				--save_model \
# 				--seed 0 \
# 				--warm_up_epoch 0 \
# 				--save_dir './logs_trash_invariant_test/'

############################## 220218 ###########################
# ICML2
# CUDA_VISIBLE_DEVICES=0 python invariant_test.py \
# 				--dataset cifar10 \
# 				--epochs 2 \
# 				--batch_size 512 \
# 				--weight_decay1 0.0002 \
# 				--weight_decay1 0.0008 \
# 				--lr1 0.4 \
# 				--lr2 0.1 \
# 				--momentum 0.9 \
# 				--net_type efficientnet_b0_inv \
# 				--save_model \
# 				--seed 0 1 \
# 				--zero_init_residual \
# 				--warm_up_epoch 0 \
# 				--base_norm 128 \
# 				--amp 


# GCP
# for i in 0
# do
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.2.py \
# 					--GCP \
# 					--eps 1e-05 \
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

############################## 220214 ###########################
# ICML
# for i in 0
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 python direction.0.0.2.py \
# 					--dataset imagenet \
# 					--num_workers 40 \
# 					--num_sample $i \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 256 \
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
# ICML
# for i in 0
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 python direction.0.0.2.py \
# 					--dataset imagenet \
# 					--num_workers 40 \
# 					--num_sample $i \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 256 \
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
# ICML3
# for i in 0
# do
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size 128 \
# 					--lr 0.1 \
# 					--weight_decay 0.0002 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_inv_observe/'
# done

# 8gpu-ICML1
# for i in 0
# do
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.2.py \
# 					--dataset tinyimagenet \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 120 \
# 					--epoch_step 60 90 \
# 					--batch_size 256 \
# 					--lr 0.5657 \
# 					--weight_decay 0.0005657 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_inv_stl \
# 					--save_model \
# 					--seed 0 1 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_inv/'
# done

# group4
# for i in 12500
# do
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.2.py \
# 					--dataset tinyimagenet \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 120 \
# 					--epoch_step 60 90 \
# 					--batch_size 256 \
# 					--lr 0.5657 \
# 					--weight_decay 0.0005657 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_inv_stl \
# 					--save_model \
# 					--seed 0 1 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_inv/'
# done

# # 8gpu-ICML3
# for i in 12500 25000 50000 0
# do
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.2.py \
# 					--dataset tinyimagenet \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 120 \
# 					--epoch_step 60 90 \
# 					--batch_size 256 \
# 					--lr 0.1414 \
# 					--weight_decay 0.0001414 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_inv_stl \
# 					--save_model \
# 					--seed 0 1 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_inv/'
# done

############################## 220126 ###########################
# 8gpu-ICML1
# for i in 25000 50000
# do
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.2.py \
# 					--dataset tinyimagenet \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 120 \
# 					--epoch_step 60 90 \
# 					--batch_size 256 \
# 					--lr 0.2 \
# 					--weight_decay 0.0001 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_inv_stl \
# 					--save_model \
# 					--seed 0 1 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_inv/'
# done

# # 8gpu-ICML1
# for i in 12500 25000 50000 0
# do
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.2.py \
# 					--dataset tinyimagenet \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 120 \
# 					--epoch_step 60 90 \
# 					--batch_size 256 \
# 					--lr 0.1 \
# 					--weight_decay 0.0064 \
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
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.2.py \
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

# for i in 6250
# do
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.2.py \
# 					--dataset tinyimagenet \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 120 \
# 					--epoch_step 60 90 \
# 					--batch_size 256 \
# 					--lr 0.8 \
# 					--weight_decay 0.0001 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_inv_stl \
# 					--save_model \
# 					--seed 0 1 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_inv/'
# done

# 8gpu-ICML1
# for i in 12500
# do
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.2.py \
# 					--dataset tinyimagenet \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 120 \
# 					--epoch_step 60 90 \
# 					--batch_size 256 \
# 					--lr 0.2 \
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
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.2.py \
# 					--dataset cifar100 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size 128 \
# 					--lr 0.1 \
# 					--weight_decay 0.0002 \
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
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.2.py \
# 					--dataset tinyimagenet \
# 					--num_workers 2 \
# 					--num_sample $i \
# 					--epochs 120 \
# 					--epoch_step 60 90 \
# 					--batch_size 256 \
# 					--lr 0.1 \
# 					--weight_decay 0.0001 \
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
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.2.py \
# 					--dataset cifar100 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size 128 \
# 					--lr 0.1 \
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

# python stl_mean_std.py



# 8gpu-ICML3
# for i in 6250 12500 25000 0
# do
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size 128 \
# 					--lr 0.1 \
# 					--weight_decay 0.0002 \
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
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size 128 \
# 					--lr 0.1 \
# 					--weight_decay 0.0001 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_var/'
# done

# # 8gpu-ICML3
# for i in 3125 6250 12500 25000
# do
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size 128 \
# 					--lr 6.4 \
# 					--weight_decay 0.0001 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_inv/'
# done

############################## 220124 ###########################
# 8gpu-group4
# for i in 3125 6250 12500 25000 50000
# do
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size 128 \
# 					--lr 1 \
# 					--weight_decay 0.0002 \
# 					--momentum 0 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_inv_wo_moment/'
# done

# 8gpu-ICML2
# for i in 3125 6250 12500 25000 50000
# do
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample $i \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size 128 \
# 					--lr 64 \
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
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.2.py \
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

# 8gpu-ICML2
# for i in 2048
# do
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample 0 \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size $i \
# 					--lr 0.1 \
# 					--weight_decay 0.0001 \
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
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.2.py \
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

# 8gpu-JDG
# for i in 0.1 0.2 0.4 0.8 1.6 3.2
# do
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample 25000 \
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
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample 0 \
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

############################## 220122 ###########################
# 8gpu-ICML
# for i in 0.25
# do
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample 25000 \
# 					--epochs 300 \
# 					--epoch_step 150 225 \
# 					--batch_size 128 \
# 					--weight_decay 0.0004 \
# 					--lr 0.1 \
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
# 	CUDA_VISIBLE_DEVICES=0 python direction.0.0.2.py \
# 					--dataset cifar10 \
# 					--num_sample 0 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.0002 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--warm_up_epoch 0 \
# 					--save_dir './logs_observation/'
# done


# # 8gpu
# for i in 0.015625
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python invariant.1.0.2.1.py \
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



############################## 200603 ###########################
# DGX
# for i in 0.25
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python speed_test.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 80 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 1024 \
# 					--weight_decay 0.0001 \
# 					--lr 0.4 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 0 \
# 					--alpha_sqaure $i
# done

############################## 200603 ###########################
# 8gpu - hong
# for i in 0.25
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python invariant.1.0.2.3.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 80 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 1024 \
# 					--weight_decay 0.0001 \
# 					--lr 0.4 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 0 \
# 					--alpha_sqaure $i
# done

############################## 200603 ###########################
# 8gpu - hong
# for i in 0.03125
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python invariant.1.0.2.3.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 8192 \
# 					--weight_decay 0.0001 \
# 					--lr 3.2 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--alpha_sqaure $i
# done

# DGX
# for i in 0.5
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 python invariant.1.0.2.3.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 90 \
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


############################## 200602 ###########################
# DGX
# for i in 0.25
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python invariant.1.0.2.3.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 1024 \
# 					--weight_decay 0.0001 \
# 					--lr 0.4 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 0 \
# 					--alpha_sqaure $i \
# 					--load_dir logs_invariant_no_scale_w/imagenet/resnet50_GBN_invariant2/batch_1024/invariant_normx_alpha_sq_0.25_WD_0.0004_lr0.1_eps_2.5e-06_zero_init_True_warm_iter_0_filter_bn_bias_False_momentum0.9_nester_True_epoch90_amp_True_seed0_save0/0/
# done

# DGX
# for i in 0.25
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python invariant.1.0.2.3.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 1024 \
# 					--weight_decay 0.0001 \
# 					--lr 0.4 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 0 \
# 					--alpha_sqaure $i \
# 					--load_dir logs_invariant_no_scale_w/imagenet/resnet50_GBN_invariant2/batch_1024/invariant_normx_alpha_sq_0.25_WD_0.0004_lr0.1_eps_2.5e-06_zero_init_True_warm_iter_0_filter_bn_bias_False_momentum0.9_nester_True_epoch90_amp_True_seed0_save0/0/
# done

# 8gpu - h
# for i in 0.5
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 python invariant.1.0.2.3.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 10 \
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
# 					--warm_up_epoch 0 \
# 					--alpha_sqaure $i
# done

# 8gpu - h
# for i in 0.25
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python invariant.1.0.2.3.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 30 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 1024 \
# 					--weight_decay 0.0001 \
# 					--lr 0.4 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 0 \
# 					--alpha_sqaure $i \
# 					--load_dir logs_invariant_no_scale_w/imagenet/resnet50_GBN_invariant2/batch_1024/invariant_normx_alpha_sq_0.25_WD_0.0004_lr0.1_eps_2.5e-06_zero_init_True_warm_iter_0_filter_bn_bias_False_momentum0.9_nester_True_epoch90_amp_True_seed0_save0/0/
# done

############################## 200601 ###########################
# DGX
# for i in 0.0625
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python invariant.1.0.2.3.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 4096 \
# 					--weight_decay 0.0001 \
# 					--lr 1.6 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 0 \
# 					--alpha_sqaure $i
# done

# 8gpu-h
# for i in 0.25
# do
# 	CUDA_VISIBLE_DEVICES=7 python invariant.1.0.2.3.py \
# 					--dataset cifar10 \
# 					--num_workers 40 \
# 					--epochs 2 \
# 					--epoch_step 2 4 \
# 					--batch_size 1024 \
# 					--weight_decay 0.0001 \
# 					--lr 0.4 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 0 \
# 					--alpha_sqaure $i
# done

# 8gpu-h
# for i in 0.25
# do
# 	CUDA_VISIBLE_DEVICES=7 python invariant.1.0.2.3.py \
# 					--dataset cifar10 \
# 					--num_workers 40 \
# 					--epochs 10 \
# 					--epoch_step 2 4 \
# 					--batch_size 1024 \
# 					--weight_decay 0.0001 \
# 					--lr 0.4 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 0 \
# 					--alpha_sqaure $i \
# 					--load_dir logs_invariant_no_scale_w/cifar10/resnet18_GBN_invariant2/batch_1024/invariant_normx_alpha_sq_0.25_WD_0.0004_lr0.1_eps_2.5e-06_zero_init_True_warm_iter_0_filter_bn_bias_False_momentum0.9_nester_True_epoch2_amp_True_seed0_save0/0/
# done



# 8gpu
# for i in 0.25
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python invariant.1.0.2.3.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 1024 \
# 					--weight_decay 0.0001 \
# 					--lr 0.4 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 0 \
# 					--alpha_sqaure $i
# done

# DGX
# for i in 0.125
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python invariant.1.0.2.3.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 2048 \
# 					--weight_decay 0.0001 \
# 					--lr 0.8 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 0 \
# 					--alpha_sqaure $i
# done

# DGX
# for i in 0.125
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python invariant.1.0.2.3.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 2048 \
# 					--weight_decay 0.0001 \
# 					--lr 0.8 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 0 \
# 					--alpha_sqaure $i
# done

############################## 200527 ###########################
# # 8gpu
# for i in 0.015625
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python invariant.1.0.2.1.py \
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
# 	CUDA_VISIBLE_DEVICES=0,1 python invariant.1.0.2.1.py \
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
#  					--warm_up_epoch 0 \
#  					--save_dir './logs_invariant_scaler/' \
#  					--alpha_sqaure $i
# done

# DGX
# for i in 0.25
# do
# 	CUDA_VISIBLE_DEVICES=0 python invariant.1.0.2.3.py \
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
# 					--warm_up_epoch 0 \
# 					--alpha_sqaure $i
# done

# 8gpu
# for i in 0.03125
# do
# 	CUDA_VISIBLE_DEVICES=0,1 python invariant.1.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay 0.0002 \
# 					--lr 3.2 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_bn_fix \
# 					--save_model \
# 					--seed 0 1 2 \
# 					--zero_init_residual \
# 					--warm_up_epoch 0 \
# 					--alpha_sqaure $i
# done

# 8gpu
# for i in 0.25
# do
# 	CUDA_VISIBLE_DEVICES=0 python invariant.1.0.2.3.py \
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
# 					--warm_up_epoch 0 \
# 					--alpha_sqaure $i
# done

############################## 200526 ###########################
# 8gpu
# for i in 0.015625
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 python invariant.1.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 8192 \
# 					--weight_decay 0.0002 \
# 					--lr 6.4 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2_avg \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
# 					--warm_up_epoch 0 \
# 					--alpha_sqaure $i
# done

############################## 200523 ###########################
# # 8gpu
# for i in 0.015625
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python invariant.1.0.2.py \
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
# 					--warm_up_epoch 15 \
# 					--alpha_sqaure $i
# done

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python invariant.1.0.2.py \
# 				--dataset cifar10 \
# 				--epochs 300 \
# 				--batch_size 4096 \
# 				--weight_decay 0.0001 \
# 				--lr 6.4 \
# 				--momentum 0.9 \
# 				--net_type densenetBC100_GBN_invariant \
# 				--save_model \
# 				--seed 0 1 2 3 4 \
# 				--zero_init_residual \
# 				--warm_up_epoch 15

############################## 200523 ###########################
# # DGX
	# CUDA_VISIBLE_DEVICES=0 python invariant.1.0.2.py \
	# 				--dataset cifar10 \
	# 				--epochs 300 \
	# 				--batch_size 64 \
	# 				--weight_decay 0.0001 \
	# 				--lr 0.1 \
	# 				--momentum 0.9 \
	# 				--net_type densenetBC100_GBN_invariant \
	# 				--save_model \
	# 				--seed 0 1 2 3 4 \
	# 				--zero_init_residual \
	# 				--warm_up_epoch 15

# 8gpu
# CUDA_VISIBLE_DEVICES=0 python invariant.1.0.2.py \
# 				--dataset cifar10 \
# 				--epochs 300 \
# 				--batch_size 128 \
# 				--weight_decay 0.0002 \
# 				--lr 0.1 \
# 				--momentum 0.9 \
# 				--net_type resnet18_GBN_invariant2 \
# 				--save_model \
# 				--seed 0 1 2 3 4 \
# 				--zero_init_residual \
# 				--warm_up_epoch 15 \
# 				--alpha_sqaure 1

# # DGX
# 	CUDA_VISIBLE_DEVICES=0 python invariant.1.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 256 \
# 					--weight_decay 0.0002 \
# 					--lr 0.2 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
# 					--warm_up_epoch 15
 					
# 8gpu
# for i in 0.5
# do
# 	CUDA_VISIBLE_DEVICES=0 python invariant.1.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 256 \
# 					--weight_decay 0.0002 \
# 					--lr 0.2 \
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
# for i in 1e-3
# do
# 	CUDA_VISIBLE_DEVICES=0 python invariant.1.0.2.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.00001 \
# 					--lr 0.001 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--alpha $i
# done

# 8gpu
# for i in 0.01
# do
# 	CUDA_VISIBLE_DEVICES=0 python invariant.1.0.2.2.py \
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
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0 python invariant.1.0.2.py \
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
# for i in 1563
# do
# 	CUDA_VISIBLE_DEVICES=0 python invariant_num.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.0002 \
# 					--lr 3.2 \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--num_sample $i \
#  					--base_norm 4096 \
#  					--eps 0.00032
# done

############################## 200520 ###########################
# DGX
# for i in 1e+13
# do
# 	CUDA_VISIBLE_DEVICES=0 python invariant.1.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-18  \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e+16 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e+09
# done

# 8gpu
# for i in 1e-01
# do
# 	CUDA_VISIBLE_DEVICES=0 python invariant.1.0.2.py \
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
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 python invariant.1.0.2.py \
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
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5
# done

############################## 200518 ###########################
# 8gpu
# for i in 1e-01
# do
# 	CUDA_VISIBLE_DEVICES=0 python invariant.1.0.3.2.py \
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
#  					--save_dir './logs_delete/' \
#  					--eps 1e-05 \
#  					--epoch_step 150 225 \
#  					--lr_decay 0.1 \
#  					--wd_linear 0.0002
# done

############################## 200517 ###########################
# DGX
# for i in 0.4
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python invariant.1.0.2.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 1024 \
# 					--weight_decay 0.0001 \
# 					--lr $i \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5
# done

############################## 200517 ###########################
# # 8gpu
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=0,1 python invariant.1.0.2.1.py \
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
#  					--eps 7.8125e-08 \
#  					--base_norm 32
# done

# # 8gpu
# for i in 0.0004
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python invariant.1.0.2.1.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 1024 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--base_norm 256 \
# 					--eps 2.5e-06
# done

############################## 200516 ###########################
# 8gpu-h
# for i in 1e-01
# do
# 	CUDA_VISIBLE_DEVICES=0 python invariant.1.0.3.1.py \
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
#  					--save_dir './logs_invariant_tuning/' \
#  					--eps 1e-05 \
#  					--epoch_step 150 225 \
#  					--lr_decay 0.1 \
#  					--wd_linear 0.0002
# done

############################## 200515 ###########################
# 8gpu-h
# for i in 1e-00
# do
# 	CUDA_VISIBLE_DEVICES=0 python invariant.1.0.3.1.py \
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
#  					--epoch_step 30 150 250 \
#  					--lr_decay 0.1 \
#  					--wd_linear 0.0005
# done

# DGX
# for i in 1e-04
# do
# 	CUDA_VISIBLE_DEVICES=0 python invariant.1.0.2.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-01 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e-01 \
#  					--save_dir './logs_invariant_scaler/' \
#  					--eps 1e-08 \
#  					--amp
# done

# 8gpu-h
# for i in 1e-01
# do
# 	CUDA_VISIBLE_DEVICES=0 python invariant.1.0.3.py \
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
# for i in 1e-04
# do
# 	CUDA_VISIBLE_DEVICES=0 python invariant.1.0.2.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-01 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e-01 \
#  					--save_dir './logs_invariant_scaler/' \
#  					--eps 1e-08
# done

# DGX
# for i in 1e-04
# do
# 	CUDA_VISIBLE_DEVICES=0 python variant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-01 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e-01 \
#  					--save_dir './logs_variant_same/' \
#  					--amp \
#  					--eps 1e-08
# done

# group4
# for i in 1e+08
# do
# 	CUDA_VISIBLE_DEVICES=0 python invariant.1.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-13  \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e+11 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e+04
# done

# # 8gpu-h
# for i in 1e-04
# do
# 	CUDA_VISIBLE_DEVICES=0 python invariant.1.0.3.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-01 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e-01 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e-08
# done

############################## 200514 ###########################
# DGX
# for i in 1.6
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python invariant.1.0.2.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 4096 \
# 					--weight_decay 0.0001 \
# 					--lr $i \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5
# done

# # DGX
# for i in 0.8
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python invariant.1.0.2.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 2048 \
# 					--weight_decay 0.0001 \
# 					--lr $i \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5
# done

# # 8gpu-h
# for i in 1e-04
# do
# 	CUDA_VISIBLE_DEVICES=0 python invariant.1.0.3.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-01 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1.28e-01 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e-08
# done

############################## 200513 ###########################
# # 8gpu
# for i in 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python invariant.1.0.2.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 2048 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--base_norm 256 \
# 					--eps 1.25e-06
# done

# # 8gpu
# for i in 0.0016
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python invariant.1.0.2.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 4096 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--base_norm 256 \
# 					--eps 6.25e-07
# done

# 8gpu-h
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0 python variant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.2 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 0.128 \
#  					--save_dir './logs_variant_same/' \
#  					--amp \
#  					--eps 1.00E-08 \
#  					--filter_bn_bias \
#  					--lr_linear 0.0001
# done

############################## 200512 ###########################
# 8gpu
# for i in 3.2
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python invariant.1.0.2.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 8192 \
# 					--weight_decay 0.0001 \
# 					--lr $i \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5
# done

############################## 200511 ###########################
# 8gpu-h
# for i in 1e+05
# do
# 	CUDA_VISIBLE_DEVICES=0 python invariant.1.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 2e-10 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 128000000 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 10
# done

# 8gpu
# for i in 0.0004
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python invariant.1.0.2.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 1024 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--base_norm 256 \
# 					--eps 2.5e-06
# done

# 8gpu-lee
# for i in 0.000123123
# do
# 	CUDA_VISIBLE_DEVICES=0 python invariant.1.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 2 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_invariant \
# 					--save_model \
# 					--seed 0 1 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--epoch_step 1 
# done

############################## 200510 ###########################
# # 8gpu-p
# for i in 0
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 python invariant.1.0.1.py \
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
# for i in 0
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 python invariant.1.0.0.py \
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

# DGX
# for i in 6250 12500 25000 50000
# do
# 	CUDA_VISIBLE_DEVICES=0 python invariant_num.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 64 \
# 					--weight_decay 0.0064 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--num_sample $i
# done

# DGX
# for i in 0 1 2 3 4
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python invariant.1.0.1.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 8192 \
# 					--weight_decay 0.0032 \
# 					--lr 0.1 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN_invariant2 \
# 					--save_model \
# 					--seed $i \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--base_norm 256 \
# 					--eps 3.125e-07
# done

############################## 200509 ###########################
# 8gpu-lee
# for i in 6250
# do
# 	CUDA_VISIBLE_DEVICES=0 python invariant_num.1.0.0.py \
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

############################## 200508 ###########################
# DGX
# for i in 1563 3125 6250 12500 25000 50000
# do
# 	CUDA_VISIBLE_DEVICES=0 python invariant_num.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 64 \
# 					--weight_decay 0.0064 \
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
# for i in 0.1
# do
# 	CUDA_VISIBLE_DEVICES=0 python invariant.1.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 64 \
# 					--weight_decay $i \
# 					--lr 0.0001 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 0.064 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp \
#  					--eps 1e-08
# done

# # 8gpu-lee
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0 python invariant_num.1.0.0.py \
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

############################## 200507 ###########################
# 8gpu-D
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python invariant.1.0.1.py \
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
# 					--net_type resnet50_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--base_norm 256 \
# 					--eps 3.125e-07
# done

# # group4
# for i in 0.0032 0.0002 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0 python invariant_num.1.0.0.py \
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
# for i in 0.0008 0.0004 0.0002 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0 python invariant_num.1.0.0.py \
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
#  					--num_sample 6250
# done

# # group4
# for i in 0.0128 0.0064 0.0032 0.0016
# do
# 	CUDA_VISIBLE_DEVICES=0 python invariant_num.1.0.0.py \
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
#  					--num_sample 6250
# done

# # group2
# for i in 0.4
# do
# 	CUDA_VISIBLE_DEVICES=0 python invariant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 256 \
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
# for i in 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 python invariant.1.0.1.py \
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
# # group2
# for i in 1.6
# do
# 	CUDA_VISIBLE_DEVICES=0 python invariant.1.0.0.py \
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

# 8gpu-p
# for i in 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=0 python invariant.1.0.1.py \
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
#  					--base_norm 128 \
#  					--save_dir './logs_invariant_same/' \
#  					--amp
# done

############################## 200503 ###########################
# # group4
# for i in 0.00002
# do
# 	CUDA_VISIBLE_DEVICES=0 python invariant.1.0.1.py \
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
#  					--eps 1e-7
# done

# 8gpu-b
# for i in 0.00002
# do
# 	CUDA_VISIBLE_DEVICES=0 python invariant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size de \
# 					--weight_decay $i \
# 					--lr 1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 1280 \
#  					--save_dir './logs_invariant_same/'
# done

# # group2
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0 python invariant.1.0.0.py \
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

############################## 200501 ###########################
# # 8gpu-b
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 python invariant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 2048 \
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

# 8gpu
# for i in 0.1
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 python invariant.1.0.0.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 256 \
# 					--weight_decay 0.0001 \
# 					--lr $i \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
# 					--warm_up_epoch 0
# done

# # 8gpu-p
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 python invariant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
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


# # group2
# for i in 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=0 python invariant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 512 \
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

# # group2
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



# 8gpu-b
# for i in 0.00002
# do
# 	CUDA_VISIBLE_DEVICES=0 python invariant.1.0.0.py \
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
#  					--save_dir './logs_invariant_same/'
# done

############################## 200430 ###########################
# 8gpu-p
# for i in 0.064
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 python invariant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
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

# # 8gpu
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 python invariant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 2048 \
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
# for i in 3.2
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 python invariant.1.0.0.py \
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

############################## 200429 ###########################
# 8gpu-b
# for i in 0.0128
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 python invariant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 8192 \
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
# for i in 0.0004
# do
# 	CUDA_VISIBLE_DEVICES=0 python invariant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 256 \
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

# # group2
# for i in 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=0 python invariant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 512 \
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

# 8gpu-p
# for i in 0.0128
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python invariant.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 8192 \
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

# 8gpu-p
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

# 8gpu
# for i in 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=0 python invariant.1.0.0.py \
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
#  					--base_norm 128
# done



############################## 200427 ###########################
# # group2
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=1 python invariant.1.0.0.py \
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
#  					--epoch_step 2
# done


# # 8gpu
# for i in 3.2
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scaler.1.3.1.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 8192 \
# 					--weight_decay 0.0001 \
# 					--lr $i \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5
# done

# 8gpu
# CUDA_VISIBLE_DEVICES=0 python invariant_test.py \
# 				--dataset cifar10 \
# 				--epochs 2 \
# 				--batch_size 512 \
# 				--weight_decay1 0.0002 \
# 				--weight_decay1 0.0002 \
# 				--lr1 0.1 \
# 				--lr2 0.1 \
# 				--momentum 0.9 \
# 				--net_type resnet18_GBN_invariant2 \
# 				--save_model \
# 				--seed 0 1 \
# 				--zero_init_residual \
# 				--warm_up_epoch 0 \
# 				--base_norm 512 \
# 				--amp 

############################## 200425 ###########################
# # 8gpu
# for i in 3.2
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scaler.1.0.0.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 8192 \
# 					--weight_decay 0.0001 \
# 					--lr $i \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--filter_bn \
# 					--filter_bias
# done

############################## 200425 ###########################
# # 8gpu
# for i in 3.2
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scaler.1.0.0.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 8192 \
# 					--weight_decay 0.0001 \
# 					--lr $i \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN_invariant2 \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--filter_bn \
# 					--filter_bias
# done

# # 8gpu
# for i in 3.2
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scaler.1.0.0.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 8192 \
# 					--weight_decay 0.0001 \
# 					--lr $i \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN_invariant \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--filter_bn \
# 					--filter_bias
# done

# # 8gpu-p
# for i in 0.8
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scaler.1.0.0.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 2048 \
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

# 8gpu
# CUDA_VISIBLE_DEVICES=0 python invariant_test.py \
# 				--dataset cifar10 \
# 				--epochs 2 \
# 				--batch_size 512 \
# 				--weight_decay1 0.0002 \
# 				--weight_decay1 0.0008 \
# 				--lr1 0.4 \
# 				--lr2 0.1 \
# 				--momentum 0.9 \
# 				--net_type resnet18_GBN_invariant2 \
# 				--save_model \
# 				--seed 0 1 \
# 				--zero_init_residual \
# 				--warm_up_epoch 0 \
# 				--base_norm 128 \
# 				--amp 

############################## 200420 ###########################
# # 8gpu
# for i in 3.2
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 python scaler.1.3.1.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 8192 \
# 					--weight_decay 0.0001 \
# 					--lr $i \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN_invariant \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 0
# done

############################## 200419 ###########################
# # group2
# for i in 3.2
# do
# 	CUDA_VISIBLE_DEVICES=0,1 python scaler.1.3.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay 0.0002 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0
# done

# # group2
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=0 python scaler.1.3.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 8 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant \
# 					--save_model \
# 					--seed 123123 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 128
# done

############################## 200418 ###########################
# # 8gpu-p
# for i in 0.4
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scaler.1.0.0.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 1024 \
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

# # 8gpu
# for i in 3.2
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 python scaler.1.3.1.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 8192 \
# 					--weight_decay 0.0001 \
# 					--lr $i \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN_invariant \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 0 \
# 					--filter_bn_bias
# done

############################## 200418 ###########################
# # group2
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=0,1 python scaler.1.3.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN_invariant \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0 \
#  					--base_norm 128
# done

############################## 200410 ###########################
# group2
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=0,1 python scaler.1.3.0.py \
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
#  					--warm_up_epoch 0 \
#  					--base_norm 128
# done

############################## 200410 ###########################
# 8gpu-p
# num_data densenet
# for i in 0.05 0.1 0.2 0.4 0.8 1.6 3.2 6.4
# do
# 	CUDA_VISIBLE_DEVICES=0 python num_data.1.0.0.py \
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
#  					--num_sample 3125
# done

# # 8gpu-p
# # num_data densenet
# for i in 0.05 0.1 0.2 0.4 0.8 1.6 3.2 6.4
# do
# 	CUDA_VISIBLE_DEVICES=0 python num_data.1.0.0.py \
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
#  					--num_sample 6250
# done

############################## 200409 ###########################
# 8gpu-p
# for i in 0.05 0.2 0.4 0.8 1.6 3.2 6.4 12.8
# do
# 	CUDA_VISIBLE_DEVICES=0 python num_data.1.0.0.py \
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
#  					--num_sample 6250
# done

# # 8gpu
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 python scaler.1.0.0.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 256 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
# 					--warm_up_epoch 0 \
# 					--filter_bn \
# 					--filter_bias
# done

############################## 200408 ###########################
# group2-0
# for i in 0.05 0.2 0.4 0.8 1.6 3.2 6.4 12.8
# do
# 	CUDA_VISIBLE_DEVICES=0 python num_data.1.0.0.py \
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
#  					--num_sample 12500
# done

############################## 200407 ###########################
# 8gpu
# for i in 0.0016
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scaler.1.1.1.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 4096 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 0 \
# 					--filter_bn \
# 					--filter_bias \
# 					--base_norm 256
# done

# group2
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=0,1 python scaler.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 2048 \
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
# for i in 0.0016
# do
# 	CUDA_VISIBLE_DEVICES=0 python scaler.1.0.0.py \
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
#  					--warm_up_epoch 0 \
#  					--base_norm 64
# done

############################## 200405 ###########################
# 8gpu
# for i in 0.032
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scaler.1.1.2.py \
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
# 					--net_type resnet50_GBN \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 0 \
# 					--filter_bn \
# 					--filter_bias \
# 					--base_norm 256
# done

# group2
# for i in 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=0 python scaler.1.0.0.py \
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
#  					--warm_up_epoch 0 \
#  					--base_norm 64
# done

# # group2
# for i in 0.0016
# do
# 	CUDA_VISIBLE_DEVICES=0 python scaler.1.0.0.py \
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
#  					--warm_up_epoch 0 \
#  					--base_norm 64
# done

############################## 200404 ###########################
# 8gpu
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scaler.1.1.1.py \
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
# 					--net_type resnet50_GBN \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 0 \
# 					--filter_bn \
# 					--filter_bias \
# 					--base_norm 256
# done


# IITP
# for i in 1.6
# do
# 	CUDA_VISIBLE_DEVICES=0 python lr_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 1024 \
# 					--weight_decay 0.0001 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0
# done

# 8gpu-p
# for i in 0.0128
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scaler.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 8192 \
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

# # 8gpu-p
# for i in 0.0128
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scaler.1.1.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 8192 \
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
# 8gpu
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scaler.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
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

# # 8gpu
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scaler.1.1.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
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

# 8gpu-p
# for i in 12.8
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python lr_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 8192 \
# 					--weight_decay 0.0001 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0
# done

# IITP
# for i in 1.6
# do
# 	CUDA_VISIBLE_DEVICES=0 python lr_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 1024 \
# 					--weight_decay 0.0001 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0
# done

# 8gpu
# for i in 6.4
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python lr_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay 0.0001 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0
# done

# # 8gpu
# for i in 3.2
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 python lr_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 2048 \
# 					--weight_decay 0.0001 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0
# done

# group2
# for i in 0.2
# do
# 	CUDA_VISIBLE_DEVICES=0 python lr_scaler.py \
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
#  					--warm_up_epoch 0
# done

# # group2
# for i in 0.2
# do
# 	CUDA_VISIBLE_DEVICES=0 python lr_scaler.py \
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
#  					--warm_up_epoch 0
# done

############################## 200401 ###########################
# IITP
# for i in 6.4
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 python lr_scaler.py \
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
#  					--warm_up_epoch 0
# done

############################## 200331 ###########################

# 8gpu
# for i in 0.1 0.2 0.4 0.8 1.6 3.2 6.4 12.8
# do
# 	CUDA_VISIBLE_DEVICES=0 python num_data.1.0.0.py \
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
#  					--num_sample 1563
# done

# # 8gpu
# for i in 0.1 0.2 0.4 0.8 1.6 3.2 6.4 12.8
# do
# 	CUDA_VISIBLE_DEVICES=0 python num_data.1.0.0.py \
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
#  					--num_sample 3125
# done

# 8gpu-j
# for i in 12.8
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python lr_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 8192 \
# 					--weight_decay 0.0001 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 15
# done

# # 8gpu-j
# for i in 0.0128
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python wd_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 8192 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 0
# done

# IITP
# for i in 3.2
# do
# 	CUDA_VISIBLE_DEVICES=0,1 python lr_scaler.py \
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
#  					--warm_up_epoch 0
# done


############################## 200330 ###########################
# IITP
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=0,1 python scaler.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 2048 \
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

############################## 200329 ###########################
# IITP
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=0,1 python scaler.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 2048 \
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

# 8gpu
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scaler.1.1.0.py \
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
# 					--net_type resnet50_GBN \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--filter_bn \
# 					--filter_bias \
# 					--base_norm 256
# done

# # 8gpu
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scaler.1.0.0.py \
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
# 					--net_type resnet50_GBN \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--filter_bn \
# 					--filter_bias \
# 					--base_norm 256
# done

# group2
# for i in 0.0004 0.0008 0.0016 0.0032 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=0 python num_data.1.0.0.py \
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
#  					--num_sample 25000
# done

############################## 200326 ###########################


# group2
# for i in 0.0001 0.0002 0.004 0.008 0.0016 0.0032 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=0 python num_data.1.0.0.py \
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
#  					--num_sample 25000
# done

# 8gpu
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scaler.1.0.1.py \
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
# 					--net_type resnet50_GBN \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--filter_bn \
# 					--filter_bias \
# 					--base_norm 256 \
# 					--amp
# done

# # 8gpu
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scaler.1.1.0.py \
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
# 					--net_type resnet50_GBN \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--filter_bn \
# 					--filter_bias \
# 					--base_norm 256
# done

############################## 200325 ###########################
# 8gpu
# for i in 3.2
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scaler.1.0.0.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 8192 \
# 					--weight_decay 0.0001 \
# 					--lr $i \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--filter_bn \
# 					--filter_bias 
# done


# IITP
# for i in 0.0001 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=0 python wd_scaler.py \
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
# for i in 1.6
# do
# 	CUDA_VISIBLE_DEVICES=0 python lr_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 1024 \
# 					--weight_decay 0.0001 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 15
# done

############################## 200323 ###########################
# group2
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0 python scaler.1.0.0.py \
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
#  					--warm_up_epoch 15 \
#  					--base_norm 256
# done

# # IITP
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 python scaler.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
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

# # IITP
# for i in 6.4
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 python lr_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay 0.0001 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 15
# done

# # IITP
# for i in 3.2
# do
# 	CUDA_VISIBLE_DEVICES=0,1 python lr_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 2048 \
# 					--weight_decay 0.0001 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 15
# done

# # 8gpu
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scaler.1.0.0.py \
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
# 					--net_type resnet50_GBN \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--filter_bn \
# 					--filter_bias \
# 					--base_norm 256
# done

# # 8gpu
# for i in 0.0004
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scaler.1.0.0.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 40 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 1024 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--filter_bn \
# 					--filter_bias \
# 					--base_norm 256
# done

############################## 200323 ###########################
# 8gpu
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python wd_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 8192 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
#  					--warm_up_epoch 15
# done

############################## 200321 ###########################
# 8gpu
# for i in 0.0128
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scaler.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 8192 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
#  					--warm_up_epoch 15
# done

# # IITP
# for i in 0.4
# do
# 	CUDA_VISIBLE_DEVICES=0 python lr_scaler.py \
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
#  					--warm_up_epoch 15
# done

# 8gpu
# for i in 6.4
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scaler.1.0.0.py \
# 					--dataset cifar10 \
# 					--epochs 3000000000000000000 \
# 					--batch_size 32768 \
# 					--weight_decay 0.0001 \
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

############################## 200320 ###########################
# 8gpu
# for i in 6.4
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python lr_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
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
# for i in 0.4
# do
# 	CUDA_VISIBLE_DEVICES=0 python lr_scaler.py \
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
#  					--warm_up_epoch 15 \
# 					--filter_bn \
# 					--filter_bias	
# done

############################## 200319 ###########################
# IITP
# for i in 1.6
# do
# 	CUDA_VISIBLE_DEVICES=0 python lr_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 1024 \
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

# 8gpu-b
# for i in 12.8
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python lr_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 8192 \
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
# IITP
# for i in 3.2
# do
# 	CUDA_VISIBLE_DEVICES=0,1 python lr_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 2048 \
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

# 8gpu
# for i in 0.0256 0.0128 0.0064 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 python wd_scaler.py \
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

############################## 200317 ###########################
# 8gpu
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python large_wd_scaler.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 32 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 8192 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--filter_bn \
# 					--filter_bias 
# done

############################## 200315 ###########################
# 8gpu
# for i in 0.0064 0.0016
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python large_wd_scaler.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 32 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 8192 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--filter_bn \
# 					--filter_bias	
# done

############################## 200312 ###########################
# IITP
# for i in 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=0,1 python critical.0.0.1.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 32 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 256 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--filter_bn \
# 					--filter_bias
# done

# 8gpu-ha
# for i in 0.0004 0.0002 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python wd_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 8192 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual
# done

############################## 200311 ###########################
# 8gpu-ha
# for i in 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python wd_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 8192 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual
# done

############################## 200310 ###########################
# 8gpu
# for i in 0.0256 0.0128 0.0004 0.0002 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python wd_scaler.py \
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

# 8gpu-ha
# for i in 0.0016
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python wd_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 8192 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual
# done

# group2
# for i in 0.0001 0.0002 0.0004 0.0008 0.0016 0.0032 0.0064 0.0128 0.0256
# do
# 	CUDA_VISIBLE_DEVICES=0 python wd_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 64 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual
# done

# 8gpu-h
# for i in 0.0016
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

# 8gpu-ha
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python wd_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 8192 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual
# done

############################## 200308 ###########################
# 8gpu
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python large_wd_scaler.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 32 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 8192 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--filter_bn \
# 					--filter_bias	
# done

# 8gpu
# for i in 0.0004
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python wd_scaler.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 32 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 1024 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--filter_bn \
# 					--filter_bias \
# 					--weight_multiplier 1		
# done

# group2
# for i in 0.2
# do
# 	CUDA_VISIBLE_DEVICES=0 python lr_scaler.py \
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
#  					--warm_up_epoch 15 \
# 					--filter_bn \
# 					--filter_bias	
# done

############################## 200306 ###########################
# 8gpu
# for i in 0.0004
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python wd_scaler.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 32 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 1024 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--filter_bn \
# 					--filter_bias		
# done

# 8gpu-b
# for i in 0.0128 0.0064 0.0256
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python wd_scaler.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 8192 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN \
# 					--save_model \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual
# done

# 8gpu-b
# for i in 0.0064 0.0032
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

# 8gpu-h
# for i in 0.0032
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

# 8gpu
# for i in 0.4
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python lr_scaler.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 32 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 1024 \
# 					--weight_decay 0.0001 \
# 					--lr $i \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN \
# 					--save_model \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--filter_bn \
# 					--filter_bias					 
# done

# 8gpu-ha
# for i in 0.0008 0.0004
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 python wd_scaler.py \
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
# 8gpu-ha
# for i in 0.0032 0.0016
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

# 8gpu-o
# for i in 0.0256 0.0064
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

# 8gpu-h
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
# 					--save_dir ./logs_lr/ \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
# 					--warm_up_epoch 15 \
# 					--filter_bn \
# 					--filter_bias					 
# done

# group2
# for i in 0.0001 0.0002 0.0004 0.0008 0.0016 0.0032 0.0128 0.0256
# do
# 	CUDA_VISIBLE_DEVICES=0,1 python critical.0.0.1.py \
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
# 					--zero_init_residual
# done

# 8gpu
# for i in 0.0004
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python critical.0.0.1.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 32 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 1024 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN \
# 					--save_model \
# 					--save_dir ./logs/ \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5
# done

# IITP
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0,1 python critical.0.0.1.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 32 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 256 \
# 					--weight_decay $i \
# 					--lr 0.1 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN \
# 					--save_model \
# 					--save_dir ./logs/ \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--filter_bn \
# 					--filter_bias
# done

# group4
# for i in 3.2
# do
# 	CUDA_VISIBLE_DEVICES=2,3 python main.0.0.2.py \
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
# 					--warm_up_epoch 15 
# done

# 8gpu
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=0,1 python critical.0.0.1.py \
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
# 					--filter_bn \
# 					--filter_bias
# done

############################## 200304 ###########################
# 8gpu
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=0,1 python critical.0.0.1.py \
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
# 					--filter_bn \
# 					--filter_bias
# done

# # 8gpu
# for i in 0.0128
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 8192 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--save_dir ./logs/ \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
# 					--filter_bn \
# 					--filter_bias
# done

# # 8gpu
# for i in 0.0128
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 8192 \
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

# # 8gpu
# for i in 0.0128
# do
# 	CUDA_VISIBLE_DEVICES=4,5,6,7 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 8192 \
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







# 8GPU
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.0.0.2.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 32 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 1024 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.4 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--filter_bias_and_bn
# done

# group2
# for i in 0.0001 0.0002 0.0004 0.0008 0.0016 0.0032 0.0064 0.0128 0.0256
# do
# 	CUDA_VISIBLE_DEVICES=0 python critical.0.0.1.py \
# 					--dataset cifar10 \
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

############################## 200303 ###########################
# 8gpu
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 32 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 1024 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.4 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--filter_bias_and_bn
# done

# 8gpu
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 32 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 1024 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.4 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 0 \
# 					--filter_bias_and_bn 
# done

# IITP
# for i in 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=2,3 python main.0.0.2.py \
# 					--dataset cifar10 \
# 					--dataroot /home/user/ssd1/dataset/ \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 3.2 \
# 					--momentum 0.9 \
# 					--net_type resnet18 \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
# 					--warm_up_epoch 15 
# done

# 8GPU
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 32 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 1024 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.4 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--filter_bias_and_bn 
# done

# IITP
# for i in 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=2,3 python main.0.0.2.py \
# 					--dataset cifar10 \
# 					--dataroot /home/user/ssd1/dataset/ \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 3.2 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
# 					--warm_up_epoch 15 
# done

# 8GPU
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 32 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 512 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.2 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--filter_bias_and_bn 
# done

# 8gpu-o
# for i in 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=6,7 python main.0.0.2.py \
# 					--dataset cifar10 \
# 					--dataroot /home/user/ssd1/dataset/ \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 3.2 \
# 					--momentum 0.9 \
# 					--net_type resnet18_GBN \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--seed 0 1 2 3 4 \
# 					--zero_init_residual \
# 					--warm_up_epoch 15 
# done

############################## 200302 ###########################
# 8GPU
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 32 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 512 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.2 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--filter_bias_and_bn \
# 					--amp
# done

# 8GPU
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 32 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 1024 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.4 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5
# done

# 8GPU
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 32 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 1024 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.4 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50 \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--filter_bias_and_bn
# done

############################## 200301 ###########################
# 8GPU
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 32 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 1024 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.4 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50 \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--filter_bias_and_bn
# done

# IITP
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0,1 python main.0.0.2.py \
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

# 8gpu-b
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0 python critical.0.0.1.py \
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

# group2
# for i in 0.0001 0.0002 0.0004 0.0008 0.0016 0.0032 0.0064 0.0128 0.0256
# do
# 	CUDA_VISIBLE_DEVICES=0,1 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 2048 \
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

# 8GPU
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.0.0.2.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 32 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 1024 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.4 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--filter_bias_and_bn
# done

############################## 200228 ###########################
# 8gpu-o
# for i in 0.0004 0.0002
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 python critical.0.0.1.py \
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

# 8GPU
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.0.0.2.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 32 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 1024 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.4 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--filter_bias_and_bn
# done

# # 8gpu-h
# for i in 0.0008 0.0004
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--dataroot /home/user/ssd1/dataset/ \
# 					--epochs 300 \
# 					--batch_size 8192 \
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
# 	CUDA_VISIBLE_DEVICES=0 python critical.0.0.1.py \
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

# 8GPU-o
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.0.0.2.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 32 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 1024 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.4 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--filter_bias_and_bn
# done

# 8GPU
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.0.0.2.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/data/ILSVRC2012/ \
# 					--num_workers 32 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 1024 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 1 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 0 \
# 					--filter_bias_and_bn
# done

# 8GPU-o
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.0.0.2.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 32 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 1024 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.4 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--filter_bias_and_bn
# done

# # 8gpu-h
# for i in 0.0256
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--dataroot /home/user/ssd1/dataset/ \
# 					--epochs 300 \
# 					--batch_size 8192 \
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

# 8GPU
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.0.0.2.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/data/ILSVRC2012/ \
# 					--num_workers 32 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 1024 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.4 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--filter_bias_and_bn
# done

# 8gpu-b
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0 python critical.0.0.1.py \
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
# 8GPU
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.0.0.2.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/data/ILSVRC2012/ \
# 					--num_workers 32 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 1024 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.4 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50 \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--filter_bias_and_bn
# done

# 8gpu-b
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0 python critical.0.0.1.py \
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
# for i in 0.0256 0.0128 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 python critical.0.0.1.py \
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

# # # 8gpu-h
# for i in 0.0032 0.0016
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--dataroot /home/user/ssd1/dataset/ \
# 					--epochs 300 \
# 					--batch_size 8192 \
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

# 8GPU-o
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0 python main.0.0.2.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/dataset/ILSVRC2012/ \
# 					--num_workers 32 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.4 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--filter_bias_and_bn
# done

# 8GPU
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.0.0.2.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/data/ILSVRC2012/ \
# 					--num_workers 32 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 1024 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.4 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--filter_bias_and_bn
# done

# # 8gpu-j
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 8192 \
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
# # 8gpu-j
# for i in 0.0128
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 8192 \
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

# # group4
# for i in 0.0001 0.0002 0.0004 0.0008 0.0016 0.0032 0.0064 0.0128
# do
# 	CUDA_VISIBLE_DEVICES=0,1 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 2048 \
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

# group4
# for i in 0.0001 0.0002 0.0004 0.0008 0.0016 0.0032 0.0064 0.0128
# do
# 	CUDA_VISIBLE_DEVICES=0 python critical.0.0.1.py \
# 					--dataset cifar10 \
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

# IITP
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0,1 python main.0.0.2.py \
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

# 8GPU
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.0.0.2.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/data/ILSVRC2012/ \
# 					--num_workers 32 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 1024 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.4 \
# 					--nesterov \
# 					--momentum 0.9 \
# 					--net_type resnet50_GBN \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--seed 0 \
# 					--zero_init_residual \
# 					--warm_up_epoch 5 \
# 					--filter_bias_and_bn
# done


############################## 200225 ###########################
# group2
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0,1 python critical.0.0.1.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/data/ILSVRC2012/ \
# 					--num_workers 8 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 1024 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--nestrov \
# 					--momentum 0.9 \
# 					--net_type resnet50 \
# 					--save_model \
# 					--save_dir ./logs/ \
# 					--seed 0 \
# 					--zero_init_residual
# done


# 8GPU
# for i in 0.0004
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python critical.0.0.1.py \
# 					--dataset imagenet \
# 					--dataroot /home/user/ssd1/data/ILSVRC2012/ \
# 					--num_workers 32 \
# 					--epochs 90 \
# 					--epoch_step 30 60 80 \
# 					--batch_size 1024 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--nestrov \
# 					--momentum 0.9 \
# 					--net_type resnet50 \
# 					--save_model \
# 					--save_dir ./logs/ \
# 					--seed 0 \
# 					--zero_init_residual
# done


############################## 200224 ###########################
# IITP
# for i in 1.6 0.8
# do
# 	CUDA_VISIBLE_DEVICES=0 python main.0.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 2048 \
# 					--weight_decay 0.0002 \
# 					--wd_off_epoch 300 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18 \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--seed 0 \
# 					--warm_up_epoch 15 \
# 					--zero_init_residual
# done

# IITP
# for i in 1.6 0.8
# do
# 	CUDA_VISIBLE_DEVICES=0 python main.0.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 2048 \
# 					--weight_decay 0.0002 \
# 					--wd_off_epoch 300 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18 \
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
# 	CUDA_VISIBLE_DEVICES=0,1 python main.0.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay 0.0002 \
# 					--wd_off_epoch 300 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type resnet18 \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--seed 0 \
# 					--warm_up_epoch 15
# done

# IITP
# for i in 3.2 1.6 0.8
# do
# 	CUDA_VISIBLE_DEVICES=0,1 python main.0.0.2.py \
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
# 					--warm_up_epoch 15
# done

############################## 200222 ###########################
# IITP
# for i in 15 30 5 
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 python main.0.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay 0.0001 \
# 					--wd_off_epoch 300 \
# 					--lr 6.4 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--seed 0 1 2 \
# 					--warm_up_epoch $i
# done

# 8GPU
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100_GBN \
# 					--save_model \
# 					--save_dir ./logs/ \
# 					--seed 0 1 2
# done

# 8GPU
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs/ \
# 					--seed 0 1 2 \
# 					--weight_multiplier 1
# done

############################## 200221 ###########################
# 8GPU
# for i in 15 30 5 
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 python main.0.0.2.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay 0.0001 \
# 					--wd_off_epoch 300 \
# 					--lr 6.4 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--seed 0 1 2 \
# 					--warm_up_epoch $i
# done

############################## 200216 ###########################
# 8GPU
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0 python critical.0.0.1.py \
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
############################## 200128 ###########################
# 8GPU
# for i in 6.4
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay 0.0001 \
# 					--wd_off_epoch 300 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--warm_up_epoch 20 \
# 					--filter_bias_and_bn 
# done

############################## 200127 ###########################
# IITP
# for i in 0.2 0.4 0.8
# do
# 	CUDA_VISIBLE_DEVICES=0 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.0001 \
# 					--wd_off_epoch 300 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--warm_up_epoch 5
# done

# 8GPU
# for i in 6.4
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
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
# for i in 0.2 0.4 0.8
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay 0.0001 \
# 					--wd_off_epoch 300 \
# 					--lr $i \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--warm_up_epoch 5
# done

# 8GPU
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0,1 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 1024 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 1.6 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs_lr/ \
# 					--warm_up_epoch 5
# done


############################## 200119 ###########################
# group4
# for i in 0.0256
# do
# 	CUDA_VISIBLE_DEVICES=0 python critical.0.0.1.py \
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


# group4
# for i in 0.0008
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
# 	CUDA_VISIBLE_DEVICES=0 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 3000 \
# 					--epoch_step 1000 2000 \
# 					--batch_size 128 \
# 					--weight_decay $i \
# 					--wd_off_epoch 3000 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs/ \
# 					--seed 0 1 2
# done


# IITP
# for i in 0.0016
# do
# 	CUDA_VISIBLE_DEVICES=0 python critical.0.0.1.py \
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

############################## 200113 ###########################
# 8GPU
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0,1 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 1024 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 1.6 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs/
# done

# IITP
# for i in 0.0016
# do
# 	CUDA_VISIBLE_DEVICES=0 python critical.0.0.1.py \
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

############################## 200112 ###########################
# # 8GPU
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0,1 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 1024 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 1.6 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs/
# done

# group2
# for i in 0.00005
# do
# 	CUDA_VISIBLE_DEVICES=0 python critical.0.0.1.py \
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

# # group2
# for i in 0.00005
# do
# 	CUDA_VISIBLE_DEVICES=0 python critical.0.0.1.py \
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
# 8GPU
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0 python critical.0.0.1.py \
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
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0 python critical.0.0.1.py \
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
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0 python critical.0.0.1.py \
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
# 	CUDA_VISIBLE_DEVICES=0 python critical.0.0.1.py \
# 					--dataset cifar10 \
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

############################## 200109 ###########################
# # 8gpu - old
# for i in 0.00005
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 8192 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs/
# done

# # 8gpu - old
# for i in 0.0128 0.0064 0.0032 0.0016 0.0008 0.0256 0.0004 0.0002 0.0001 0.00005
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python critical.0.0.1.py \
# 					--dataset cifar100 \
# 					--epochs 300 \
# 					--batch_size 8192 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs/
# done

############################## 200109 ###########################
# group4
# for i in 0.00005
# do
# 	CUDA_VISIBLE_DEVICES=0,1 python critical.0.0.1.py \
# 					--dataset cifar10 \
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

# 8GPU
# for i in 0.00005
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python critical.0.0.1.py \
# 					--dataset cifar100 \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs/
# done

# group2
# for i in 0.00005
# do
# 	CUDA_VISIBLE_DEVICES=0,1 python critical.0.0.1.py \
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

# group2
# for i in 0.0128 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=0 python critical.0.0.1.py \
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

# IITP
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=0 python critical.0.0.1.py \
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

# 8gpu - J
# for i in 0.00005
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs/
# done

# group2
# for i in 0.0256
# do
# 	CUDA_VISIBLE_DEVICES=0,1 python critical.0.0.1.py \
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

############################## 200108 ###########################
# 8gpu - old
# for i in 0.0016 0.0008 0.0004 0.0002 0.0001 0.0256
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 8192 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs/
# done

# 8GPU
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0 python critical.0.0.1.py \
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

# 8gpu
# for i in 0.0256 0.0016 0.0008 0.0004 0.0002 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python critical.0.0.1.py \
# 					--dataset cifar100 \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs/
# done

# # 8gpu - jong
# for i in 0.0128
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python critical.0.0.1.py \
# 					--dataset cifar100 \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs/
# done

# # group4
# for i in 0.0008
# do
# 	CUDA_VISIBLE_DEVICES=2,3 python critical.0.0.1.py \
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

############################## 200107 ###########################
# # group2
# for i in 0.0008 0.0004 0.0002 0.0001 0.0256
# do
# 	CUDA_VISIBLE_DEVICES=0,1 python critical.0.0.1.py \
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

# # 8gpu - old
# for i in 0.0064 0.0128
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python critical.0.0.1.py \
# 					--dataset cifar10 \
# 					--epochs 300 \
# 					--batch_size 8192 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs/
# done

# # group2
# for i in 0.0016
# do
# 	CUDA_VISIBLE_DEVICES=0,1 python critical.0.0.1.py \
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

# # # 8gpu - jong
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python critical.0.0.1.py \
# 					--dataset cifar100 \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs/
# done

############################## 200106 ###########################
# # 8gpu - jong
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python critical.0.0.1.py \
# 					--dataset cifar100 \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs/
# done


# # # group4
# for i in 0.0064 0.0128 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=2,3 python critical.0.0.1.py \
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

# # 8gpu - jong
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python critical.0.0.1.py \
# 					--dataset cifar100 \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs/
# done

# # IITP
# for i in 0.0004
# do
# 	CUDA_VISIBLE_DEVICES=0 python critical.0.0.1.py \
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

# # group4
# for i in 0.0256
# do
# 	CUDA_VISIBLE_DEVICES=0 python critical.0.0.1.py \
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

############################## 200104 ###########################
# # 8GPU
# for i in 0.0016 0.0008 0.0004 0.0002 0.0001 0.0128 0.0256
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python critical.0.0.1.py \
# 					--epochs 300 \
# 					--batch_size 4096 \
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
# for i in 0.0001
# do
# 	CUDA_VISIBLE_DEVICES=0 python critical.0.0.1.py \
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

# # 8GPU-Jong
# for i in 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python critical.0.0.1.py \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs/
# done

############################## 200103 ###########################
# # IITP
# for i in 0.0032 0.0128 0.0256
# do
# 	CUDA_VISIBLE_DEVICES=0,1 python critical.0.0.1.py \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs/
# done

# # 8GPU
# for i in 0.0032 0.0064 0.0128 0.0256
# do
# 	CUDA_VISIBLE_DEVICES=0,1 python critical.0.0.1.py \
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
# for i in 0.0016 0.0032
# do
# 	CUDA_VISIBLE_DEVICES=0 python critical.0.0.1.py \
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
# for i in 0.0016 0.0032 0.0064  
# do
# 	CUDA_VISIBLE_DEVICES=0 python critical.0.0.1.py \
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

# # 8GPU-Jong
# for i in 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python critical.0.0.1.py \
# 					--epochs 300 \
# 					--batch_size 4096 \
# 					--weight_decay $i \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--net_type densenetBC100 \
# 					--save_model \
# 					--save_dir ./logs/
# done

############################## 201225 ###########################
# # IITP
# for i in 0.0001 0.0002 0.0004 0.0008 0.0016 0.0032 0.0064
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 python critical.0.0.1.py \
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

# 8GPU - J
# for i in 0.0001 0.0002 
# do
# 	CUDA_VISIBLE_DEVICES=0,1 python critical.0.0.1.py \
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
# 	CUDA_VISIBLE_DEVICES=0 python critical.0.0.1.py \
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
# for i in 0.03
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 python critical.0.0.1.py \
# 					--epochs 300 \
# 					--batch_size 8192 \
# 					--weight_decay 0.0128 \
# 					--wd_off_epoch 300 \
# 					--lr 0.1 \
# 					--momentum 0.9 \
# 					--std_weight 1 \
# 					--norm_grad_ratio $i \
# 					--weight_multiplier 0.1 \
# 					--net_type densnetBC100
# done

# 8GPU
# for i in 300
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python critical.0.0.1.py \
# 					--epochs 300 \
# 					--batch_size 16384 \
# 					--weight_decay 0.0256 \
# 					--wd_off_epoch $i \
# 					--lr 0.1 \
# 					--std_weight 3

# done

# 8GPU
# for i in 300
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 python critical.0.0.1.py \
# 					--epochs 300 \
# 					--batch_size 8192 \
# 					--weight_decay 0.0256 \
# 					--wd_off_epoch $i \
# 					--lr 0.05

# done


############################## 201129 ###########################
# 8GPU
# for i in 500
# do
# 	CUDA_VISIBLE_DEVICES=0 python critical.0.0.1.py \
# 					--epochs 500 \
# 					--epoch_step 500 \
# 					--batch_size 128 \
# 					--weight_decay 0.002 \
# 					--wd_off_epoch $i \
# 					--seed 0 1 2 3 4
# done


############################## 201125 ###########################
# 8GPU
# for i in 0 200
# do
# 	CUDA_VISIBLE_DEVICES=0,1,2,3 python critical.0.0.1.py \
# 					--epochs 300 \
# 					--batch_size 8192 \
# 					--weight_decay 0.0128 \
# 					--wd_off_epoch $i
# done


############################## 201124 ###########################
# IITP
# for i in 0 100 200 300
# do
# 	CUDA_VISIBLE_DEVICES=0 python critical.0.0.1.py \
# 					--epochs 300 \
# 					--batch_size 1024 \
# 					--weight_decay 0.0005 \
# 					--wd_off_epoch $i
# done


# 8GPU
# # for i in $(seq 0 25 100)
# for i in 200
# do
# 	CUDA_VISIBLE_DEVICES=0 python critical.0.0.1.py \
# 					--epochs 300 \
# 					--batch_size 128 \
# 					--weight_decay 0.0005 \
# 					--wd_off_epoch $i
# done
