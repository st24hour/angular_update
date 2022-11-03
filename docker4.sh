# ICML
# group4
# nvidia-docker run --shm-size=128G --cpus=12 -it --rm -v $(pwd):$(pwd) \
# 	-v /home/user/jusung/:/home/user/jusung/ -w $(pwd) \
# 	st24hour/pytorch:1.6.0_large_batch_v2 \
# 	sh train3.sh

# 8GPU - ECCV
nvidia-docker run --shm-size=1024G --cpus=40 -it --rm --pid=host -v $(pwd):$(pwd) \
	-v /home/siit/ssd1/:/home/user/ -w $(pwd) \
	st24hour/pytorch:1.10.0_angular_update_seg \
	sh train4.sh

# GCP
# docker run --gpus all --shm-size=40g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm --pid=host -v $(pwd):$(pwd) -w $(pwd) \
# 	-v /etc/localtime:/etc/localtime:ro -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
# 	-v /shared_storage:/home/user/ st24hour/pytorch:1.10.0_angular_update \
# 	sh train4.sh

###############################################################################################

# 42, 113
# nvidia-docker run --cpus=2 -it --rm -v $(pwd):$(pwd) -v /home/user/jusung/:/home/user/jusung/ -w $(pwd) \
# 	st24hour/pytorch:0.4.1_js00 \
# 	sh train3.sh \

# group4, group2, hutom, 8gpu
# nvidia-docker run --cpus=2 -it --rm -v $(pwd):$(pwd) -v /home/siit/jusung/:/home/user/jusung/ -w $(pwd) \
# 	st24hour/pytorch:0.4.1_js00 \
# 	sh train3.sh \


# # 176, 121, 8GPU(198)
# nvidia-docker run --shm-size=1024M --cpus=4 -it --rm -v $(pwd):$(pwd) -v /home/siit/jusung/:/home/user/jusung/ -w $(pwd) \
# 	st24hour/pytorch:0.4.1_js00 \
# 	sh train3.sh \

# 8GPU(108)
# docker run --gpus all --shm-size=1024M --cpus=4 -it --rm -v $(pwd):$(pwd) -v /home/siit/jusung/:/home/user/jusung/ -w $(pwd) \
# 	st24hour/pytorch:0.4.1_js00 \
# 	sh train3.sh \




# IITP half precision
# nvidia-docker run --shm-size=1024M --cpus=2 -it --rm -v $(pwd):$(pwd) -v /home/user/jusung/:/home/user/jusung/ -w $(pwd) \
# 	st24hour/pytorch:1.6.0_large_batch \
# 	sh train3.sh


# group4, group2 half precision
# nvidia-docker run --shm-size=1024M --cpus=2 -it --rm -v $(pwd):$(pwd) -v /home/siit/jusung/:/home/user/jusung/ -w $(pwd) \
# 	st24hour/pytorch:1.6.0_large_batch \
# 	sh train4.sh


# 159.108,      157.214, 158.158 (check SSD)
# docker run --gpus all --shm-size=128G --cpus=40 -it --rm -v $(pwd):$(pwd) \
# 	-v /home/siit/jusung/:/home/user/jusung/ -v /home/siit/ssd1/:/home/user/ssd1/ -w $(pwd) \
# 	st24hour/pytorch:1.7.1_large_batch_v1 \
# 	sh train4.sh                                                                                                               

# 159.109(8gpu-h)
# docker run --gpus all --shm-size=128G --cpus=40 -it --rm -v $(pwd):$(pwd) \
# 	-v /home/siit/jusung/:/home/user/jusung/ -v /home/ssd/:/home/user/ssd1/ -w $(pwd) \
# 	st24hour/pytorch:1.7.1_large_batch_v1 \
# 	sh train4.sh

# 157.215
# nvidia-docker run --shm-size=128G --cpus=40 -it --rm -v $(pwd):$(pwd) \
# 	-v /home/siit/jusung/:/home/user/jusung/ -v /home/siit/ssd1/:/home/user/ssd1/ -w $(pwd) \
# 	st24hour/pytorch:1.7.1_large_batch_v1 \
# 	sh train4.sh    

# DGX
# nvidia-docker run --shm-size=512G --cpus=80 -it --rm -v $(pwd):$(pwd) \
# 	-v /home/user/jusung/:/home/user/jusung/ -v /home/user/raid/ILSVRC2012/:/home/user/ssd1/dataset/ILSVRC2012/ -w $(pwd) \
# 	st24hour/pytorch:1.7.1_large_batch_v1 \
# 	sh train4.sh

# 158.68 (8gpu-hyejin)
# nvidia-docker run --shm-size=128G --cpus=40 -it --rm -v $(pwd):$(pwd) \
# 	-v /home/siit/jusung/:/home/user/jusung/ -v /home/siit/ssd1/:/home/user/ssd1/ -w $(pwd) \
# 	st24hour/pytorch:1.7.1_large_batch_v1 \
# 	sh train4.sh