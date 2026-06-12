# CHD pretrain
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port 21604 \
train_contrast.py --device cuda:0 \
--model_name UNet2D_MACL --find_unused_parameters \
--dataset chd --batch_size 16 --checkpoint_pretrain_interval 5 --epochs 100 \
--data_dir "datasets/chd/out_unlabeled/" --do_contrast --lr 0.01 \
--experiment_name CHD_pretrain_your_experiment_name_ --save JCL --slice_threshold 0.1 \
--temp 0.1 --patch_size 512 512 --initial_filter_size 32 --classes 512 \
--contrastive_method pcl --GPU_Name '0,1' --scale_factor 0.25 --pixel_use \
--parallel DDP --alpha 0.5 --alpha_ER 1.0 --AMP --n_segments 100 --compactness 10

# BraTS pretrain
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port 21182 \
train_contrast.py --device cuda:0 \
--model_name UNet2D_MACL --find_unused_parameters \
--dataset BraTS --batch_size 32 --checkpoint_pretrain_interval 10 --epochs 100 \
--data_dir "datasets/BraTS_unlabeled/unlabeled" --do_contrast --lr 0.01 \
--experiment_name BraTS_pretrain_your_experiment_name_ --save JCL --slice_threshold 0.05 \
--temp 0.1 --patch_size 192 192 --initial_filter_size 32 --classes 512 \
--contrastive_method pcl --GPU_Name '0,1' --scale_factor 0.25 \
--pixel_use --parallel DDP --alpha 0.5 --alpha_ER 1.0 --AMP \
--mode pretrain --n_segments 100 --compactness 10

# KiTS pretrain
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port 29244 \
train_contrast.py --device cuda:0 \
--model_name UNet2D_MACL --find_unused_parameters \
--dataset KiTS --batch_size 16 --checkpoint_pretrain_interval 10 --epochs 100 \
--data_dir "datasets/KITS" --do_contrast --lr 0.01 \
--experiment_name KiTS_pretrain_your_experiment_name_ --save JCL --slice_threshold 0.1 \
--temp 0.1 --patch_size 512 512 --initial_filter_size 32 --classes 512 \
--contrastive_method pcl --GPU_Name '0,1' --scale_factor 0.25 --pixel_use \
--parallel DDP --alpha 0.5 --alpha_ER 1.0 --AMP \
