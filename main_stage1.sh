## noise-free degradations with isotropic Gaussian blurs or anisotropic + noise
# using oracle degradation training
CUDA_VISIBLE_DEVICES=0 python3 main_stage1.py --dir_data='/root/datasets' \
               --model='blindsr' \
               --scale='4' \
               --n_GPUs=1 \
               --epochs_encoder 0 \
               --epochs_sr 600 \
               --data_test Set14 \
               --st_save_epoch 590 \
               --n_feats 128 \
               --batch_size 64 \
               --patch_size 64 \
               --data_train DF2K \
               --save stage1








