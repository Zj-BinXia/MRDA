## noise-free degradations with isotropic Gaussian blurs
# training knowledge distillation
CUDA_VISIBLE_DEVICES=2 python3 main_stage4.py --dir_data='/root/datasets' \
               --model='blindsr' \
               --scale='4' \
               --blur_type='iso_gaussian' \
               --noise=0.0 \
               --sig_min=0.2 \
               --sig_max=4.0 \
               --sig 2.6 \
               --n_GPUs=1 \
               --epochs_encoder 100 \
               --epochs_sr 500 \
               --data_test Set5 \
               --st_save_epoch 480 \
               --n_feats 128 \
               --patch_size 64 \
               --task_iter 5 \
               --test_iter 5 \
               --meta_batch_size 5 \
               --batch_size 16 \
               --lr_sr 1e-4 \
               --lr_task 1e-2 \
               --lr_encoder 1e-4 \
               --pre_train_TA="./experiment/stage3.pt" \
               --pre_train_ST="./experiment/stage3.pt" \
               --pre_train_meta="./experiment/stage2.pt" \
               --resume 0 \
               --test_every 800 \
               --print_every 250 \
               --lr_decay_sr 125 \
               --data_train DF2K \
               --save stage4


#anisotropic Gaussian kenrnel + noise
# CUDA_VISIBLE_DEVICES=0 python3 main_stage4.py --dir_data='/root/datasets' \
#                --model='blindsr' \
#                --scale='4' \
#                --blur_type='aniso_gaussian' \
#                --noise=25.0 \
#                --lambda_min=0.2 \
#                --lambda_max=4.0 \
#                --n_GPUs=1 \
#                --epochs_encoder 100 \
#                --epochs_sr 500 \
#                --data_test Set5 \
#                --st_save_epoch 480 \
#                --n_feats 128 \
#                --patch_size 64 \
#                --task_iter 5 \
#                --test_iter 5 \
#                --meta_batch_size 5 \
#                --batch_size 16 \
#                --lr_sr 1e-4 \
#                --lr_task 1e-2 \
#                --lr_encoder 1e-4 \
#                --pre_train_TA="./experiment/stage3.pt" \
#                --pre_train_ST="./experiment/stage3.pt" \
#                --pre_train_meta="./experiment/stage2.pt" \
#                --resume 0 \
#                --test_every 800 \
#                --print_every 250 \
#                --lr_decay_sr 125 \
#                --data_train DF2K \
#                --save stage4