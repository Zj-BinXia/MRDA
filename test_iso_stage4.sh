CUDA_VISIBLE_DEVICES=1 python3 main_stage4.py --dir_data='/mnt/bn/xiabinsr/datasets' \
               --model='blindsr' \
               --scale='4' \
               --blur_type='iso_gaussian' \
               --noise=0.0 \
               --sig_min=0.2 \
               --sig_max=4.0 \
               --sig 3.6 \
               --save ours_iso_36\
               --n_GPUs=1 \
               --epochs_encoder 100 \
               --epochs_sr 500 \
               --data_test Set14 \
               --st_save_epoch 480 \
               --n_feats 128 \
               --patch_size 48 \
               --task_iter 5 \
               --test_iter 5 \
               --meta_batch_size 5 \
               --batch_size 16 \
               --lr_sr 1e-4 \
               --lr_task 1e-2 \
               --lr_encoder 1e-4 \
               --pre_train_ST="./experiment/iso_ST.pt" \
               --pre_train="./experiment/iso_ST.pt" \
               --save_results False \
               --resume 0 \
               --test_only