 CUDA_LAUNCH_BLOCKING=1 python -u train_ptsn.py \
        --IMG_SIZE 224 \
        --img_root_path /home/jun/DATA1/datasets/mimic_cxr_dsr2/images \
        --backbone_resume_path ./resume_model/swin_base_patch4_window7_224_22k.pth \
        --annotation_folder ./datasets/mimic_cxr \
        --num_gpus 1 \
        --batch_size 16 \

