CUDA_VISIBLE_DEVICES=1 python tools/export_model.py \
-c configs/rec/rec_TBrain_train_all.yml \
-o Global.pretrained_model=/root/Storage/PaddleOCR/output/rec_chinese_all/best_accuracy  \
Global.save_inference_dir=/root/Storage/PaddleOCR/inference/ch_infer_all/
