scp -P 5003 root@140.115.51.97:~/Storage/PaddleOCR/output/rec_chinese_hor/iter_epoch_1203.pdopt ./inference/ch_infer_hor/best_accuracy.pdopt
scp -P 5003 root@140.115.51.97:~/Storage/PaddleOCR/output/rec_chinese_hor/iter_epoch_1203.pdparams ./inference/ch_infer_hor/best_accuracy.pdparams
scp -P 5003 root@140.115.51.97:~/Storage/PaddleOCR/output/rec_chinese_hor/iter_epoch_1203.states ./inference/ch_infer_hor/best_accuracy.states

CUDA_VISIBLE_DEVICES=1 python tools/export_model.py -c configs/rec/rec_TBrain_train_hor.yml \
-o Global.pretrained_model=/root/Storage/PaddleOCR/inference/ch_infer_hor/best_accuracy  \
Global.save_inference_dir=/root/Storage/PaddleOCR/inference/ch_infer_hor/

CUDA_VISIBLE_DEVICES=1 python tools/export_model.py -c configs/rec/rec_TBrain_train_ver.yml \
-o Global.pretrained_model=/root/Storage/PaddleOCR/output/rec_chinese_ver/best_accuracy  \
Global.save_inference_dir=/root/Storage/PaddleOCR/inference/ch_infer_ver/