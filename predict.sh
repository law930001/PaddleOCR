# python tools/infer/predict_system.py \
# --image_dir="/root/Storage/datasets/TBrain/train/train_images/img_1.jpg" \
# --det_model_dir="./inference/ch_ppocr_server_v2.0_det_infer/"  \
# --rec_model_dir="./inference/ch_ppocr_server_v2.0_rec_infer/" \
# --cls_model_dir="./inference/ch_ppocr_mobile_v2.0_cls_infer/" \
# --use_angle_cls=True --use_space_char=True


# python tools/infer/predict_rec.py \
# --image_dir="/root/Storage/datasets/TBrain/train/train_images/img_1.jpg" \
# --rec_model_dir="./inference/ch_ppocr_server_v2.0_rec_infer/"


CUDA_VISIBLE_DEVICES=1 python predict_TBrain.py \
--rec_char_type="ch" \
--rec_char_dict_path="/root/Storage/PaddleOCR/ppocr/utils/ppocr_keys_v1.txt"
