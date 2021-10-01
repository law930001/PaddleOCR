# recommended paddle.__version__ == 2.0.0
# python -m paddle.distributed.launch --gpus '0'  tools/train.py -c configs/rec/rec_TBrain_train.yml
# python tools/train.py -c configs/rec/rec_TBrain_train.yml -o Global.checkpoints=./output/rec_chinese/latest
python tools/train.py -c configs/rec/rec_TBrain_train_hor.yml