from natsort.natsort import natsorted
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


import cv2
import numpy as np
import math
import time
import traceback
import paddle

import tools.infer.utility as utility
from ppocr.postprocess import build_post_process
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import get_image_file_list, check_and_read_gif
from tools.infer.predict_det import TextDetector

import natsort
from tqdm import tqdm
from paddleocr import PaddleOCR, draw_ocr

ocr_ch = PaddleOCR(use_angle_cls=False, lang="ch", det=False)
ocr_cht = PaddleOCR(use_angle_cls=False, lang="chinese_cht", det=False)
ocr_det = PaddleOCR(use_angle_cls=False, rec=False)

from opencc import OpenCC



logger = get_logger()


# ignore txt

ignore_char = []

ignore = open('./ignore_char.txt', 'r').readlines()

for char in ignore:
    if char not in ignore_char:
        ignore_char.append(char.strip())


class TextRecognizer(object):
    def __init__(self, args):
        self.rec_image_shape = [int(v) for v in args.rec_image_shape.split(",")]
        self.character_type = args.rec_char_type
        self.rec_batch_num = args.rec_batch_num
        self.rec_algorithm = args.rec_algorithm
        postprocess_params = {
            'name': 'CTCLabelDecode',
            "character_type": args.rec_char_type,
            "character_dict_path": args.rec_char_dict_path,
            "use_space_char": args.use_space_char
        }
        if self.rec_algorithm == "SRN":
            postprocess_params = {
                'name': 'SRNLabelDecode',
                "character_type": args.rec_char_type,
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char
            }
        elif self.rec_algorithm == "RARE":
            postprocess_params = {
                'name': 'AttnLabelDecode',
                "character_type": args.rec_char_type,
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char
            }
        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.input_tensor, self.output_tensors, self.config = \
            utility.create_predictor(args, 'rec', logger)
        self.benchmark = args.benchmark
        if args.benchmark:
            import auto_log
            pid = os.getpid()
            self.autolog = auto_log.AutoLogger(
                model_name="rec",
                model_precision=args.precision,
                batch_size=args.rec_batch_num,
                data_shape="dynamic",
                save_path=args.save_log_path,
                inference_config=self.config,
                pids=pid,
                process_name=None,
                gpu_ids=0 if args.use_gpu else None,
                time_keys=[
                    'preprocess_time', 'inference_time', 'postprocess_time'
                ],
                warmup=10)

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        assert imgC == img.shape[2]
        if self.character_type == "ch":
            imgW = int((32 * max_wh_ratio))
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def resize_norm_img_srn(self, img, image_shape):
        imgC, imgH, imgW = image_shape

        img_black = np.zeros((imgH, imgW))
        im_hei = img.shape[0]
        im_wid = img.shape[1]

        if im_wid <= im_hei * 1:
            img_new = cv2.resize(img, (imgH * 1, imgH))
        elif im_wid <= im_hei * 2:
            img_new = cv2.resize(img, (imgH * 2, imgH))
        elif im_wid <= im_hei * 3:
            img_new = cv2.resize(img, (imgH * 3, imgH))
        else:
            img_new = cv2.resize(img, (imgW, imgH))

        img_np = np.asarray(img_new)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        img_black[:, 0:img_np.shape[1]] = img_np
        img_black = img_black[:, :, np.newaxis]

        row, col, c = img_black.shape
        c = 1

        return np.reshape(img_black, (c, row, col)).astype(np.float32)

    def srn_other_inputs(self, image_shape, num_heads, max_text_length):

        imgC, imgH, imgW = image_shape
        feature_dim = int((imgH / 8) * (imgW / 8))

        encoder_word_pos = np.array(range(0, feature_dim)).reshape(
            (feature_dim, 1)).astype('int64')
        gsrm_word_pos = np.array(range(0, max_text_length)).reshape(
            (max_text_length, 1)).astype('int64')

        gsrm_attn_bias_data = np.ones((1, max_text_length, max_text_length))
        gsrm_slf_attn_bias1 = np.triu(gsrm_attn_bias_data, 1).reshape(
            [-1, 1, max_text_length, max_text_length])
        gsrm_slf_attn_bias1 = np.tile(
            gsrm_slf_attn_bias1,
            [1, num_heads, 1, 1]).astype('float32') * [-1e9]

        gsrm_slf_attn_bias2 = np.tril(gsrm_attn_bias_data, -1).reshape(
            [-1, 1, max_text_length, max_text_length])
        gsrm_slf_attn_bias2 = np.tile(
            gsrm_slf_attn_bias2,
            [1, num_heads, 1, 1]).astype('float32') * [-1e9]

        encoder_word_pos = encoder_word_pos[np.newaxis, :]
        gsrm_word_pos = gsrm_word_pos[np.newaxis, :]

        return [
            encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1,
            gsrm_slf_attn_bias2
        ]

    def process_image_srn(self, img, image_shape, num_heads, max_text_length):
        norm_img = self.resize_norm_img_srn(img, image_shape)
        norm_img = norm_img[np.newaxis, :]

        [encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1, gsrm_slf_attn_bias2] = \
            self.srn_other_inputs(image_shape, num_heads, max_text_length)

        gsrm_slf_attn_bias1 = gsrm_slf_attn_bias1.astype(np.float32)
        gsrm_slf_attn_bias2 = gsrm_slf_attn_bias2.astype(np.float32)
        encoder_word_pos = encoder_word_pos.astype(np.int64)
        gsrm_word_pos = gsrm_word_pos.astype(np.int64)

        return (norm_img, encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1,
                gsrm_slf_attn_bias2)

    def __call__(self, img_list):
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))
        rec_res = [['', 0.0]] * img_num
        batch_num = self.rec_batch_num
        st = time.time()
        if self.benchmark:
            self.autolog.times.start()
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                if self.rec_algorithm != "SRN":
                    norm_img = self.resize_norm_img(img_list[indices[ino]],
                                                    max_wh_ratio)
                    norm_img = norm_img[np.newaxis, :]
                    norm_img_batch.append(norm_img)
                else:
                    norm_img = self.process_image_srn(
                        img_list[indices[ino]], self.rec_image_shape, 8, 25)
                    encoder_word_pos_list = []
                    gsrm_word_pos_list = []
                    gsrm_slf_attn_bias1_list = []
                    gsrm_slf_attn_bias2_list = []
                    encoder_word_pos_list.append(norm_img[1])
                    gsrm_word_pos_list.append(norm_img[2])
                    gsrm_slf_attn_bias1_list.append(norm_img[3])
                    gsrm_slf_attn_bias2_list.append(norm_img[4])
                    norm_img_batch.append(norm_img[0])
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()
            if self.benchmark:
                self.autolog.times.stamp()

            if self.rec_algorithm == "SRN":
                encoder_word_pos_list = np.concatenate(encoder_word_pos_list)
                gsrm_word_pos_list = np.concatenate(gsrm_word_pos_list)
                gsrm_slf_attn_bias1_list = np.concatenate(
                    gsrm_slf_attn_bias1_list)
                gsrm_slf_attn_bias2_list = np.concatenate(
                    gsrm_slf_attn_bias2_list)

                inputs = [
                    norm_img_batch,
                    encoder_word_pos_list,
                    gsrm_word_pos_list,
                    gsrm_slf_attn_bias1_list,
                    gsrm_slf_attn_bias2_list,
                ]
                input_names = self.predictor.get_input_names()
                for i in range(len(input_names)):
                    input_tensor = self.predictor.get_input_handle(input_names[
                        i])
                    input_tensor.copy_from_cpu(inputs[i])
                self.predictor.run()
                outputs = []
                for output_tensor in self.output_tensors:
                    output = output_tensor.copy_to_cpu()
                    outputs.append(output)
                if self.benchmark:
                    self.autolog.times.stamp()
                preds = {"predict": outputs[2]}
            else:
                self.input_tensor.copy_from_cpu(norm_img_batch)
                self.predictor.run()

                outputs = []
                for output_tensor in self.output_tensors:
                    output = output_tensor.copy_to_cpu()
                    outputs.append(output)
                if self.benchmark:
                    self.autolog.times.stamp()
                preds = outputs[0]
            rec_result = self.postprocess_op(preds)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]
            if self.benchmark:
                self.autolog.times.end(stamp=True)
        return rec_res, time.time() - st


def main(args):

    cc = OpenCC('s2tw')

    # TBrain_root = '/root/Storage/datasets/TBrain/public/img_public/'
    # csv_root = '/root/Storage/datasets/TBrain/public/Task2_Public_String_Coordinate.csv'
    TBrain_root = '/root/Storage/datasets/TBrain/private/img_private/'
    csv_root = '/root/Storage/datasets/TBrain/private/Task2_Private_String_Coordinate.csv'

    # text detector

    det_model_dir = "/root/Storage/PaddleOCR/inference/ch_ppocr_server_v2.0_det_infer"

    args.det_model_dir = det_model_dir

    text_detector = TextDetector(args)

    if args.warmup:
        img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
        for i in range(10):
            res = text_detector(img)

    # server recognition model

    rec_model_dir = "/root/Storage/PaddleOCR/inference/ch_ppocr_server_v2.0_rec_infer"

    args.rec_model_dir = rec_model_dir
    args.max_text_length = 25
    args.rec_image_shape = "3, 32, 320"

    text_recognizer_server = TextRecognizer(args)

    # pre recognition model

    rec_model_dir = "/root/Storage/PaddleOCR/inference/ch_infer_server_pre"

    args.rec_model_dir = rec_model_dir
    args.max_text_length = 25
    args.rec_image_shape = "3, 32, 320"

    text_recognizer_pre= TextRecognizer(args)

    # horizontal recognition model

    rec_model_dir = "/root/Storage/PaddleOCR/inference/ch_infer_all"

    args.rec_model_dir = rec_model_dir
    args.max_text_length = 25
    args.rec_image_shape = "3, 32, 320"

    text_recognizer_hor = TextRecognizer(args)
  
    # vertical recognition model

    rec_model_dir = "/root/Storage/PaddleOCR/inference/ch_infer_all"

    args.rec_model_dir = rec_model_dir
    args.max_text_length = 25
    args.rec_image_shape = "3, 32, 320"

    text_recognizer_ver = TextRecognizer(args)

    # chinese cht recognition model

    rec_model_dir = "/root/Storage/PaddleOCR/inference/chinese_cht_mobile_v2.0_rec_infer"

    args.rec_model_dir = rec_model_dir
    args.rec_char_dict_path = '/root/Storage/PaddleOCR/ppocr/utils/dict/chinese_cht_dict.txt'
    args.rec_char_type = 'chinese_cht'
    args.max_text_length = 25
    args.rec_image_shape = "3, 32, 320"

    text_recognizer_cht= TextRecognizer(args)


    # warmup 10 times
    if args.warmup:
        img = np.random.uniform(0, 255, [32, 320, 3]).astype(np.uint8)
        for i in range(10):
            text_recognizer_hor([img])
            text_recognizer_ver([img])
            text_recognizer_server([img])
            text_recognizer_pre([img])
            text_recognizer_cht([img])

    # result file

    result_file = open('result.csv', 'w')

    # temp file

    # temp_dir = natsorted(os.listdir('./temp_929/'))

    # img_have_word =[]

    # for file in temp_dir:
    #     img_num = file.split('_')[0]
    #     if img_num not in img_have_word:
    #         img_have_word.append(img_num)

    with open(csv_root, 'r') as csv_file:
            

            for i, line in enumerate(tqdm(csv_file.readlines())):

                if i == 236:
                    print('0')

                line_split = line.strip().split(',')

                img_name = line_split[0]

                img = cv2.imread(TBrain_root + img_name + '.jpg')

                line_split = list(map(int, line_split[1:]))

                points1 = np.float32(points2order([[line_split[0],line_split[1]],[line_split[2],line_split[3]],[line_split[4],line_split[5]],[line_split[6],line_split[7]]]))
                
                img_height, img_width = max(points1[:,1]) - min(points1[:,1]), max(points1[:,0]) - min(points1[:,0])
                
                points2 = np.float32([[0,0], [0,img_height], [img_width,0], [img_width,img_height]])

                M = cv2.getPerspectiveTransform(points1, points2)
                processed_img = cv2.warpPerspective(img, M, (img_width, img_height))

                cv2.imwrite('processed_img.jpg', processed_img)
                cv2.imwrite('ori_img.jpg', img)

                result_det1 = ocr_det.ocr(processed_img)
                result_det2, _ = text_detector(processed_img)
                # print(result_det)

                # if str(i+1) in img_have_word:
                if result_det1 != [] or result_det2 != []:

                    # horizontal
                    if img_width >= img_height:
                    # if True:
                        
                        result1 = ocr_ch.ocr(processed_img)

                        if result1 != []:
                            result1 = [result1[0][1]]
                        else:
                            result1 = [['', 0]]

                        result2 = ocr_cht.ocr(processed_img)

                        if result2 != []:
                            result2 = [result2[0][1]]
                        else:
                            result2 = [['', 0]]

                        result3, _ = text_recognizer_server([processed_img])

                        result4, _ = text_recognizer_hor([processed_img])

                        result5, _ = text_recognizer_pre([processed_img])

                        result6, _ = text_recognizer_cht([processed_img])

                        result = list_post_process([result1, result2, result3, result4, result5, result6])
                        # result = list_post_process([result3])



                    # vertical
                    else:
                        processed_img = np.rot90(processed_img)
                        
                        result1 = ocr_ch.ocr(processed_img)

                        if result1 != []:
                            result1 = [result1[0][1]]
                        else:
                            result1 = [['', 0]]

                        result2 = ocr_cht.ocr(processed_img)

                        if result2 != []:
                            result2 = [result2[0][1]]
                        else:
                            result2 = [['', 0]]

                        result3, _ = text_recognizer_server([processed_img])

                        result5, _ = text_recognizer_pre([processed_img])

                        result6, _ = text_recognizer_cht([processed_img])

                        result4, _ = text_recognizer_ver([processed_img])

                        result = list_post_process([result1, result2, result3, result4, result5, result6])
                        result = result
                        # result = list_post_process([result3])


                    # result = cc.convert(result3[0][0])

                    # ans = ""
                    # for s in result:
                    #     if s in ignore_char:
                    #         continue
                    #     else:
                    #         ans += s
                        
                    # if result == "" or result3[0][1] < 0.4:
                    #     result = "###"

                    result_file.write(line.strip() + ',' + result + '\n')

                else:
                    result_file.write(line.strip() + ',' + '###' + '\n')
                
                result_file.flush()

def list_post_process(input_list):

    cc = OpenCC('s2tw')

    temp_list = []
    final_result = ''

    max_len = 0

    sim = ['台','庄','里','斗','托']

    len_dict = {}

    for i in input_list:

        word = ''
        for s in i[0][0]:
            if s in ignore_char:
                continue
            elif s in sim:
                word += s
            else:
                word += cc.convert(s)
        if i[0][1] < 0.0: # threshold
            temp_list.append((word, '0'))
        else:
            temp_list.append((word, i[0][1]))

        if len(word) > 0:
            if len(word) not in len_dict:
                len_dict[len(word)] = 1
            else:
                len_dict[len(word)] += 1

        # if len(word) > max_len:
        #     max_len = len(word)

    if len(len_dict) != 0:
        max_len = max(len_dict, key=len_dict.get)
    else:
        max_len = 0

    temp_list = np.array(temp_list)

    # print(temp_list)

    for word_num in range(0, max_len):

        char_list = []

        for item in temp_list:
            if len(item[0]) == max_len:
                if word_num+1 <= len(item[0]):
                    char_list.append([item[0][word_num], item[1]])

        # print(char_list)
        score_dict = {}

        for char in char_list:
            if char[0] not in score_dict:
                score_dict[char[0]] = float(char[1])
            else:
                score_dict[char[0]] += float(char[1])

        # print(score_dict)
        # print(max(score_dict, key=score_dict.get))

        if len(score_dict) != 0:
            final_result += max(score_dict, key=score_dict.get)


    if final_result == '':
        final_result = '###'

    return final_result



def points2order(points):

    ans = []
    center = [0,0]

    points = sorted(points, key = lambda x : x[0] + x[1])

    for x,y in points:
        center[0] += x/4
        center[1] += y/4
    center[0] = round(center[0])
    center[1] = round(center[1])

    ans.append(points[0])
    ans.append(points[1] if points[1][0] < center[0] else points[2])
    ans.append(points[1] if points[1][0] > center[0] else points[2])
    ans.append(points[3])

    return ans


if __name__ == "__main__":
    main(utility.parse_args())
