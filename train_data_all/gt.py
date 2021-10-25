
import os
import cv2
from tqdm import tqdm
import json
from natsort import natsorted
import numpy as np


def crop_img():
    gt_root = '/root/Storage/datasets/TBrain/train/train_gts/'
    img_root = '/root/Storage/datasets/TBrain/train/train_images/'

    index = 0

    for file in tqdm(natsorted(os.listdir(gt_root))):

        img = cv2.imread(img_root + file.replace('.txt', ''))

        with open(gt_root + file) as gt_file:
            for line in gt_file.readlines():

                label = line.strip().split(',')[8:][0]
                if '###' in label:
                    label = label.replace('###', '')

                line_split = list(map(int, line.strip().split(',')[:8]))

                points1 = np.float32(points2order([[line_split[0],line_split[1]],[line_split[2],line_split[3]],[line_split[4],line_split[5]],[line_split[6],line_split[7]]]))
                
                img_height, img_width = max(points1[:,1]) - min(points1[:,1]), max(points1[:,0]) - min(points1[:,0])
                
                points2 = np.float32([[0,0], [0,img_height], [img_width,0], [img_width,img_height]])

                # cv2.circle(img, (points1[0][0], points1[0][1]), 5, (255, 0, 0), -1) # left up
                # cv2.circle(img, (points1[1][0], points1[1][1]), 5, (0, 0, 255), -1) # left down
                # cv2.circle(img, (points1[2][0], points1[2][1]), 5, (0, 255, 0), -1) # right up
                # cv2.circle(img, (points1[3][0], points1[3][1]), 5, (255, 255, 255), -1) # right down

                M = cv2.getPerspectiveTransform(points1, points2)
                processed_img = cv2.warpPerspective(img, M, (img_width, img_height))
                
                if img_width < img_height:

                    processed_img = np.rot90(processed_img)

                    cv2.imwrite('/root/Storage/PaddleOCR/train_data_all/rec/train/img_' + str(index) + '.jpg', processed_img)

                    with open('/root/Storage/PaddleOCR/train_data_all/rec/rec_gt_train.txt', 'a') as txt_file:

                        txt_file.write('/root/Storage/PaddleOCR/train_data_all/rec/train/img_' + str(index) + '.jpg\t' + label + '\n')

                    index += 1
                
                else:

                    cv2.imwrite('/root/Storage/PaddleOCR/train_data_all/rec/train/img_' + str(index) + '.jpg', processed_img)

                    with open('/root/Storage/PaddleOCR/train_data_all/rec/rec_gt_train.txt', 'a') as txt_file:

                        txt_file.write('/root/Storage/PaddleOCR/train_data_all/rec/train/img_' + str(index) + '.jpg\t' + label + '\n')

                    index += 1

        # break

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




def json2txt():

    json_root = '/root/Storage/datasets/TBrain/train/json/'
    gt_root = '/root/Storage/datasets/TBrain/train/train_gts/'


    for file in natsorted(os.listdir(json_root)):

        print(file)

        json_file = json.load(open(json_root + file, 'r'))

        with open(gt_root + file.replace('.json', '.jpg.txt'), 'w') as gt_file:

            for gt in json_file['shapes']:
                if gt['group_id'] == 0:
                    flat_point = [str(x) for sublist in gt['points'] for x in sublist]
                    gt_file.write(','.join(flat_point) + ',' + gt['label'] + '\n')



if __name__ == '__main__':
    json2txt()
    crop_img()