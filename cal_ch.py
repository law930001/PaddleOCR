import random
import numpy as np
import copy

rec_gt_train_file = open('train_data/rec/rec_gt_train_old.txt', 'r').readlines()

dict_file = open('/root/Storage/PaddleOCR/ppocr/utils/dict/chinese_cht_dict.txt', 'r').readlines()
new_dict_file = open('/root/Storage/PaddleOCR/ppocr/utils/dict/chinese_cht_new_dict.txt', 'w')

rec_list = {}

# read training data
for line in rec_gt_train_file:
    line_split = line.strip().split('\t')
    if line_split[1] not in rec_list:
        rec_list[line_split[1]] = []

    rec_list[line_split[1]].append(line_split[0])

print(len(rec_list))

for k, v in rec_list.items():
    new_dict_file.write(k + '\n')

# # cal dict 
# dict = []
# for line in dict_file:
#     dict.append(line.strip())

# err = 0
# for k in rec_list:
#     if k not in dict:
#         err += 1
#         print(k)


# balance training data
# num = 100

# for k, v in rec_list.items():

#     if len(v) > num:
#         rec_list[k] = random.sample(v, num)
#     elif len(v) < num:
#         ori_v = copy.deepcopy(v)
#         while(len(v) < num):
#             rec_list[k].append(random.choice(ori_v))


#     # print(k, len(rec_list[k]))

# # output training data

# with open('train_data/rec/rec_gt_train.txt', 'w') as output_file:

#     for k, v in rec_list.items():

#         for pth in v:
#             output_file.write(pth.strip() + '\t' + k + '\n')
