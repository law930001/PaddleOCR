import os
from re import I
import cv2
import numpy as np


from natsort import natsorted

temp_root = "/root/Storage/PaddleOCR/temp/"

temp_file = natsorted(os.listdir(temp_root))

index = 1

image_list = []
image_shape_list = []

for file in temp_file:

    while str(index) != file.split('_')[0]:            
        if image_list == []:
            pass
        else:
            ## combine images
            # find max
            
            image_shape_list = np.array(image_shape_list)
            max_width = max(image_shape_list[:, 1:2])
            width_threshold = max_width[0] * 0.75

            save_index = 1
            over = False
            for i in range(len(image_list)):

                # for jump over the combined second image
                if over:
                    over = False
                    continue

                # for the last image
                if i == len(image_list) - 1:
                    cv2.imwrite('temp_929/' + str(index) + '_' + str(save_index) + '.jpg', image_list[i])
                    break

                # combine two image
                if image_list[i].shape[1] < width_threshold and image_list[i+1].shape[1] < width_threshold:
                    h1, w1, _ = image_list[i].shape
                    h2, w2, _ = image_list[i+1].shape
                    mask = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)

                    mask[:h1, :w1, :3] = image_list[i]
                    mask[:h2, w1:w1+w2, :3] = image_list[i+1]

                    cv2.imwrite('temp_929/' + str(index) + '_' + str(save_index) + '.jpg', mask)

                    over = True
                # output the no-need combined image
                else:
                    cv2.imwrite('temp_929/' + str(index) + '_' + str(save_index) + '.jpg', image_list[i])

                save_index += 1

            image_list = []
            image_shape_list = []
        
        index += 1
    
    img = cv2.imread(temp_root + file)
    image_list.append(img)
    image_shape_list.append(img.shape)
    print(file)



