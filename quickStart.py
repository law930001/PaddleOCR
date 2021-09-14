from paddleocr import PaddleOCR, draw_ocr

import cv2


ocr = PaddleOCR(use_angle_cls=True, lang="chinese_cht", det=False)
# img_path = '/root/Storage/PaddleOCR/doc/imgs_words/chinese_traditional/chinese_cht_2.png'
# result = ocr.ocr(img_path)
# print(result[0][1][0])


TBrain_root = '/root/Storage/datasets/TBrain/public/img_public/'
csv_root = '/root/Storage/datasets/TBrain/public/Task2_Public_String_Coordinate.csv'


result_file = open('result.csv', 'w')

with open(csv_root, 'r') as csv_file:
        

        for i, line in enumerate(csv_file.readlines()):

            line_split = line.strip().split(',')

            img_name = line_split[0]

            img = cv2.imread(TBrain_root + img_name + '.jpg')

            x_max = max(int(line_split[1]),int(line_split[3]),int(line_split[5]),int(line_split[7]))
            y_max = max(int(line_split[2]),int(line_split[4]),int(line_split[6]),int(line_split[8]))
            x_min = min(int(line_split[1]),int(line_split[3]),int(line_split[5]),int(line_split[7]))
            y_min = min(int(line_split[2]),int(line_split[4]),int(line_split[6]),int(line_split[8]))

            # crop image
            cropped = img[y_min:y_max, x_min:x_max]

            cv2.imwrite('cropped.jpg', cropped)
            cv2.imwrite('img.jpg', img)

            result = ocr.ocr(cropped)

            ans = ''
            if result == []:
                ans = '###'
            else:
                for r in result:
                    ans += r[1][0]

            print(i, ans)

            result_file.write(line.strip() + ',' + ans + '\n')
            result_file.flush()
