# 資料處理
請執行train_data_all/gt.py產生訓練資料
gt.py中有兩個function
第一個是json2txt，請根據裡面的路徑設定為自己的路徑並將json轉為txt檔案
第二個是crop_img，請根據裡面的路徑設定為自己的路徑並將img crop出來並產生訓練txt檔案

# 訓練流程
可以修改configs/rec/rec_TBrain_train_all.yml參數
訓練請執行train.sh，其中可以更改cuda所使用的顯示卡是第幾張

# 預測方法
請在訓練完之後，執行export.sh，會將訓練model轉換為infer model
最後執行predict.sh預測結果會生出result.txt檔案

# 除錯
1. 請確認所有訓練檔案及預訓練模型的路徑皆正確
預訓練模型的路徑可以在predict_TBrain.py中修改
訓練檔案生成的路徑可以在train_dall_all/gt.py中修改
整個模型訓練資料可以在configs/rec/rec_TBrain_train_all.yml中修改
2. 請執行pip install requirements.txt安裝需要的library
3. 請確定下載的預訓練模型皆為infer model，若為training model請透過export.sh中的指令修改後執行
此處有問題請參閱 https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.3/doc/doc_ch/recognition.md
4. 有任何問題請參閱 https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.3/README_ch.md
5. 此文件已放在GitHub中 https://github.com/law930001/PaddleOCR

