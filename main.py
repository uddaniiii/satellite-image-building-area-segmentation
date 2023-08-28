import os
import csv
import building_seg as seg
import cv2

csv_file="./open/train.csv"
# train_csv="train.csv"

train_image_path="./open/train_img/"
rle_image_path="./open/rle_img/"
dacon_rle_image_path="./open/dacon_rle_img/"
json_path="./open/train_json/"

train_test_path="./open/train_test_img/"
rle_test_path="./open/rle_test_img/"
json_test_path="./open/train_test_json/"

reader=[]
# csv 파일 정보 접근
if os.path.isfile(csv_file): # csv 파일 존재할때만 실행
    with open(csv_file, 'r') as f: # csv 열기
        reader = csv.reader(f) # csv 파일 읽기
        reader=list(reader) # csv 파일 list로 변경

    # seg.csv_to_mask_img(reader,train_image_path,dacon_rle_image_path)
    # seg.csv_to_json(reader, train_image_path, json_path)

    seg.json_to_img(json_test_path,train_test_path)
    seg.json_to_img(json_test_path,rle_test_path)
    cv2.waitKey()