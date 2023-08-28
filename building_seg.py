import numpy as np
from PIL import Image

def rle_to_mask(rle, shape):
    """
    RLE 형식의 마스크를 이미지로 변환하는 함수
    :param rle: RLE 형식의 마스크
    :param shape: 이미지의 형태 (높이, 너비)
    :return: 변환된 이미지
    """
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    rle_list = rle.split()
    starts = rle_list[::2]
    lengths = rle_list[1::2]

    for start, length in zip(starts, lengths):
        start = int(start) - 1
        length = int(length)
        mask[start:start + length] = 255

    mask = mask.reshape((shape[0], shape[1]))
    #픽셀값 범위조정
    mask_image = Image.fromarray(mask, mode='L')

    return mask_image

import os 
import cv2
from tqdm import tqdm

def csv_to_mask_img(reader,image_path,save_path):
    image_list=os.listdir(image_path)

    cnt=0
    for line in tqdm(reader):
        # 1행은 넘김
        if line[1]=="img_path":
            continue
        else:
            image= cv2.imread(image_path+image_list[cnt], cv2.IMREAD_GRAYSCALE) #불러오기
            rle=line[2]
            h,w=image.shape
            img_shape=(h,w)

            # RLE 형식의 마스크를 이미지로 변환
            mask_image = rle_to_mask(rle, img_shape)
            
            mask_image.save(f"{save_path}{line[0]}.png")

        cnt+=1

# 다각형 좌표로 바운딩 박스 좌표 계산하는 함수
def calculate_bounding_box(coordinates):
    x_values = [coord[0] for coord in coordinates]
    y_values = [coord[1] for coord in coordinates]
    
    min_x = min(x_values)
    min_y = min(y_values)
    max_x = max(x_values)
    max_y = max(y_values)
    
    width = max_x - min_x
    height = max_y - min_y
    
    return min_x, min_y, width, height

import base64
import json

def csv_to_json(reader, image_path,json_path): 
    new_data={}

    image_list=os.listdir(image_path)
    cnt=0
    for line in tqdm(reader): # csv 한 행마다
        # print("line:",line)
        
        # 1행은 넘김
        if line[1]=="img_path":
            continue
 
        shape_type="polygon"
        label="building"
        shapes_list = []

        rle_path=image_path+image_list[cnt]
        rle_path=rle_path.replace("train_img","rle_img")
        # print(rle_path)
        image= cv2.imread(rle_path, cv2.IMREAD_GRAYSCALE) #불러오기

        # 윤곽선 찾기
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # # 다각형 객체 개수 출력
        # num_polygons = len(contours)
        # # print(f"다각형 객체 개수: {num_polygons}")

        for i, contour in enumerate(contours):
            # print(f"윤곽선 {i+1}의 좌표:")
            points=[]
            for point in contour:
                x, y = point[0]
                points.append([x.item(),y.item()])
            
            # print("points:",points)
            x, y, width, height = calculate_bounding_box(points)
            bbox=[x,y,width,height]

            shape_data={
                'points':points,
                'label':label,
                'bbox' : bbox,
                'shape_type': shape_type
            }

            shapes_list.append(shape_data)
            # print("shapes_list:",shapes_list)

        imagePath=line[0]+".png"
        # print(imagePath)

        # print(image_path+imagePath)
        # 이미지 데이터를 Base64로 인코딩
        with open(image_path+imagePath, "rb") as f:
            image_data = f.read()
            encoded_image_data = base64.b64encode(image_data).decode("utf-8")
 
        with Image.open(image_path+imagePath) as img:
            width,height=img.size
 
        new_data = {
        'shapes': shapes_list,
        'imagePath': imagePath,
        'imageData':encoded_image_data,
        'imageHeight' : height,
        'imageWidth' : width
        }

        json_file=line[0]+".json"
        # print(json_file)

        # 수정된 JSON 파일 저장
        output_json_file_path =json_path+json_file
        # print(output_json_file_path)
        with open(output_json_file_path, "w") as f:
            json.dump(new_data, f)

        cnt+=1

def json_to_img(json_path,image_path):
    for j in os.listdir(json_path):
        print(j)
        json_file=os.path.join(json_path,j)
        with open(json_file,'r') as file:
            data=json.load(file)
        
        # 이미지 경로
        image_file = data['imagePath']

        # 이미지 로드
        image = cv2.imread(image_path+image_file)
        # print(image_path+image_file)

        # 각 도형에 대해 좌표 추출 및 그리기
        shapes = data['shapes']
        for shape in shapes:
            points = shape['points']
            label = shape['label']
            shape_type = shape['shape_type']
            
            if points=="none":
                break

            # 좌표를 NumPy 배열로 변환
            points = np.array(points, np.int32)
            points = points.reshape((-1, 1, 2))
            
            # 각 점에 대해 선으로 연결
            if shape_type == 'polygon':
                cv2.polylines(image, [points], isClosed=True, color=(0, 0, 255), thickness=2)
            
            # 점들을 순서대로 연결하여 선 그리기
            elif shape_type == 'line':
                cv2.polylines(image, [points], isClosed=False, color=(0, 0, 255), thickness=2)
            
            # 점들을 순서대로 연결하여 다각형 그리기
            elif shape_type == 'polyline':
                cv2.polylines(image, [points], isClosed=False, color=(0, 0, 255), thickness=2)

        # 이미지 저장 또는 출력
        cv2.imshow(f"{image_path+j}", image)
    # cv2.waitKey()