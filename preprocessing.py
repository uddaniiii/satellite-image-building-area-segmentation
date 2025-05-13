import numpy as np
from PIL import Image
import os
import cv2
import csv
import argparse
from tqdm import tqdm

def rle_to_mask(rle, shape):
    """
    RLE 형식의 마스크를 이미지로 변환하는 함수.
    
    :param rle: RLE 형식의 마스크
    :param shape: 이미지의 형태 (높이, 너비)
    :return: 변환된 마스크 이미지
    """
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    rle_list = rle.split()
    starts = map(int, rle_list[::2])
    lengths = map(int, rle_list[1::2])

    for start, length in zip(starts, lengths):
        mask[start - 1:start - 1 + length] = 255

    return Image.fromarray(mask.reshape(shape), mode='L')


def process_csv(csv_file, image_dir, save_dir):
    """
    CSV 파일을 읽어 이미지 마스크를 생성하고 저장하는 함수.

    :param csv_file: CSV 파일 경로
    :param image_dir: 이미지 파일이 위치한 디렉토리
    :param save_dir: 마스크 이미지 저장 디렉토리
    """
    if not os.path.isfile(csv_file):
        raise FileNotFoundError(f"CSV 파일이 존재하지 않습니다: {csv_file}")

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 첫 번째 행 건너뛰기 (헤더)

        image_list = os.listdir(image_dir)
        
        # tqdm을 사용하여 진행 상황 표시
        for idx, line in enumerate(tqdm(reader, desc="Processing CSV", total=len(image_list))):
            img_name = image_list[idx]
            img_path = os.path.join(image_dir, img_name)
            rle = line[2]

            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask_image = rle_to_mask(rle, image.shape)

            save_path = os.path.join(save_dir, f"{line[0]}.png")
            mask_image.save(save_path)


def main():
    # CLI 인자 처리
    parser = argparse.ArgumentParser(description='RLE to Mask Image Converter')
    parser.add_argument('csv_file', type=str, help='CSV file path')
    parser.add_argument('image_dir', type=str, help='Image directory path')
    parser.add_argument('save_dir', type=str, help='Mask image directory path')

    args = parser.parse_args()

    # 디렉토리가 존재하지 않으면 생성
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # CSV 파일을 읽고 마스크 이미지 저장
    process_csv(args.csv_file, args.image_dir, args.save_dir)


if __name__ == '__main__':
    main()
