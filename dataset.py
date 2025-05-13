import os
import cv2
from torch.utils.data import Dataset
from utils import rle_decode
import pandas as pd
from config import DATA_DIR

class SatelliteDataset(Dataset):
    def __init__(self, csv_file, transform=None, infer=False): # transform 전처리하거나 다른 형태로 변환하는데 사용, infer 데이터를 추론 모드로 설정할지 여부
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(DATA_DIR, self.data.iloc[idx, 1]) # 해당 인덱스에 위치한 데이터프레임의 두 번째 열(column)에 있는 이미지 경로를 가져옴

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
            return image

        mask_rle = self.data.iloc[idx, 2] # 데이터프레임의 idx 행에서 세 번째 열(column)에 있는 마스크 정보를 가져옴
        mask = rle_decode(mask_rle, (image.shape[0], image.shape[1])) # 원래의 마스크 이미지로 변환

        if self.transform:
            augmented = self.transform(image=image, mask=mask) # 이미지와 마스크를 변환
            image = augmented['image'] # 변환된 이미지를 딕셔너리에서 가져와 image 변수에 할당
            mask = augmented['mask'] # 변환된 이미지를 딕셔너리에서 가져와 mask 변수에 할당

        return image, mask