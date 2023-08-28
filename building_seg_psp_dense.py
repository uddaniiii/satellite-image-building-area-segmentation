#!/usr/bin/env python
# coding: utf-8

# # Import

# In[2]:


import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models.segmentation as segmentation
import segmentation_models_pytorch as smp
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
from albumentations.core.composition import Compose
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# # Fuctions

# ## RLE decoding/encoding

# In[3]:


# RLE 디코딩 함수
def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# ## Custom Dataset

# In[4]:


class SatelliteDataset(Dataset):
    def __init__(self, csv_file, transform=None, infer=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_rle = self.data.iloc[idx, 2]
        mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if self.infer:
            return image

        return image, mask


# ## Loss Function

# In[5]:


class FocalDiceLoss(nn.Module):
    def __init__(self, gamma=2, smooth=1e-6):
        super(FocalDiceLoss, self).__init__()
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Focal Loss 계산
        inputs_prob = torch.sigmoid(inputs)
        focal_loss = -targets * (1 - inputs_prob) ** self.gamma * torch.log(inputs_prob + self.smooth) \
                     - (1 - targets) * inputs_prob ** self.gamma * torch.log(1 - inputs_prob + self.smooth)
        focal_loss = focal_loss.mean()

        # Dice Loss 계산
        dice_target = targets
        dice_output = inputs_prob
        intersection = (dice_output * dice_target).sum()
        union = dice_output.sum() + dice_target.sum()
        dice_loss = 1 - (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Focal Loss와 Dice Loss를 더해서 총 손실을 계산
        total_loss = focal_loss + dice_loss

        return total_loss


# # Model Define

# In[6]:


model = smp.PSPNet(encoder_name="densenet161",  # 필수 파라미터: 사용할 인코더 백본의 이름
    in_channels=3,    # 필수 파라미터: 입력 이미지의 채널 수 (일반적으로 3(RGB) 또는 1(Grayscale))
    classes=1,        # 필수 파라미터: 세그멘테이션 클래스의 수 (예: 물체 탐지의 경우 물체 클래스 수)
    encoder_weights="imagenet"  # 선택적 파라미터: 사용할 사전 훈련된 인코더 가중치의 경로 또는 'imagenet'으로 
    설정하여 ImageNet 가중치 사용
)
# model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1)) # 1*1 컨볼루션 레이어 생성, 
# 입력 채널 256, 출력 채널 1로 설정
model = model.to(device)

criterion = FocalDiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4) 


# # Model Train

# ## Model Save & Model load

# In[7]:


save_dir = "./path/save/psp_dense_base/"  # 모델 저장 디렉토리
model_name = "psp_dense_base_trained_epoch{}.pth"  # 모델 파일 이름 패턴

# 훈련된 모델을 저장하는 함수
def save_model(model, epoch):
    save_path = save_dir + model_name.format(epoch)
    torch.save(model.state_dict(), save_path)
    print(f"Epoch {epoch} 모델 저장이 완료되었습니다.")

# 모델 불러오는 함수
def load_model(model, load_path):
    state_dict = torch.load(load_path)
    # 이전에 저장된 모델과 현재 모델 간 레이어 일치 여부 확인
    model_dict = model.state_dict()
    new_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("모델 불러오기가 완료되었습니다.")


# In[8]:


#그래프 생성 리스트 초기화
train_losses = []
val_losses = []


# In[8]:


# # 데이터 증강 확인

# for epoch in range(50): # 에폭

#     # 데이터 증강을 위한 transform 파이프라인 정의
#     transform = A.Compose([
#         A.RandomCrop(width=224,height=224,p=0.7),
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.ToGray(p=0.2),
#         A.Rotate(limit=30,p=0.3),
#         A.ColorJitter(p=0.3),

#         # # 고정값
#         A.Resize(224, 224),  # 이미지 크기 조정
#         ToTensorV2()  # 이미지를 텐서로 변환
#         # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 이미지 픽셀값 정규화
#     ])
#     dataset = SatelliteDataset(csv_file='./train.csv') # dataset 불러오기
#     aug_dataset = SatelliteDataset(csv_file='./train.csv', transform=transform) # dataset 불러오기

#     image,mask=dataset.__getitem__(epoch)
#     aug_image,aug_mask=aug_dataset.__getitem__(epoch)

#     plt.figure()
#     plt.subplot(1, 3, 1)
#     plt.imshow(image)
#     plt.subplot(1, 3, 2)
#     plt.imshow(aug_image.permute(1,2,0).numpy())
#     plt.subplot(1, 3, 3)
#     plt.imshow(aug_mask.permute(0,1).numpy())
#     plt.show()


# In[ ]:


# # 이전에 저장된 모델을 불러옵니다.
# load_path = "./path/save/psp_dense_base/psp_dense_base_trained_epoch43.pth"  # 이전에 저장된 모델 파일 경로
# load_model(model, load_path)

for epoch in range(50): # 에폭
    epoch+=1
    model.train() # 학습 모드 설정
    epoch_loss = 0
    correct_pixels = 0
    total_pixels = 0
    correct_pixels_train = 0
    total_pixels_train = 0 

    # 데이터 증강을 위한 transform 파이프라인 정의
    transform = A.Compose(
        [
            A.RandomCrop(width=224,height=224,p=0.7),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30,p=0.3),
            A.ColorJitter(p=0.3),
            A.ToGray(p=0.2),

            # # 고정값
            A.Resize(224, 224),  # 이미지 크기 조정
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 이미지 픽셀값 정규화
            ToTensorV2()  # 이미지를 텐서로 변환
        ]
    )

    val_transform = A.Compose(
        [
            # A.RandomCrop(width=224,height=224,p=0.7),
            # A.RandomRotate90(),  # 90도 회전 (랜덤하게)
            # A.HorizontalFlip(p=0.5),  # 수평 뒤집기 확률 50%
            # A.VerticalFlip(p=0.5),  # 수직 뒤집기 확률 50%
            # A.CLAHE(p=0.2),  # CLAHE를 통한 대비 개선

            # A.Resize(224, 224),  # 이미지 크기 조정
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 이미지 픽셀값 정규화
            ToTensorV2()  # 이미지를 텐서로 변환
        ]
    )

    dataset = SatelliteDataset(csv_file='./train.csv', transform=transform) # dataset 불러오기
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    val_dataset = SatelliteDataset(csv_file='./validation.csv', transform=val_transform)  # validation 
    데이터셋 로딩
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    for images, masks in tqdm(dataloader):
        # GPU 디바이스로 데이터 이동
        images = images.float().to(device)
        masks = masks.float().to(device)

        optimizer.zero_grad() # 옵티마이저에 누적된 변화도(gradient)를 초기화
        outputs = model(images) # 입력 이미지를 모델에 전달하여 모델의 출력을 계산
        loss = criterion(outputs, masks.unsqueeze(1)) # 모델의 출력과 정답 마스크 사이의 손실을 계산
        
        loss.backward() # 역전파를 통해 모델의 파라미터에 대한 손실 함수의 기울기(gradient)를 계산
        optimizer.step() # 옵티마이저를 사용하여 모델의 파라미터를 업데이트

        epoch_loss += loss.item() # 학습 과정에서 전체 에포크의 손실을 계산하기 위해 사용

        # 정확도 계산
        predicted_masks_train = (torch.sigmoid(outputs) > 0.5).float()
        correct_pixels_train += (predicted_masks_train == masks.unsqueeze(1)).sum().item()
        total_pixels_train += masks.numel()

    epoch_loss /= len(dataloader)
    accuracy_train = correct_pixels_train / total_pixels_train
    # training_loss_values[epoch]=epoch_loss

    # Validation 과정
    model.eval()  # 모델을 평가 모드로 설정 (Dropout 등의 레이어들이 평가 모드로 동작)
    total_validation_loss = 0.0
    correct_pixels_val = 0
    total_pixels_val = 0

    with torch.no_grad():  # 그라디언트 계산 비활성화
        for val_images, val_masks in val_dataloader:
            val_images = val_images.float().to(device)
            val_masks = val_masks.float().to(device)

            val_outputs = model(val_images)
            val_loss = criterion(val_outputs, val_masks.unsqueeze(1))

            total_validation_loss += val_loss.item() * val_images.size(0)  # 배치 내 샘플 수로 스케일링

            # 정확도 계산
            predicted_masks_val = (torch.sigmoid(val_outputs) > 0.5).float()
            correct_pixels_val += (predicted_masks_val == val_masks.unsqueeze(1)).sum().item()
            total_pixels_val += val_masks.numel()


    average_validation_loss = total_validation_loss / len(val_dataset)
    accuracy_val = correct_pixels_val / total_pixels_val

    # 각 에폭이 끝날 때마다 모델 저장
    save_path = save_dir + model_name.format(epoch)
    torch.save(model.state_dict(), save_path)
    print(f"Epoch {epoch} 모델 저장이 완료되었습니다.")
    print(f'Epoch {epoch}, Training Loss: {epoch_loss}, Training Accuracy: {accuracy_train}, \
          Validation Loss: {average_validation_loss}, Validation Accuracy: {accuracy_val}')
    
    train_losses.append(epoch_loss)
    val_losses.append(average_validation_loss)

    # Convert the loss values to numpy arrays using .cpu() method
    train_losses_np = torch.tensor(train_losses).cpu().numpy()
    val_losses_np = torch.tensor(val_losses).cpu().numpy()

    # Plot the losses after each epoch
    plt.plot(range(1, len(train_losses_np) + 1), train_losses_np, label='Train Loss')
    plt.plot(range(1, len(val_losses_np) + 1), val_losses_np, label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# # Transform Define

# In[14]:


# albumentations 라이브러리를 사용하여 이미지 데이터에 대한 변환(transform) 파이프라인 정의
transform = A.Compose(
    [
        # A.Resize(224, 224), # 이미지 크기 조정
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 이미지 픽셀값 정규화
        ToTensorV2() # 이미지를 텐서로 변환
    ]
)


# # Test Data Loader

# In[ ]:


test_dataset = SatelliteDataset(csv_file='./test.csv', transform=transform, infer=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)
print(len(test_dataset))
print(len(test_dataloader))


# # Load Model

# In[ ]:


model = smp.PSPNet(encoder_name="densenet161",  # 필수 파라미터: 사용할 인코더 백본의 이름
    in_channels=3,    # 필수 파라미터: 입력 이미지의 채널 수 (일반적으로 3(RGB) 또는 1(Grayscale))
    classes=1,        # 필수 파라미터: 세그멘테이션 클래스의 수 (예: 물체 탐지의 경우 물체 클래스 수)
    encoder_weights="imagenet"  # 선택적 파라미터: 사용할 사전 훈련된 인코더 가중치의 경로 또는 'imagenet'으로
      설정하여 ImageNet 가중치 사용
)

# 저장된 모델의 파라미터 불러오기 (strict=False 옵션 사용)
state_dict = torch.load('./path/save/ensemble/psp_dense_base_trained_epoch55.pth', map_location=
                        torch.device('cpu'))

# 저장된 모델의 클래스 수 (1개의 클래스일 때)
saved_num_classes = 1

# 현재 모델의 클래스 수 (예시로 21로 설정, 실제 사용하는 클래스 수로 수정)
current_num_classes = 1

# 모델의 분류기 레이어 크기 변경
if saved_num_classes != current_num_classes:
    # 모델의 분류기 레이어를 1x1 컨볼루션 레이어로 수정
    model.classifier[4] = torch.nn.Conv2d(256, current_num_classes, kernel_size=(1, 1), stride=(1, 1))
    # 모델의 분류기 레이어를 초기화
    torch.nn.init.xavier_uniform_(model.classifier[4].weight)  # 또는 다른 초기화 방법 사용

# 모델의 파라미터 로드
model.load_state_dict(state_dict, strict=False)

# GPU 사용이 가능한 경우에는 GPU로 데이터 이동
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# # Inference

# In[ ]:


# 결과를 저장할 리스트 초기화
result = []

with torch.no_grad(): # 역전파 비활성화, 파라미터 업데이트 금지
    # print(len(test_dataloader))
    for images in tqdm(test_dataloader): # 데이터 로드
        images = images.float().to(device) 

        outputs = model(images) # 테스트 이미지 전달하여 예측 결과 얻음
        masks = torch.sigmoid(outputs).cpu().numpy() # outputs는 모델 예측 결과, 확률값으로 변환하기 위해 
        시그모이드 함수 적용한 후 각 픽셀 값을 0과 1사이의 확률값으로 변환하고, 넘파이 배열로 변환
        masks = np.squeeze(masks, axis=1) # 불필요한 차원 제거
        masks = (masks > 0.35).astype(np.uint8) # 최종 이진화 예측 마스크 얻음
        
        # print(len(images))
        for i in range(len(images)):
            mask_rle = rle_encode(masks[i]) # RLE로 변환, mask_rle에 인코딩 결과 저장
            if mask_rle == '':
                result.append(-1) # 빌딩 없으면 -1 저장
            else:
                result.append(mask_rle) # 아니면 mask_rle 저장

            visualized_image = images[i].cpu().numpy().transpose((1, 2, 0)) # 이미지 시각화하기 위해 넘파이 
            배열로 가져옴
            masks_visualized = masks[i] * 255 # 이진화 마스크로 변환

            plt.subplot(1, 2, 1)
            plt.imshow(visualized_image)
            plt.title("Input Image")

            plt.subplot(1, 2, 2)
            plt.imshow(masks_visualized, cmap='gray')
            plt.title("Predicted Mask")

            plt.show()


# # Submission

# In[ ]:


submit = pd.read_csv('./sample_submission.csv')
submit['mask_rle'] = result


# In[ ]:


submit.to_csv('./submit.csv', index=False)