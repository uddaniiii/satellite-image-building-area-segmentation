import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from torchvision.models import segmentation
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
from config import DEVICE, BATCH_SIZE, NUM_CLASSES, NUM_EPOCHS, LEARNING_RATE, SAVE_DIR
from dataset import SatelliteDataset
from utils import load_model, save_model

# 데이터 증강을 위한 transform 파이프라인 정의
transform = A.Compose([
    A.RandomCrop(224, 224),
    A.RandomRotate90(),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.CLAHE(p=0.2),
    A.Resize(224, 224),
    A.Normalize(),
    ToTensorV2()
])

# 데이터셋과 데이터로더 정의
def get_data_loader(csv_file):
    dataset = SatelliteDataset(csv_file=csv_file, transform=transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 모델 초기화
def initialize_model():
    model = segmentation.deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=(1, 1), stride=(1, 1))  # 마지막 레이어 수정
    return model.to(DEVICE)

# 모델 학습 및 저장
def train_model(model, dataloader, optimizer, criterion, num_epochs=NUM_EPOCHS, start_epoch=0, save_dir=SAVE_DIR, log_file='progress.txt'):
    # 로그 설정
    logging.basicConfig(filename=log_file, level=logging.INFO)
    logging.info("Starting training...")

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()  # 학습 모드 설정
        epoch_loss = 0
        correct_pixels = 0
        total_pixels = 0

        for images, masks in tqdm(dataloader, desc=f"Epoch {epoch}"):
            images, masks = images.float().to(DEVICE), masks.float().to(DEVICE)
            optimizer.zero_grad()

            outputs = model(images)['out']
            loss = criterion(outputs, masks.unsqueeze(1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # 정확도 계산
            predicted_masks = outputs > 0.5  # 이진 분류 문제
            correct_pixels += (predicted_masks == masks.byte().unsqueeze(1)).sum().item()
            total_pixels += masks.numel()


        # 모델 저장
        save_model(epoch,epoch_loss, model, save_dir)

        # 에폭 손실 및 정확도 출력
        epoch_loss /= len(dataloader)
        accuracy = correct_pixels / total_pixels
        logging.info(f'Epoch {epoch}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}, Val Loss: {val_loss:.4f}')

        # 진행 상황을 로그에 기록
        logging.info(f"Epoch {epoch} completed. Train Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}, Val Loss: {val_loss:.4f}")

# 모델 로딩 후 epoch 번호 추출
def extract_epoch_from_model_filename(model_filename):
    try:
        epoch = int(model_filename.split('_')[1].replace('epoch', ''))  # 'epoch20.pth' -> 20 추출
        return epoch
    except Exception as e:
        raise ValueError(f"모델 파일에서 에폭을 추출하는 데 오류가 발생했습니다: {e}")

# 메인 실행 코드
def main():
    # CLI 인자 파서
    parser = argparse.ArgumentParser(description="Train Model")
    parser.add_argument('--csv', type=str, required=True, help="Path to the CSV file for training data")
    parser.add_argument('--load_model', type=str, help="Path to the pre-trained model checkpoint (optional)")
    parser.add_argument('--save_dir', type=str, default=SAVE_DIR, help="Directory to save the trained model")
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help="Number of epochs to train")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument('--log_file', type=str, default='progress.txt', help="Log file to save the progress")

    args = parser.parse_args()

    # 모델 초기화
    model = initialize_model()

    # 모델 로드가 필요한 경우
    start_epoch = 0  # 새로 시작할 때는 epoch 0부터
    if args.load_model:
        load_model(model, args.load_model)
        start_epoch = extract_epoch_from_model_filename(args.load_model)  # 로드된 모델의 epoch 번호 추출

    # 데이터 로더 설정
    dataloader = get_data_loader(args.csv)
    
    # 손실 함수 및 옵티마이저 설정
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

    # 모델 학습
    train_model(model, dataloader, optimizer, criterion, num_epochs=args.epochs, start_epoch=start_epoch, save_dir=args.save_dir, log_file=args.log_file)

if __name__ == "__main__":
    main()
