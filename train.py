import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from torchvision.models import segmentation
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
from config import BATCH_SIZE, NUM_CLASSES, NUM_EPOCHS, LEARNING_RATE, SAVE_DIR, SAVE_MODEL_NAME
from dataset import SatelliteDataset
from utils import load_model, save_model
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_logger(log_file):
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        dir_name = os.path.dirname(log_file)
        if dir_name:  # 디렉토리 이름이 있을 때만 생성
            os.makedirs(dir_name, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger

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
    return model.to(device)

# train_model 수정
def train_model(model, dataloader, optimizer, criterion, logger, num_epochs=NUM_EPOCHS, start_epoch=0, save_dir=SAVE_DIR):
    logger.info("Starting training...")

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        epoch_loss = 0
        correct_pixels = 0
        total_pixels = 0

        for images, masks in tqdm(dataloader, desc=f"Epoch {epoch}"):
            images, masks = images.float().to(device), masks.float().to(device)
            optimizer.zero_grad()

            outputs = model(images)['out']
            loss = criterion(outputs, masks.unsqueeze(1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            predicted_masks = outputs > 0.5
            correct_pixels += (predicted_masks == masks.byte().unsqueeze(1)).sum().item()
            total_pixels += masks.numel()
        
        epoch_loss /= len(dataloader)
        
        # 모델 저장
        save_model(epoch, epoch_loss, model, save_dir, save_model_name=SAVE_MODEL_NAME)

        # 로그 출력
        accuracy = correct_pixels / total_pixels
        logger.info(f"Epoch {epoch} - Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")

# 모델 로딩 후 epoch 번호 추출
def extract_epoch_from_model_filename(model_filename):
    try:
        epoch = int(model_filename.split('_')[1].replace('epoch', ''))  # 'epoch20.pth' -> 20 추출
        return epoch
    except Exception as e:
        raise ValueError(f"모델 파일에서 에폭을 추출하는 데 오류가 발생했습니다: {e}")
    
# main 함수 수정
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

    logger = setup_logger(args.log_file)
    logger.info("Logger initialized.")

    model = initialize_model()

    start_epoch = 0
    if args.load_model:
        load_model(model, args.load_model)
        start_epoch = extract_epoch_from_model_filename(args.load_model)

    dataloader = get_data_loader(args.csv)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

    # logger 전달
    train_model(model, dataloader, optimizer, criterion, logger,
                num_epochs=args.epochs, start_epoch=start_epoch,
                save_dir=args.save_dir)

if __name__ == "__main__":
    main()
