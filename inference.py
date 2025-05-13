import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.models import segmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import SAVE_MODEL_NAME

from utils import rle_encode, load_model  # load_model은 현재 사용 X
from dataset import SatelliteDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_transform():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2()
    ])

def load_model(model_path, num_classes=1, device=device):
    model = segmentation.deeplabv3_resnet50(pretrained=False, num_classes=num_classes)

    state_dict = torch.load(model_path, map_location=device)

    # 클래스 수 다르면 classifier 레이어 재정의
    if model.classifier[4].out_channels != num_classes:
        model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
        torch.nn.init.xavier_uniform_(model.classifier[4].weight)

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return model


def run_inference(args):
    transform = get_transform()

    test_dataset = SatelliteDataset(csv_file=args.test_csv, transform=transform, infer=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = load_model(args.model_path, num_classes=1, device=device)

    result = []

    with torch.no_grad():
        for images in tqdm(test_loader, desc="Inferencing"):
            images = images.to(device).float()
            outputs = model(images)['out']
            masks = torch.sigmoid(outputs).cpu().numpy()
            masks = np.squeeze(masks, axis=1)
            masks = (masks > args.threshold).astype(np.uint8)

            for i in range(len(images)):
                encoded = rle_encode(masks[i])
                result.append(encoded if encoded else -1)

    # 결과 저장
    submit = pd.read_csv(args.sample_submission)
    submit['mask_rle'] = result
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    submit.to_csv(args.output_csv, index=False)
    print(f"[✅] Inference result saved to: {args.output_csv}")


def get_args():
    parser = argparse.ArgumentParser(description="Inference for Satellite Building Segmentation")

    parser.add_argument('--test_csv', type=str, default='./data/test.csv', help='Path to test CSV file')
    parser.add_argument('--sample_submission', type=str, default='./data/sample_submission.csv', help='Path to sample submission CSV')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model (.pth)')
    parser.add_argument('--output_csv', type=str, default=f'./{SAVE_MODEL_NAME}_submit.csv', help='Path to save output CSV')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--threshold', type=float, default=0.35, help='Threshold for binary mask')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    run_inference(args)

# example usage
'''
python inference.py \
  --test_csv ./data/test.csv \
  --sample_submission ./data/sample_submission.csv \
  --model_path ./checkpoints/deeplabv3_epoch40.pth \
  --output_csv ./result/submit.csv \
  --batch_size 4 \
  --threshold 0.35
  '''