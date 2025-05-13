import torch
import os
import cv2
import matplotlib.pyplot as plt
from model import get_deeplabv3
from utils import load_model
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_test_transform():
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(),
        ToTensorV2()
    ])


def visualize_inference(model_path, image_path, device='cuda'):
    model = get_deeplabv3(num_classes=2)
    load_model(model, model_path)
    model.to(device)
    model.eval()

    transform = get_test_transform()

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformed = transform(image=image_rgb)
    input_tensor = transformed['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)['out']
        pred_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(pred_mask, cmap='gray')
    plt.title("Predicted Mask")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# 예시 실행:
# visualize_inference('checkpoints/model_epoch_20.pth', './test_images/xxx.png')