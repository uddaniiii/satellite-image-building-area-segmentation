import numpy as np
import torch
import os


def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# 모델 저장 함수
def save_model(epoch, epoch_loss, model, save_dir,save_model_name):
    os.makedirs(save_dir, exist_ok=True)  # 디렉토리 없으면 생성
    model_name = f"{save_model_name}_epoch{epoch}_loss{epoch_loss:.4f}.pth"
    save_path = os.path.join(save_dir, model_name)
    torch.save(model.state_dict(), save_path)

def load_model(model, load_path):
    state_dict = torch.load(load_path)
    model_dict = model.state_dict()
    new_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"모델 불러오기 완료: {load_path}")