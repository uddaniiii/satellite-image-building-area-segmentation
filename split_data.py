import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # tqdm 라이브러리 import

def split_data(csv_file, train_ratio=0.8, random_seed=42, train_file="./data/train_data.csv", test_file="./data/val_data.csv"):
    """
    CSV 파일을 읽고, 주어진 비율로 데이터를 학습용과 테스트용으로 분할하여 저장합니다.

    :param csv_file: 원본 CSV 파일 경로
    :param train_ratio: 학습 데이터 비율 (기본값: 0.8)
    :param random_seed: 랜덤 시드 (기본값: 42)
    :param train_file: 학습 데이터 저장 파일 경로 (기본값: ./data/train_data.csv)
    :param test_file: 테스트 데이터 저장 파일 경로 (기본값: ./data/val_data.csv)
    """
    # CSV 파일 읽기
    data = pd.read_csv(csv_file)

    # 데이터 분할
    train_data, test_data = train_test_split(data, train_size=train_ratio, random_state=random_seed)

    # 진행률 표시
    tqdm.write(f"학습 데이터 분할 진행 중...")

    # 분할된 데이터를 각각 파일로 저장
    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)

    tqdm.write(f"학습 데이터는 {train_file}에, 테스트 데이터는 {test_file}에 저장되었습니다.")


def main():
    # CLI 인자 처리
    parser = argparse.ArgumentParser(description='CSV 파일을 학습용과 테스트용으로 분할')
    parser.add_argument('csv_file', type=str, help='원본 CSV 파일 경로')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='학습 데이터 비율 (기본값: 0.8)')
    parser.add_argument('--random_seed', type=int, default=42, help='랜덤 시드 (기본값: 42)')
    parser.add_argument('--save_dir', type=str, default='./data/', help='분할된 데이터를 저장할 디렉토리 (기본값: 현재 디렉토리)')
    parser.add_argument('--train_file', type=str, default='./data/train_data.csv', help='학습 데이터 저장 파일 경로 (기본값: ./data/train_data.csv)')
    parser.add_argument('--test_file', type=str, default='./data/val_data.csv', help='테스트 데이터 저장 파일 경로 (기본값: ./data/val_data.csv)')
    
    args = parser.parse_args()

    # 함수 호출
    split_data(args.csv_file, train_ratio=args.train_ratio, random_seed=args.random_seed, train_file=args.train_file, test_file=args.test_file)

if __name__ == '__main__':
    main()
