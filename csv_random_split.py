import random
import pandas as pd

# CSV 파일 경로
csv_file = "./train.csv"

# CSV 파일 읽기
data = pd.read_csv(csv_file)

# 데이터 개수
data_length = len(data)
# print(data_length)

# 데이터 랜덤 순서로 섞기
random_data = data.sample(frac=1, random_state=42)  # random_state는 재현 가능한 랜덤 결과를 위한 시드값

# 데이터 분할 비율
train_ratio = 0.8
test_ratio = 0.2

# 데이터 분할 인덱스 계산
train_end_index = int(data_length * train_ratio)
test_start_index = train_end_index + 1

# 데이터 분할
train_data = random_data.iloc[:train_end_index, :]
test_data = random_data.iloc[test_start_index:, :]

# 분할된 데이터를 각각 파일로 저장
train_data.to_csv("./train_train.csv", index=False)
test_data.to_csv("./test_train.csv", index=False)
