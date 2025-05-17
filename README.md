# Building Area Segmentation

## 📌 Project Description  
위성 이미지의 건물 영역 분할(Image Segmentation)을 수행하는 AI모델 개발 프로젝트

## 📁 Dataset  
- 데이터 출처: SW중심대학 공동 AI 경진대회 2023
- train_img: TRAIN_0000.png ~ TRAIN_7139.png (1024x1024)  
- test_img: TEST_00000.png ~ TEST_60639.png (224x224)  
- train.csv: 이미지 ID, 경로, RLE 인코딩된 건물 마스크 정보  
- test.csv: 이미지 ID, 경로 정보

## 🧹 Data Preprocessing & Augmentation  
- RandomCrop: 이미지 일부분을 랜덤하게 잘라내어 다양한 크기와 위치의 건물 인식 학습
- Horizontal / Vertical Flip: 이미지를 좌우·상하로 뒤집어 건물 방향에 대한 불변성 학습
- Rotate: 건물이 다양한 각도에서 나타나는 모습을 학습하도록 회전 적용
- ColorJitter: 밝기, 대비, 채도, 색조를 무작위 변환해 다양한 조명 조건에 적응
- ToGray: 확률적으로 이미지를 회색조로 변환해 색상 정보 없이도 학습 가능

## 🧠 Model  
- 기본 구조: PSPNet  
- Backbone: DenseNet161 (효율적 특징 추출 및 높은 성능 제공)  
- 손실 함수: Focal Loss + Dice Loss  

## 📈 Evaluation  
- Dice Score: 0.621 (모델 성능 지표)  
