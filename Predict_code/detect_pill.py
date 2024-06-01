from ultralytics import YOLO
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# 모델 경로 지정
model_path = '../Pill_model/weights/best.pt'

# 예측할 image 이름 입력
predict_image = '5'
predict_image_path = f"../input_image/{predict_image}.jpg"
predict_image_crop_path = f"../input_image/crop/{predict_image}_crop.jpg"

# 모델 로드
model = YOLO(model_path)

# 클래스 이름 로드 (yaml 파일의 내용)
class_names = [
  '196000001', '196500004', '199502575', '200008571', '200402485', '200402928', '200404719',
  '200410999', '200600116', '200702709', '200702970', '200806299', '201207004', '201403243',
  '201503295', '201701435', '201705221', '201901014', '201906965', '201907607'
]

# 이미지 읽기
img_path = predict_image_path
img = cv2.imread(img_path)

# 모델 예측
prediction = model.predict(img)[0]

# 검출된 객체 수
num = len(prediction.boxes)
print("Number of objects detected: ", num)

# 예측된 클래스 인덱스 및 확률 출력
class_indices = prediction.boxes.cls.cpu().numpy()
confidences = prediction.boxes.conf.cpu().numpy()

# 바운딩 박스
BBox = []

# 예측 결과를 이미지로 저장 (OpenCV로 텍스트 크기 조정)
for box, cls_idx, conf in zip(prediction.boxes, class_indices, confidences):
    # x1, y1: 바운딩 박스의 왼쪽 상단 모서리의 좌표
    # x2, y2: 바운딩 박스의 오른쪽 하단 모서리의 좌표
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

    BBox.append([x1, y1, x2, y2])

print(BBox)


###############################################################################

# 텍스트 추가
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
font_color = (255, 255, 255)  # 흰색
line_type = 2

# 시작 좌표 설정 (좌측 상단)
x, y = 10, 30

class ResizeAndPad(object):
    def __init__(self, desired_size):
        self.desired_size = desired_size

    def __call__(self, image):
        desired_width, desired_height = self.desired_size
        original_width, original_height = image.size

        # 비율 유지하여 리사이즈
        ratio = min(desired_width / original_width, desired_height / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)

        # 새 이미지 생성 및 패딩
        new_image = Image.new("RGB", (desired_width, desired_height))
        pad_left = (desired_width - new_width) // 2
        pad_top = (desired_height - new_height) // 2
        new_image.paste(resized_image, (pad_left, pad_top))

        return new_image

class PillModel(nn.Module):
    # bulid cnn model
    def __init__(self):
        super(PillModel, self).__init__()

        # 클래스 개수 정의
        self.m_ClassNum = 20
        # ResNet 모델 초기화
        self.resnet = models.resnet18(pretrained=True)

        # 마지막 fully connected layer의 출력 크기를 클래스 수에 맞게 조정
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, self.m_ClassNum)


    def forward(self, x):
        # ResNet 모델의 forward pass 수행
        x = self.resnet(x)
        return x

# 이미지 전처리 함수
def preprocess_image(image_path):
    transform = transforms.Compose([ResizeAndPad((500, 500)),
                                           transforms.ToTensor()])
    image = Image.open(image_path).convert("RGB")  # 이미지를 RGB 형식으로 열기
    image = transform(image).unsqueeze(0)         # 배치 차원 추가
    return image

# 모델 로드
model = PillModel()
checkpoint = torch.load('../Pill_model/weights/ResNet.pt', map_location=torch.device('cpu'))  # 모델 파일 로드
model.load_state_dict(checkpoint['model_state_dict'])  # 모델의 상태 사전 로드
model.eval()

# 클래스 이름 목록
class_names = checkpoint['label_name']

# 이미지 분류 함수
def classify_image(image, model, class_names):
    # 모델 추론
    with torch.no_grad():
        output = model(image)

    # 확률로 변환
    probabilities = torch.softmax(output, dim=1)

    # 클래스 이름과 해당 확률 출력
    # for i, class_name in enumerate(class_names):
    #     print(f"{class_name}: {probabilities[0][i].item():.2f}")

    # 상위 3개 클래스와 해당 확률 가져오기
    top3_probabilities, top3_indices = torch.topk(probabilities, 3)

    # K 리스트에 클래스 이름을 확률 순서대로 저장
    K = [class_names[idx] for idx in top3_indices[0].cpu().numpy()]
    S = [f"{prob:.5f}" for prob in top3_probabilities[0].cpu().numpy()]

    print(f'K = {K}\nS = {S}')

    # 클래스 예측
    _, predicted_class = torch.max(probabilities, 1)

    # 클래스 이름 가져오기
    class_name = class_names[predicted_class.item()]

    return class_name, probabilities[0][predicted_class].item(), K, S

# 이미지 분류
image_path = predict_image_path  # 이미지 파일 경로

# 이미지 크롭
image = cv2.imread(image_path)

# 바운딩 박스 좌표 확인
print(f'바운딩 박스 = {BBox}')

# 박스의 개수만큼 ResNet
for i in range(len(BBox)):
    # 좌표를 정수로 변환
    x1, y1, x2, y2 = map(int, BBox[i])

    # 이미지 크롭
    cropped_image = image[y1:y2, x1:x2]

    # 이미지 저장
    cv2.imwrite(predict_image_crop_path, cropped_image)

    # 새 이미지 분류
    image_path_new = predict_image_crop_path  # 이미지 파일 경로

    # 이미지 전처리
    image_preprocess = preprocess_image(image_path_new)
    predicted_class, confidence, K, S = classify_image(image_preprocess, model, class_names)

    # 결과 출력
    print("예측된 클래스:", predicted_class)
    print("확신도:", confidence)

    # YOLO 스타일 텍스트 추가
    text = f"{predicted_class}"  # 예시 텍스트
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, line_type)

    # 텍스트 배경 사각형 그리기 (YOLO 스타일)
    cv2.rectangle(image, (x1, y1 - text_height - baseline), (x1 + text_width, y1), (60, 170, 120), cv2.FILLED)
    cv2.putText(image, text, (x1, y1 - baseline), font, 0.5, font_color, 2)


    cv2.rectangle(image, (x1, y1), (x2, y2), (60, 170, 120), 3)  # 붉은색 사각형, 두께는 2

    text_intro = f'The probability of the {i+1} pill'
    # 텍스트 크기 계산
    (text_width, text_height), baseline = cv2.getTextSize(text_intro, font, font_scale, line_type)
    # 배경 사각형 그리기
    cv2.rectangle(image, (x, y - text_height - baseline), (x + text_width, y + baseline), (0, 0, 0), cv2.FILLED)
    
    cv2.putText(image, text_intro, (x, y), font, font_scale, font_color, line_type)
    y += 30
    for k, s in zip(K, S):
        text = f"{k}: {float(s)*100}%"
        # 텍스트 크기 계산
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, line_type)
        # 배경 사각형 그리기
        cv2.rectangle(image, (x, y - text_height - baseline), (x + text_width, y + baseline), (0, 0, 0), cv2.FILLED)
        cv2.putText(image, text, (x, y), font, font_scale, font_color, line_type)
        y += 30  # 다음 텍스트의 y 좌표를 아래로 이동
    y += 30

# 결과 이미지 보여주기
cv2.imshow('Prediction', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

