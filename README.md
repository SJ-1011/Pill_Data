# Pill Data
> #### Input Image 속 알약의 이름을 CNN을 통해 알아내어 제공하는 프로그램


<br/>

## 제작 동기
현재 수많은 CNN 모델들이 여러 분야에 사용되고 있다. 하지만, 많은 사람들이 필요로 할법한 알약 검출 모델은 수가 상당히 적다. 약의 종류가 만 단위를 넘어가기도 하고, 이미지 데이터를 입수하기도 어렵기 때문일 것이다. 여기서 본인은 직접 식품의약처에 방문하여 실험용 5088개 class의 Dataset을 얻었고, 그 중에서 20가지 class로 알약 분류 모델을 만들고자 한다.

<br/>

## 실험용 Dataset
- 아래와 같이 총 20가지의 class로 구성되어있다. <br/>
![image](https://github.com/SJ-1011/Pill_Data/assets/109647265/934be6ba-800e-475e-aead-df4ab2cd5324)
<br/>

- 각 이미지들은 배경이 있는것과 없는 것으로 나누어져있다.

|![200806299_001](https://github.com/SJ-1011/Pill_Data/assets/109647265/61239bfc-3dd4-4d9c-9e82-45909c1e6ac1)|![200806299_001](https://github.com/SJ-1011/Pill_Data/assets/109647265/b30fb2fa-a4f1-4811-a936-ab70ad0de86e)|
|:---:|:---:|

<br/>

## 문제점 및 Insight
- 이미지의 크기가 굉장히 크고, 배경이 있는 이미지는 용량이 약 7MB 정도이다.
- 때문에 배경이 있는 이미지들로 Data Augmentation하기엔 저장공간이 매우 부족하다.
- 따라서, 기존 배경이 있는 이미지들로 객체 탐지 모델을 돌려서 객체의 위치를 탐지하고, 배경이 없는 이미지들을 Data Augmentation으로 Data의 수를 늘려서 Classification 모델을 만들었다.
  - 이렇게 함으로써 저장 공간을 아끼고, classification의 정확도를 높일 수 있다.
  - 다만, detection의 정확도가 떨어지는 단점이 있다.
 
<br/>

## 훈련 과정 (Pill_Classification_YOLO.ipynb, Pill_Classification_Model_Tarining_ResNet.ipynb)
- Google Colab Pro를 사용하였고, ResNet의 경우 약학처에서 제공한 기본 코드를 수정하여 사용하였다.
  <br/>
- 데이터 구조를 yolo의 훈련 셋에 맞게 설정 (train, val)
```
def restructure_dataset(base_dir='sample_split', output_dir='dataset'):
    for split in ['train', 'val']:
        image_output_dir = os.path.join(output_dir, 'images', split)
        label_output_dir = os.path.join(output_dir, 'labels', split)
        os.makedirs(image_output_dir, exist_ok=True)
        os.makedirs(label_output_dir, exist_ok=True)

        split_dir = os.path.join(base_dir, split)
        class_dirs = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]

        for class_dir in class_dirs:
            class_path = os.path.join(split_dir, class_dir)
            images = [f for f in os.listdir(class_path) if f.endswith('.jpg')]

            for image in images:
                base_name = os.path.basename(image)
                image_src_path = os.path.join(class_path, image)
                label_src_path = os.path.join(class_path, base_name.replace('.jpg', '.txt'))

                image_dst_path = os.path.join(image_output_dir, base_name)
                label_dst_path = os.path.join(label_output_dir, base_name.replace('.jpg', '.txt'))

                shutil.copy(image_src_path, image_dst_path)
                shutil.copy(label_src_path, label_dst_path)
```
- 모델 생성 및 모델 훈련
```
# 모델 생성 (YOLOv8s 사용)
model = YOLO('yolov8s.pt')

# 데이터셋 yaml 파일 경로
data_yaml = '/content/gdrive/MyDrive/dataset_yolo.yaml'

# 훈련
model.train(data=data_yaml, epochs=50, batch=16, name='Pill_model', device=0)
```
- ResNet 훈련을 위한 Dataset split
```
    def separate(self, dir_path, x):
        dirname = self.open_path + x
        filenames = os.listdir(dirname)
        i = 0
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)

            with Image.open(full_filename) as image:
                if i % 10 < 7:
                    training_directory = os.path.join(dir_path + 'training/', x)
                    shutil.copyfile(full_filename, os.path.join(training_directory, filename))

                elif i % 10 >= 7 and i % 10 < 8:
                    validation_directory = os.path.join(dir_path + 'testing/', x)
                    shutil.copyfile(full_filename, os.path.join(validation_directory, filename))

                else:
                    testing_directory = os.path.join(dir_path + 'validation/', x)
                    shutil.copyfile(full_filename, os.path.join(testing_directory, filename))
            i = i + 1
            print(f'{filename} finish')
```
- ResNet 모델
```
class PillModel(nn.Module):
    # bulid cnn model
    def __init__(self, config):
        super(PillModel, self).__init__()
        '''
        ClassNum : class number
        '''

        self.m_ClassNum = int(config['class_num'])
        # ResNet 모델 초기화
        self.resnet = models.resnet18(pretrained=True)

        # 마지막 fully connected layer의 출력 크기를 클래스 수에 맞게 조정
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, self.m_ClassNum)


    def forward(self, x):
        # ResNet 모델의 forward pass 수행
        x = self.resnet(x)
        return x
```
- Data Augmentation으로 이미지 회전 사용
```
def rotate_image_circle(self, save_rotate_img, input_image):
        i = 0
        height, width, channel = input_image.shape
    
        while i < 360:
            f_path = save_rotate_img + '_' + str(i) + '.png'
            if not os.path.isfile(f_path):
                matrix = cv2.getRotationMatrix2D((width/2, height/2), i, 1)
                dst = cv2.warpAffine(input_image, matrix, (width, height))
                dst = self.CropShape(dst)
 
                cv2.imwrite(f_path, dst)
            else:
                print('rotate file exits : ', f_path)
            
            i = i + self.rotate_angle_circle
```

<br/>

## 프로그램 기능 설명 (detect_pill.py)
- 본격적으로 알약을 탐지하고 분류하는 기능이다.
- class의 이름
```
class_names2 = [
  '196000001', '196500004', '199502575', '200008571', '200402485', '200402928', '200404719',
  '200410999', '200600116', '200702709', '200702970', '200806299', '201207004', '201403243',
  '201503295', '201701435', '201705221', '201901014', '201906965', '201907607'
]

class_eng = [
    'Tetracyclin Cap', 'Flasinyl', 'Ibuprofen', 'Zyvox', 'Gluphen', 'Gin Q Green', 'Genuone Doxazocin Mesylate',
    'Binexomeprazole Cap', 'Plunazol', 'Muless Cap', 'Panprazol', 'Parox Cr', 'Singulmon Chewable', 'Seperisone',
    'M-strong', 'Risti Cap', 'Gnal-N Nose Plus Soft Cap', 'Proloxofen', 'Alison Soft Cap', 'Perkin CR'
]
```
- YOLO 모델 예측 후 바운딩 박스 얻기
```
# 모델 예측
prediction = model.predict(img)[0]

...

# 예측 결과를 이미지로 저장 (OpenCV로 텍스트 크기 조정)
for box, cls_idx, conf in zip(prediction.boxes, class_indices, confidences):
    # x1, y1: 바운딩 박스의 왼쪽 상단 모서리의 좌표
    # x2, y2: 바운딩 박스의 오른쪽 하단 모서리의 좌표
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

    BBox.append([x1, y1, x2, y2])

...
```
- Predict
```
# 박스의 개수만큼 ResNet
for i in range(len(BBox)):
    # 좌표를 정수로 변환
    x1, y1, x2, y2 = map(int, BBox[i])

    # 이미지 크롭
    cropped_image = image[y1:y2, x1:x2]

    ...

    # 이미지 전처리 및 ResNet으로 Classify
    image_preprocess = preprocess_image(image_path_new)
    predicted_class, confidence, K, S = classify_image(image_preprocess, model, class_names)
```
- 이미지 분류 함수로 이미지 분류
```
# 이미지 분류 함수
def classify_image(image, model, class_names):
    # 모델 추론
    with torch.no_grad():
        output = model(image)

    # 확률로 변환
    probabilities = torch.softmax(output, dim=1)
```

<br/>

## 학습 결과

- YOLOv8
  - confusion matrix에 사용된 test data 수가 적어서 의미있는 결과는 아니다.

![confusion_matrix](https://github.com/SJ-1011/Pill_Data/assets/109647265/76a63039-dac8-4f4b-9499-cb766b3676e2)

![val_batch0_labels](https://github.com/SJ-1011/Pill_Data/assets/109647265/06a7b7ad-80eb-4337-adcd-bfd461f2a81c)

- ResNet
-   confusion matrix 그림 올리기

<br/>

## 예측 결과
- 이미지를 클릭하면 크게 볼 수 있습니다.

|![1](https://github.com/SJ-1011/Pill_Data/assets/109647265/0256a7d8-b008-4695-9dcb-13be00f8d8aa)|![1_predict](https://github.com/SJ-1011/Pill_Data/assets/109647265/2c539308-d4c5-4066-a9c9-26c0de781558)|
|:---:|:---:|
|![3](https://github.com/SJ-1011/Pill_Data/assets/109647265/6678592d-4d27-4b1c-b30f-9e41b1b74aaf)|![3_predict](https://github.com/SJ-1011/Pill_Data/assets/109647265/077d710c-be0a-4419-8dd9-ff75b28d5247)|
|![8](https://github.com/SJ-1011/Pill_Data/assets/109647265/946cd9fd-fd37-4c58-82b4-3a1dd08b5448)|![8_predict](https://github.com/SJ-1011/Pill_Data/assets/109647265/79794a4b-815f-42fb-982d-e1f94240bc41)|
|![9](https://github.com/SJ-1011/Pill_Data/assets/109647265/17583caf-415e-4f75-b31b-29b4dcd5d183)|![9_predict](https://github.com/SJ-1011/Pill_Data/assets/109647265/c340cc62-e366-4750-801a-8d69978f92a3)|

- 배경의 색이 달라지면 예측 확률이 상당히 떨어진다.
- 배경 색이 단조롭고 Detect만 우수하게 되면 classification은 잘 동작된다.

  <br/>

## Reference
- 약학정보원(Dataset, Code): https://www.health.kr/notice/notice_view.asp?show_idx=1001&search_value=&search_term=&paging_value=&setLine=&setCategory=
- Chat GPT
