# AI Boostcamp P Stage Mask Classification
This codes is about to AI Boostcamp P stage Mask Classification competition.  
Below are code usage and My competitions review.  

# Requirements
```
albumentations==1.0.3
numpy==1.19.2
timm==0.4.12
torch==1.7.1
torchvision==0.8.2
tqdm==4.51.0
ttach==0.0.3
wandb==0.12.1
```
You can install requirements by typing  `pip install -r requirements.txt`

```
+-- train/
|   +-- images/
|       +-- 000001_female_Asian_45/
|       +-- 000002_female_Asian_52/
|       +-- …
|   +-- new_standard.csv
+-- eval/
    +-- images/
        +-- 814bff668ae5b9c595ceabcbb6e1ea84634afbd6.jpg
        +-- 819f47db0617b3ea9725ef1f6f58e56561e7cb4b.jpg
        +-- …
    +-- info.csv
```
You must have **mask dataset** and this **data directory structure** before execution.

# Usage
**BASIC**

`python3 train.py -config config.json`

**OPTION**
```
--seed : random seed default: 1010, type=int, default=1010
--epochs : number of epochs to train, type=int, default=5
--dataset : dataset augmentation type default: MaskDataSet, type=str, default=MaskDataSet
--augmentation : data augmentation type default: MaskAugmentation, type=str, default=MaskAugmentation
--resize, resize size for image when training, type=list, default=[256, 256]
--batch_size : batch size for training,  type=int, default=128
--kfold_splits : k-fold splits number for cross validation, type=int, default=5
--model : model type for learning, type=str, default: MultiDropoutEfficientLite0
--optimizer : optimizer type, type=str, default=AdamW
--lr : learning rate, type=float, default=1e-2
--criterion : criterion type, type=str, default=FocalLoss
--weight_decay : Weight decay for optimizer, type=int, default=1e-4, 
--lr_scheduler : learning scheduler, type=str, default=OneCycleLR
--log_interval : how many batches to wait before logging training status type=int, default=30 
--file_name : model save at {results}/{file_name} default=exp
--train_csv : train data saved csv, default=/opt/ml/input/data/train/new_standard.csv 
--test_dir : test data saved directory, default="/opt/ml/input/data/eval" 
--mix_up : if True, mix-up & cut-mix use, type=boolean_string, default=False
--num_class : input the number of class, type=int,default=18
--pseudo_label : pseudo label usage, Should write pseudo.csv location at pseudo_csv option, type=boolean_string, default=False
--pseudo_csv : pseudo label usage, type=str, default=/opt/ml/input/data/train/pseudo.csv
--wandb : logging in WandB, type=boolean_string, default=True
--patience : early stopping patience number, type=Int, default=5
```

- ex. `python3 train.py --config config.json --epochs 30 --file_name my_test_net --mix_up True --seed 42 --pseudo_label True --wandb False`

- **cause** : You have to download new_standard.csv and write file location at option `--train_csv`

- Detail configuration controll(Optimizer, Scheduler, ... etc) is updated as soon as possible

# Review

# 21-08-24 : EDA

1. **age값의 분포**
    - 첫째로 각 세대별 데이터 분포를 살펴 본 결과, 30,40,50대의 데이터가 다른 데이터에 비해 상당히 적음을 알 수 있었다. 따라서 학습 과정에서 해당 세대에 대한 underfitting과 나머지 세대에 대한 overfitting이 발생할 수 있음을 주의해야함.
    - 위의 age group을 실제 분류가 되는 class별로 나눈 후에 분포를 살펴보면, 이렇게 나이대를 class로 묶음으로써 30,40대의 데이터와 50대의 데이터를 하나로 간주해 30,40대의 underfitting에 유의하지 않아도 되지만 `60대` 데이터의 경우 그 차이가 더 극명하게 나타나 더욱 주의가 필요할 것으로 보인다.
    
![Untitled](https://user-images.githubusercontent.com/70624819/131848401-e251879a-ce21-4af4-8332-2fbace0d63d9.png)

![Untitled](https://user-images.githubusercontent.com/70624819/131848424-1fbffb70-bc15-4e3e-a19d-6a306634083e.png)

1. **RGB값과 마스크 색과의 연관성**

→ 마스크를 낀 상태와 끼지 않은 상태에서의 RGB값의 연관성을 살펴보려 했지만 `코딩 실력의 문제`로 인해 알아보지 못하였다 !!

→ 추론적으로 생각해보면 이미지 데이터에 마스크의 색이 여러 색이며 단색이 아닌 패턴 계열이 존재해 마스크를 끼지 않은 경우와 극명하게 차이가 날 것 같지는 않지만, 분명 특정한 마스크 패턴이 존재할 것으로 예상된다! → 추후에 구현해보자.

1. **데이터 로드시 문제점(초기 data load code)**

        class MaskTrainDataSet(torch.utils.data.Dataset):
            def __init__(self,data_dir=pathlib.Path('/opt/ml/input/data/train/images'),transform=None,device='cpu'):
                self.data_dir = data_dir
                self.transform=transform
                self.device=device
        				...
                self.X,self.y = self.load_data(self.data_dir,transform,device)
        				...
               
            def load_data(self,data_dir,transform,device):
            # gender labeling : { female : 0, male : 1 }
            # age labeling : { [10,30) : 0, [30,60) : 1, [60,~) : 2}
            # mask labeling : { mask : 0, normal : 1, incorrect : 2 }
                gender_labeling={
                    'female':0,
                    'male':1
                }
                age_labeling={
                    10:0,
                    20:0,
                    30:1,
                    40:1,
                    50:1,
                    60:2
                }
                mask_labeling={
                    'mask':0,
                    'normal':1,
                    'incorrect':2
                }
                X=[]
                y=[]
                for files_path in list(data_dir.glob('[0-9]*')):
                    _,gender,_,age = files_path.parts[-1].split('_')
                    age = int(age)//10 * 10
                    for file_path in files_path.glob('mask*'):
                        image = Image.open(file_path)
                        if transform:
                            image = transform(image)
                        X.append(image)
                        y.append([mask_labeling['mask'],gender_labeling[gender],age_labeling[age]])

                    for file_path in files_path.glob('normal*'):
                        image = Image.open(file_path)
                        if transform:
                            image = transform(image)
                        X.append(image)
                        y.append([mask_labeling['normal'],gender_labeling[gender],age_labeling[age]])

                    for file_path in files_path.glob('incorrect*'):
                        image = Image.open(file_path)
                        if transform:
                            image = transform(image)
                        X.append(image)
                        y.append([mask_labeling['incorrect'],gender_labeling[gender],age_labeling[age]])
                        
                X= torch.stack(X).to(device)
                y = torch.tensor(y,dtype=torch.long).to(device)
                print(X.shape,y.shape)
                return X,y

    위의 코드에서는 Dataset class를 init할 때 바로 전체 데이터를 loading하도록 설계하였다.

    이렇게 하니 전체 데이터의 용량이 큰 경우 메모리에 큰 무리가 가게 된다.

    → 따라서 `__getitem__`으로 data를 불러올 때 마다 데이터를 loading하는 방식으로 바꿔야 한다.

2. **y값 분석**
    - y값들의 관계를 분포도를 통해 알아보았다.

        → 각 y값 간의 차이가 분명하게 존재하기는 하지만 편향된 데이터 수집으로 인해 해당 값이 유의미한 결과를 나타내지는 않는다!

![Untitled](https://user-images.githubusercontent.com/70624819/131848405-5bb5a77e-b3c1-4a2d-88fa-57f6b2fbf6b8.png)

![Untitled](https://user-images.githubusercontent.com/70624819/131848406-13d71c95-bc4c-4be0-8ac3-1608e0ec9e40.png)

# 21-08-24 : Dataset & DataLoader

1. **dataset**

```python
class MaskDataSet(torch.utils.data.Dataset):
    def __init__(self,csv_file,root_dir,transform=None):
        self.root_dir=pathlib.Path(root_dir)
        self.transform=transform
        # 기존의 train.csv의 path값을 참조해 파일 path를 반환
        self.file_path = self.read_csv_file(csv_file,self.root_dir)

    def __getitem__(self,idx):
        gender_labeling={
            'male':0,
            'female':1
        }
        age_labeling={
            10:0,
            20:0,
            30:1,
            40:1,
            50:1,
            60:2
        }
        mask_labeling={
            'mask':0,
            'inco':1,
            'norm':2
        }
        
				...

        # image data 불러오기
        img_path = self.root_dir/self.file_path[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # label data 생성
        file_folder,file_name= str(self.file_path[idx]).split('/')
        _,gender,_,age = file_folder.split('_')
        age = int(age)//10*10
        wear = file_name.split('.')[0][:4]
        y = np.array([mask_labeling[wear], gender_labeling[gender], age_labeling[age]])
       
        return torch.tensor(np.array(image)),torch.tensor(y)
    
    def read_csv_file(self,csv_file,root_dir):
        ''' Return file path using directory path in csv_file  '''
```

위에서의 코드와 다르게 이번 Dataset에서는 `__getitem__`에서 데이터를 불러오도록 설계하였다. 이렇게 함으로써 메모리 과적합을 방지할 수 있었다!

+ **수정된 코드(21.08.26)**

```python
def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # image data 불러오기
        image = np.array(Image.open(self.image_paths[idx]))
        if self.transform:
            image = self.transform(image=image)['image']
        
        if self.multi_label:
            y = np.array(self.labels[idx])
        else:
            y = np.array(self.label_classes[idx])
        
        return image,y

def read_csv_file(self,train_dir):
        ''' Return file path using directory path in csv_file  '''
        data_pd = pd.read_csv(train_dir,encoding='utf-8')

        return data_pd['path'], data_pd['label'], list(zip(data_pd['gender'],data_pd['age'],data_pd['mask']))
```

팀원분이 만들어주신 new_standard.csv를 통해서 더 쉽게 dataset code를 구성할 수 있었다!

→ new_standard.csv : image path와 label data가 같이 들어있는 파일!
    
    

2. **target class distribution**
    - y를 18 classes로 변환한 후의 분포를 살펴보았다.

        → 이전에 살펴봤던 것으로 60대의 데이터가 매우 적었던 사실과 같이, 60대의 데이터(2,5,8,11,14,17)만 현저하게 적은 것을 알 수 있었다.

![Untitled](https://user-images.githubusercontent.com/70624819/131848409-713e8b08-045f-46f3-b4d6-50d8f9b4b58d.png)

1. **Data augmentation**

```python
# 일반 데이터
normal_data = MaskDataSet('input/data/train/train.csv','input/data/train/images')

# tensor로 전환 후 Resize
resized1_trsfm = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((224,224))])
resized1_data = MaskDataSet('input/data/train/train.csv','input/data/train/images',transform=resized1_trsfm)

# Resize 후 tensor로 전환
resized2_trsfm = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ToTensor()])
resized2_data = MaskDataSet('input/data/train/train.csv','input/data/train/images',transform=resized2_trsfm)

# tensor로 전환 후 normalize
normalized_trsfm = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(0,0.5)])
normalized_data = MaskDataSet('input/data/train/train.csv','input/data/train/images',transform=normalized_trsfm)

# 전체적인 transform
full_trsfm = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.Normalize(0,0.5),
    torchvision.transforms.ColorJitter(),
    torchvision.transforms.RandomCrop((112,112)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.GaussianBlur(kernel_size=(7,7)),
    torchvision.transforms.RandomErasing(),
])
full_data = MaskDataSet('input/data/train/train.csv','input/data/train/images',transform=full_trsfm)
```

```python
==================================================
# normal data
100%|█████████████| 148/148 [02:06<00:00,  1.17it/s]
==================================================
# resize 1 
100%|█████████████| 148/148 [01:50<00:00,  1.34it/s]
==================================================
# resize 2
100%|█████████████| 148/148 [02:18<00:00,  1.07it/s]
==================================================
# normalized data
100%|█████████████| 148/148 [02:08<00:00,  1.15it/s]
==================================================
# full transformed data
100%|█████████████| 148/148 [02:14<00:00,  1.10it/s]
```

- `ToTensor`를 적용한 후 나머지 transform을 해주는 것이 속도가 향상된다!
- transform을 할수록 연산량이 올라가지만 `resize`와 같은 연산은 오히려 전체 계산량을 줄여줘 시간을 단축시킬 수도 있다!

# 21-08-25 : Pre-trained Model

1. **DenseNet**

    처음에는 torchvision에서 pretrained densenet을 사용하였다.

    ```python
    class MaskDenseNet(nn.Module):
        def __init__(self,device='cpu'):
            super(MaskDenseNet,self).__init__()
            self.device=device
            self.densenet = torchvision.models.densenet161(pretrained=True).to(self.device)
            self.linear = nn.Linear(1000,18).to(self.device)
        
        def forward(self,x):
            x = self.densenet(x)
            x = nn.BatchNorm1d(1000, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).to(self.device)(x)
            return self.linear(x)
    ```

1. **DenseNet 3-head**

    age, gender, mask를 각각 학습한 후 마지막에 fully connected layer로 추론하는 방식의 3 head dense net을 사용하였다.

    ```python
    class MaskDenseNetThreeHead(nn.Module):
        def __init__(self,device):
    				...
            self.densenet_mask = torchvision.models.densenet161(pretrained=True).to(self.device)
            self.densenet_mask.classifier.register_forward_hook(self.mask_hook)

            self.densenet_gender = torchvision.models.densenet161(pretrained=True).to(self.device)
            self.densenet_gender.classifier.register_forward_hook(self.gender_hook)

            self.densenet_age = torchvision.models.densenet161(pretrained=True).to(self.device)
            self.densenet_age.classifier.register_forward_hook(self.age_hook)
    				...
            
        def forward(self,x):
            mask_class = self.densenet_mask.forward(x) # B x 3
            gender_class = self.densenet_gender.forward(x) # B x 2
            age_class = self.densenet_age.forward(x) # B x 3
            concat = torch.cat([mask_class,gender_class,age_class],dim=-1)
            return self.classifier.forward(concat)
    ```

    → 하지만 왠일인지 하나의 denseNet을 사용하는 것보다 성능이 잘 나오지 않았다.

# 21-08-26

어제의 Three-head DenseNet이 잘 되지 않아 이번에는 `FixResNext`라는 모델과 `ViT` 모델을 사용하였다.  FixResNet은 기존의 ResNext 모델에서 train과 test간의 해상도 불일치 문제를 해결하여 학습한 네트워크이다. 

> FixRes is a simple method for fixing the train-test resolution discrepancy. It can improve the performance of any convolutional neural network architecture.

5 epoch naive모델로 f1 0.6이 나왔다 → 학습 시간도 너무 오래 걸림...

그 외에 ViT를 이용해봤지만 5 epoch으로는 FixResNext에 미치지 못하는 성능이 나왔다. → 하지만 학습 시간은 FixResNet보다 훨씬 적게 걸렸다!

# 21-08-27

1. 첫째로 기존의 `FixResNext`모델에 `OneCycleLR`과 `albumentation argumentations`을 사용해서 수렴 속도를 개선시켜 빠르게 결과를 보려고 했다.(FixResNext의 경우 학습 속도가 매우 느렸기 때문에..)

    → albumentations argumentations의 경우 학습 시간이 epoch당 2~3분 정도 감소하였고, OnecycleLR을 통해 빠르게 수렴에 다가갈 수 있었다.

    → albumentations argumentations를 적용할 때, 기존에는 random crop이나 vertical flip 등도 사용했지만 오히려 성능이 하락해 color jitter, center crop과 horizontal flip 등만 사용하였다.

하지만 위의 방법을 적용해 학습을 늘린 결과 `overfitting`이 발생하게 되었다(train accuracy는 증가하는 반면, validation loss는 감소..)

→ 다른 방법들을 찾는 도중 `LabelSmoothingLoss`과 `FocalLoss`를 사용하여 generalization performance를 향상시키려 시도하였고 결과적으로 Focal Loss가 조금 더 높은 validation f1 score를 보여줘 Focal Loss를 사용하게 되었다. 

→ 팀원들과 공유한 결과, 이는 모델에 따라 다른 것 같다.

→ 이에 더해 `Dropout`도 적용해서 overfitting을 방지하고자 했다!

![Untitled](https://user-images.githubusercontent.com/70624819/131848410-61bed3db-71b6-497c-aab9-5e67bb06035b.png)

> 위와 같이 적용했을 때의 f1 score가 0.934를 달성했지만, 실제 LB score는 0.6x 정도 나왔다..
→ **generalization이 부족했거나! test dataset이 실제와 많은 차이를 보이거나!**

1. 두번째로`pseudo labeling ensemble`을 진행해봤다, 아직 k-fold를 하지 않아서 모델간의 편향성이 있을 것 같아 성능 향상이 안 되지 않을까 걱정했는데, 아니다 다를까 오히려 성능이 떨어졌다..

    → `stratifed K-fold`로 모델 간의 분포를 맞추고 다시 진행해보기!!

    → `stratified K-fold` 적용 후에도 성능 향상이 발생하지 않았다.. 

    → 모델간의 편향성 이전에 에초에 데이터의 표본이 너무 적어 학습이 제대로 되지 않는 것 같다.

    **→ 추가 데이터나 generalization performance를 올릴 수 있는 방법을 적용해야 할 것!**

# 21-08-30

우선 주말에는 기존의 FixResNext를 `EfficientNet`으로 바꿧다.

→ FixResNext의 경우 학습 시간이 너무 오래걸려 학습의 입장에서는 맞지 않는 것 같아 다른 여러 이론들을 쉽게 적용해보고 효과를 눈으로 볼 수 있도록 `EfficientNetLite`를 사용하게 되었다.

→ 정확도는 확실히 내려갔지만(lb f1 score 기준 0.6x → 0.5x or 0.6x) 학습 속도가 매우 향상되었다!!

1. `TTA(Test Time Argumentations)`, `OOF(Out-Of-Fold)`를 적용했는데 성능의 유의미한 향상이 없었다.

    → TTA 시에 `five crop`을 사용한 것이 오히려 성능의 하락을 나타낸 것 같다. **five crop을 제거한 후 다시 시도하니 성능이 약간 올랐다.**

    → Random crop과 비슷한 문제로 성능이 떨어진 것으로 생각된다.

    → 하지만 위에서의 문제인 불균형한 분포 때문에 큰 향상은 없는 것 같다.

2. `MultiDropout`과 `Cutmix & Mixup` 등을 사용해 최대한 generalization performance를 향상시키려고 하였다.

    결과 : 

    - Cutmix & Mixup 성능 하락 : Cutmix & Mixup의 경우 Cutmix되면서 얼굴 부분에 cutmit가 되면 완전히 잘못된 데이터를 학습하게 되면서 제대로 학습하지 못하고 성능이 하락하는 것같다.

        → 다음 예시와 같이 cut-mix되는 부분이 다음과 같이 되면 mix된 라벨을 완벽히 예상하는 것이 오히려 더 좋음에도 불구하고 값을 상당히 다르게 학습하게 된다!

        → Random crop과 Five crop도 이와 비슷한 맥락일 것으로 예상된다

    - MultoDropout 성능 향상 : Dropout을 5개로 늘리고 각 prob을 0.5로 주었을 때 성능이 향상되었다.

        → 그 후 여러 개를 늘렸는데 많이 늘린다고 좋아지는 것은 아닌 것 같다..

3. 추가 데이터 셋으로 학습!
    - 외부 데이터를 이용해 데이터를 추가학습함으로써 데이터 불균형을 줄이면서 전체적인 학습효과를 늘리려고 하였다.
    - Data1 : 기존의 일반 데이터 + 마스크 합성, Data2 : 외부 마스크 데이터 셋

    ![Untitled](https://user-images.githubusercontent.com/70624819/131848412-e7b57866-0dde-40aa-a5c9-a6d3b7a268f5.png)
    > **마스크 사진을 합성한 외부 데이터 셋**

# 21-08-31

1. pretrained된 모델 마지막에 classifier를 깊게 쌓으면 오히려 성능이 떨어진다! 예상하기로는 아에 학습이 되지 않은 네트워크가 너무 많이 쌓이면 pretrained된 네트워크에까지 좋지 않은 영향을 주는 것 같다..

    **→ 하지만 freezing을 천천히 진행하면서 학습한다면  학습 데이터가 충분하다면 포텐셜은 더 높지 않을까??**

2. 추가적으로 오늘은 기존에 jupyter notebook으로 하던 작업 파일을 IDE에서 `.py`로 변경하는 작업을 했다.

    → .py로 바꿔 모듈화하면 팀원과의 공유가 편하고 전반적인 관리 자체가 편해지는 것 같다! prototype은 jupyter로 어느 정도 완성된 후 버전 관리는 .py로 하는 것이 편해보임!

3. **그 외에 오늘은 위에서 팀원분들이 만들어주신 추가 데이터셋을 포함해서 학습 진행!!**

→ 팀원분이 만들어주신 추가 데이터셋을 기존의 데이터셋과 합쳐 stratified k-fold를 진행했다. 그런데 train loss를 잘 줄어드는 반면, validation loss를 줄지 않는 현상이 관측되었다..!

![Untitled](https://user-images.githubusercontent.com/70624819/131848414-6436313d-a01d-46a1-859b-c733b80b62b4.png)

알아본 결과, 이러한 현상은 `**validation set이 train set을 충분히 표현하지 못하는 경우**` 발생한다는 것을 알게 되었다. 하지만 첫번째 fold 외에 다른 fold에서는 이러한 현상이 발생하지 않고, 다른 팀원에게도 이러한 현상이 발생하지 않는 것으로 보아 내가 설정한 random seed값에서 우연치 않게 fold 1의 validation set과 train set의 차이가 발생한 것으로 예상된다 !!

![Untitled](https://user-images.githubusercontent.com/70624819/131848415-59f7ee7b-50b1-45b1-b856-23fe57fb8721.png)

> 하지만 생성된 train set이 각 데이터에 대해 서로 표현하지 못하는 케이스가 존재한다는 것을 알아두면 좋을 것!

# 21-09-01

1. 위에서의 데이터 분포를 조금이나마 바로잡기 위해서 기존의 데이터에서 불균형한 데이터들을 복사해 수를 비슷하게 맞춰준 후, argumentations을 더 강하게 진행해서 overiftting을 막고자 했다.

→ 처음에는 기존의 데이터를 복사한 것만 사용해서 학습을 진행해봤는데 역시나 validation loss와 f1 score는 매우 높게 나왔다. 이는 데이터를 여러 장 복사하면서 validation set의 내용이 train에서 학습되었기 때문!!

![Untitled](https://user-images.githubusercontent.com/70624819/131848417-6c9bb3df-7c1f-4adb-9519-fe1609e829d3.png)

![Untitled](https://user-images.githubusercontent.com/70624819/131848421-5aa648af-016c-44e4-aea5-07724371ccfc.png)

→ 하지만 실제 LB score는 복사하기 전 데이터 + 추가된 데이터를 학습한 것보다 더 낮게 나왔다.

→ 이를 통해 실제 데이터의 불균형도 중요하지만 더 중요한 것은 `**절대적인 데이터 샘플의 수`** 임을 알 수 있었다.

> 따라서 **복사한 데이터 + 추가된 데이터**를 통해 학습해서 불균형도 줄이고 절대적인 데이터 샘플의 수도 줄이는 방향으로 학습할 것이다!

# 21-09-02

1. 늦었지만 마지막 성능 향상을 위해 `efficientnetB4`를 사용하여 학습을 진행

    → `efficientnetLite0`보다 학습 포텐셜이 확실히 높다!

2. 팀원들의 모델을 이용해 `pseudo labeling`을 진행

    → 확실히 validation score가 많이 상승!

    → 하지만 실제로는 overfitting을 주의해야 한다!

3. 팀원들의 모델과 `ensemble` !!

    → `hard voting`을 진행했으나 단일 모델 최고 스코어보다 낮게 나왔다.

    → `stacking`을 진행, hard voting보다는 높은 스코어가 나왔지만 단일 모델 최고 스코어는 여전히 넘기지 못 했다..

    **→ ensemble은 다양한 모델의 결과값을 가지고 하는 작업이다보니 `generalization performance`를 강화시켜줄 수는 있지만 대회 스코어에서는 큰 향상이 발생하지 않을 수 있다..!**
