import os
import shutil
import pandas as pd

# megaage_asian의 train, test 데이터 중 60대 이상 파일만 추출하여 마스크 이미지 생성

# 1.train_set
# Age, file_num을 column으로 가지는 dataframe 생성
data = pd.read_csv('./train_age.txt')
data['file_num']=list(range(1, 40001))

# 60대이상 파일번호 list
objt_idx = []

i = 0
while i < len(data):
    if data["Age"][i] >= 60:
        objt_idx.append(i + 1)
    i += 1

# 파일 복사(src_60)
src = './megaage_asian/train/'
dst = './fur_data/src_60/'

objt_dict = {}
for item in objt_idx:
    try:
        #mask 생성할 파일 복사
        shutil.copy(f'{src}{item}.jpg', dst)
        
        #'복사한 파일번호'와 '파일의 나이'를 dict로 저장
        num = int(data[data["file_num"]==item]["Age"])
        objt_dict[item] = num
    except:
        pass

# 2.test_set
# Age, file_num을 column으로 가지는 dataframe 생성
data2 = pd.read_csv('./test_age.txt')
data2['file_num']=list(range(1, 3946))

# 60대이상 파일번호 list
objt_idx2 = []

i = 0
while i < len(data2):
    if data2["Age"][i] >= 60:
        objt_idx2.append(i + 1)
    i += 1
    
# 파일 복사(src_60_2)
src2 = './megaage_asian/test/'
dst2 = './fur_data/src_60_2/'

objt_dict2 = {}
for item2 in objt_idx2:
    try:
        #mask 생성할 파일 복사
        shutil.copy(f'{src2}{item2}.jpg', dst2)
        
        #'복사한 파일번호'와 '파일의 나이'를 dict로 저장
        num = int(data2[data2["file_num"]==item2]["Age"])
        objt_dict2[item2] = num
    except:
        pass
