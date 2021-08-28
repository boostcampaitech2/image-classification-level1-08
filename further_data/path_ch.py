#생성한 마스크파일을 대회 데이터 형식으로 묶기
#디렉토리이름: ID_성별_인종_나이
#파일명 : normal, mask1, ... ,incorrect

import os
import shutil
import pandas as pd

#path = 디렉토리 저장 경로
path = './fin/'

def createFolder(dir_name):
    try:
        if not os.path.exists(path + dir_name):
            os.makedirs(path + dir_name)
    except OSError:
        print ('Error: Creating directory. ' +  dir_name)

# 1.src_60       
#src0: no mask경로, scr1: n95 mask 경로, src2: surgical mask 경로, src3: cloth mask 경로
src0 = './src_60/'
src1 = './masked_n95/src_60_masked/'
src2 = './masked_sugical/src_60_masked/'
src3 = './masked_color/src_60_masked/'

i = 0
for item in objt_idx:
    i += 1
    age = objt_dict[item]
    dir_name = f'{i}_{age}'
    createFolder(dir_name)
    try:
        shutil.copy(f'{src0}{item}.jpg', f'{path}{dir_name}/normal.jpg')
        shutil.copy(f'{src1}{item}_N95.jpg', f'{path}{dir_name}/mask1.jpg')
        shutil.copy(f'{src2}{item}_surgical.jpg', f'{path}{dir_name}/mask2.jpg')
        shutil.copy(f'{src3}{item}_cloth.jpg', f'{path}{dir_name}/mask3.jpg')
    except:
        pass
print(i)
#마지막번호 == 2099

# 2.src_60_2
#src0: no mask경로, scr1: n95 mask 경로, src2: surgical mask 경로, src3: cloth mask 경로
src0 = './src_60_2/'
src1 = './masked_n95/src_60_2_masked/'
src2 = './masked_sugical/src_60_2_masked/'
src3 = './masked_color/src_60_2_masked/'

i = 2099
for item2 in objt_idx2:
    i += 1
    age = objt_dict2[item2]
    dir_name = f'{i}_{age}'
    createFolder(dir_name)
    try:
        shutil.copy(f'{src0}{item2}.jpg', f'{path}{dir_name}/normal.jpg')
        shutil.copy(f'{src1}{item2}_N95.jpg', f'{path}{dir_name}/mask1.jpg')
        shutil.copy(f'{src2}{item2}_surgical.jpg', f'{path}{dir_name}/mask2.jpg')
        shutil.copy(f'{src3}{item2}_cloth.jpg', f'{path}{dir_name}/mask3.jpg')
    except:
        pass

#ID, age 정보를 dataframe으로 생성
fur_csv = pd.DataFrame(columns=['ID','Age'])
dir_names = os.listdir(path)

for dir in dir_names:
    try:
        ID, AGE = dir.split('_')
        fur_csv = fur_csv.append({'ID':int(ID), 'Age':AGE}, ignore_index=True)
    except:
        pass

fur_csv = fur_csv.sort_values(by='ID')
fur_csv.head(5)

#csv save(index=none)
fur_csv.to_csv(f'{path}/fur_age.csv', index=None)
