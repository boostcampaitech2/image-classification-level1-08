import os
import shutil
import pandas as pd

#디렉토리 경로설정
path = '../fur_data/fur_data/fur_data2/'

def createFolder(dir_name):
    try:
        if not os.path.exists(path + dir_name):
            os.makedirs(path + dir_name)
    except OSError:
        print ('Error: Creating directory. ' +  dir_name)

#src0 : no mask 경로, src1 : N95, surgical, cloth mask 경로
src0 = '../fur_data/All_Age_Faces_Dataset/original_images/'
src1 = '../fur_data/All_Age_Faces_Dataset/original_images_masked/'

#data setting
data1 = pd.read_csv('../fur_data/All_Age_Faces_Dataset/image sets/train.txt', sep=" ")
data2 = pd.read_csv('../fur_data/All_Age_Faces_Dataset/image sets/val.txt', sep=" ")

data1 = pd.DataFrame(data1)
data2 = pd.DataFrame(data2)

#basic columns
data1.columns = ["basic", "sex"]
data2.columns = ["basic", "sex"]

#columns = ID, sex, Age로 변경
# 1. data1
data1["ID"] = data1["basic"].str.split("A").str[0]
data1["Age_area"] = data1["basic"].str.split("A").str[1]
data1["Age"] = data1["Age_area"].str.split(".").str[0]
data1["sex"].replace(0,"f", inplace=True)
data1["sex"].replace(1,"0", inplace=True)
data1["sex"].replace("f",1, inplace=True)
data1_res = data1[["ID","sex","Age"]]

# 2. data2
data2["ID"] = data2["basic"].str.split("A").str[0]
data2["Age_area"] = data2["basic"].str.split("A").str[1]
data2["Age"] = data2["Age_area"].str.split(".").str[0]
data2["sex"].replace(0,"f", inplace=True)
data2["sex"].replace(1,"0", inplace=True)
data2["sex"].replace("f",1, inplace=True)
data2_res = data2[["ID","sex","Age"]]

data2_res.head(5)

#result data
fur2_res = pd.concat([data1_res, data2_res])
fur2_csv = fur2_res.sort_values(by='ID')
fur2_csv["ID"] = fur2_csv["ID"].astype(str)

#csv save
fur2_csv.to_csv(f'{path}fur2_info.csv', index=None)
fur2_csv.head(5)

#ID별 디렉토리 생성 및 파일 집합
objt_idx = list(fur2_csv["ID"])

for item in objt_idx:
    #age
    age = int(fur2_csv[fur2_csv["ID"]==item]["Age"])
    if age < 10:
        age = '0' + str(age)
    else:
        age = str(age)

    #sex
    if int(fur2_csv[fur2_csv["ID"]==item]["sex"]) == 0:
        sex = "male"
    else:
        sex = "female"
    
    #dir name
    dir_name = f'{item}_{sex}_{age}'
    createFolder(dir_name)
    try:
        shutil.copy(f'{src0}{item}A{age}.jpg', f'{path}{dir_name}/normal.jpg')
        shutil.copy(f'{src1}{item}A{age}_N95.jpg', f'{path}{dir_name}/mask1.jpg')
        shutil.copy(f'{src1}{item}A{age}_surgical.jpg', f'{path}{dir_name}/mask2.jpg')
        shutil.copy(f'{src1}{item}A{age}_cloth.jpg', f'{path}{dir_name}/mask3.jpg')
    except:
        pass
