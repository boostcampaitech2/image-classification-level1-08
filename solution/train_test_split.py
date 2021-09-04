import pandas as pd
import csv
import numpy as np
from numpy.random import RandomState

# df = pd.read_csv('./csvs/age_extra/extra_age_train.csv')
# df = pd.read_csv('./csvs/gender_extra/extra_gender_train.csv')
df = pd.read_csv('./csvs/mask_extra/extra_mask_train.csv')

rng = RandomState()

train = df.sample(frac=0.7, random_state=rng)
test = df.loc[~df.index.isin(train.index)]

test.to_csv("./csvs/mask_extra/test.csv",index=None)
train.to_csv("./csvs/mask_extra/train.csv",index=None)