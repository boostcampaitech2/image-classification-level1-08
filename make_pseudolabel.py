import pandas as pd

df = pd.read_csv("./submission.csv")

train = pd.read_csv("./standard.csv")

new_train = pd.DataFrame()
new_train["path"] = "../input/data/eval/images/" + df["ImageID"]
new_train["label"] = df["ans"]



all_data = pd.concat([train, new_train], axis=0, ignore_index=True, sort=False)
all_data.to_csv("./pseudo_dataset.csv", index=False)