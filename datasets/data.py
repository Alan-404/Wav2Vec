#%%
import pandas as pd
import io
# %%
data = io.open('./prompts.txt').read().strip().split("\n")
# %%
data
# %%
base = '/home/alan/datasets/vivos/train/waves'
# %%
def create_path(name: str):
    folder = name.split("_")[0]
    return f"{base}/{folder}/{name}.wav"
# %%
data
# %%
paths = []
texts = []
for item in data:
    arr = item.split(" ")
    name = arr[0]
    text = " ".join(arr[1:])

    paths.append(create_path(name))
    texts.append(text)
# %%
pd.DataFrame({
    'path': paths,
    'text': texts
}).to_csv('./vivos-train.csv', sep="\t", index=False)
# %%
