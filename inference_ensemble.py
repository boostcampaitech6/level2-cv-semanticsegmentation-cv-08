import os
import torch
from tqdm.auto import tqdm
import torch.nn.functional as F
import albumentations as A
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from modules import utils

from models.models import get_model

from modules.utils import load_yaml, save_yaml, encode_mask_to_rle, decode_rle_to_mask
from modules.dataset import XRayInferenceDataset
from modules.transforms import get_transform_function

import random, os, sys
import numpy as np
from datetime import datetime


prj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(prj_dir)

pred_serial = datetime.now().strftime("%Y%m%d_%H%M%S")

# result_dir
pred_result_dir = os.path.join(prj_dir, "results", "pred", f"ensemble_{pred_serial}")
os.makedirs(pred_result_dir, exist_ok=True)
dfs = []

csv_folder_path = os.path.join(prj_dir, "results", "ensemble")
for idx, csv_path in enumerate(os.listdir(csv_folder_path)):
    test_df = pd.read_csv(os.path.join(csv_folder_path, csv_path))
    dfs.append(test_df)

image_names = dfs[0]["image_name"]
classes = dfs[0]["class"]
rles = []
for idx in range(len(dfs[0])):
    masks = [decode_rle_to_mask(str(df["rle"][idx]), 2048, 2048) for df in dfs]
    ensemble_mask = np.stack(masks).mean(axis=0)
    ensemble_mask = np.where(ensemble_mask > 0.45, 1, 0)
    rle = encode_mask_to_rle(ensemble_mask)
    rles.append(rle)

df = pd.DataFrame(
    {"image_name": dfs[0]["image_name"], "class": dfs[0]["class"], "rle": rles}
)
df.to_csv(os.path.join(pred_result_dir, "output.csv"), index=False)
print("saved~!")
