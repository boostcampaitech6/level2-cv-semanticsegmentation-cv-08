import os
import torch
import json
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

test_serial = "20240222_132454_20240222_153241"
df = pd.read_csv(os.path.join(prj_dir, "results", "pred", test_serial, "output.csv"))

data_path = os.path.join(prj_dir, "data", "test", "DCM")
folders = sorted(os.listdir(data_path))
folders = [x for x in folders for _ in range(2)]
pngs = {
    os.path.relpath(os.path.join(root, fname), start=data_path)
    for root, _dirs, files in os.walk(data_path)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
}
pngs = sorted(pngs)

for folder, image in zip(folders, pngs):
    df_image = df[df["image_name"] == image[6:]]
    folder_path = os.path.join(prj_dir, "data", "all", "output_json", folder)
    print(folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    annotations = []
    for idx in range(len(df)):
        label = df["class"][idx]
        rle = df["rle"][idx]

        points = []
        s = rle.split()
        for start, length in zip(s[0:][::2], s[1:][::2]):
            row = int(start) // 2048
            col = int(start) % 2048

            for i in range(int(length)):
                if col > 2023:
                    col = 0
                    row += 1
                points.append([row, col])
                col += 1
        annotations.append({"points": points, "label": label})
    with open(os.path.join(folder_path, f"{image[6:-4]}.json"), "w") as f:
        json.dump({"annotations": annotations}, f)
