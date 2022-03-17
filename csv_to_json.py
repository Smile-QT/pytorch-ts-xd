# -*- coding: utf-8 -*-
"""
@Time       : 2022/03/11 21:00
@Author     : Spring
@FileName   : csv_to_json.py
@Description: 
"""
import json
import time
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Any, Dict, Optional

from gluonts.dataset.common import load_datasets, ListDataset


def read_csv(filename):
    df = pd.read_csv(filename, encoding="utf-8")
    # for i in range(len(df)):
    #     print(df.loc[i])
    # print(df)
    start = df["时间"].tolist()
    target = [df["瞬时风速"].tolist(), df["瞬时风向"].tolist(), df["有功功率"].tolist()]
    target = list(map(list, zip(*target)))
    return start, target


def set_ftime(i):
    """
    # 2021/9/21 23:20->2000-01-01 00:00:00
    """
    timeArray = time.strptime(i + ":00", "%Y/%m/%d %H:%M:%S")
    return time.strftime("%Y-%m-%d %H:%M:%S", timeArray)


# dataset_names = list(dataset_recipes.keys())

# default_dataset_path = get_download_path() / "datasets"
default_dataset_path = ""


def get_my_dataset(dataset_path):
    print(dataset_path)
    return load_datasets(
        metadata=dataset_path,
        train=dataset_path + "/train",
        test=dataset_path + "/test",
    )


if __name__ == "__main__":
    # 指定的文件存在
    start, target = read_csv("datasets/in_min.csv")
    for i in range(len(start)):
        data = json.dumps({"start": set_ftime(start[i]), "target": target[i], "item_id": "T" + str(i)})
        with open('datasets/data.json', 'a') as json_file:
            json_file.write(data + '\n')
