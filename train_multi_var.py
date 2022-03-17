import multiprocessing
from multiprocessing import freeze_support

import numpy as np
import pandas as pd

import torch
from gluonts.dataset.field_names import FieldName

from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import dataset_recipes, get_dataset
from gluonts.model.deepvar import DeepVAREstimator

from pts.model.deepar import DeepAREstimator
from pts.model.tempflow import TempFlowEstimator
from pts.model.transformer_tempflow import TransformerTempFlowEstimator
from pts import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import MultivariateEvaluator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from itertools import islice
from pathlib import Path
from gluonts.dataset.artificial import ComplexSeasonalTimeSeries
from gluonts.dataset.common import ListDataset
import warnings

warnings.filterwarnings("ignore")


def create_dataset(num_series, num_steps, period=24, mu=1, sigma=0.3):
    # create target: noise + pattern
    # noise
    noise = np.random.normal(mu, sigma, size=(num_series, num_steps))

    # pattern - sinusoid with different phase
    sin_minusPi_Pi = np.sin(np.tile(np.linspace(-np.pi, np.pi, period), int(num_steps / period)))
    sin_Zero_2Pi = np.sin(np.tile(np.linspace(0, 2 * np.pi, 24), int(num_steps / period)))

    pattern = np.concatenate(
        (
            np.tile(
                sin_minusPi_Pi.reshape(1, -1),
                (int(np.ceil(num_series / 2)), 1)
            ),
            np.tile(
                sin_Zero_2Pi.reshape(1, -1),
                (int(np.floor(num_series / 2)), 1)
            )
        ),
        axis=0
    )

    target = noise + pattern

    # create time features: use target one period earlier, append with zeros
    feat_dynamic_real = np.concatenate(
        (
            np.zeros((num_series, period)),
            target[:, :-period]
        ),
        axis=1
    )

    # create categorical static feats: use the sinusoid type as a categorical feature
    feat_static_cat = np.concatenate(
        (
            np.zeros(int(np.ceil(num_series / 2))),
            np.ones(int(np.floor(num_series / 2)))
        ),
        axis=0
    )

    return target, feat_dynamic_real, feat_static_cat


if __name__ == '__main__':
    freeze_support()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # artificial_dataset = ComplexSeasonalTimeSeries(
    #     num_series=10,  # 10组数据
    #     prediction_length=21,
    #     freq_str="H",
    #     length_low=30,
    #     length_high=200,
    #     min_val=-10000,
    #     max_val=10000,
    #     is_integer=False,
    #     proportion_missing_values=0,
    #     is_noise=True,
    #     is_scale=True,
    #     percentage_unique_timestamps=1,
    #     is_out_of_bounds_date=True,
    # )
    #
    # # print(f"prediction length: {artificial_dataset.metadata.prediction_length}")
    # # print(f"frequency: {artificial_dataset.metadata.freq}")
    # # print(f"type of train dataset: {artificial_dataset.train}")
    # # print(f"train dataset fields: {artificial_dataset.train[0].keys()}")
    # # print(f"type of test dataset: {type(artificial_dataset.test)}")
    # # print(f"test dataset fields: {artificial_dataset.test[0].keys()}")
    # train_ds = ListDataset(
    #     artificial_dataset.train,
    #     freq=artificial_dataset.metadata.freq,
    # )
    #
    # test_ds = ListDataset(
    #     artificial_dataset.test,
    #     freq=artificial_dataset.metadata.freq
    # )
    #
    # train_entry = next(iter(train_ds))
    # print(train_entry.keys())
    # define the parameters of the dataset
    custom_ds_metadata = {
        'num_series': 100,
        'num_steps': 24 * 7,
        'prediction_length': 24,
        'freq': '1H',
        'start': [
            pd.Timestamp("01-01-2019", freq='1H')
            for _ in range(100)
        ]
    }
    data_out = create_dataset(
        custom_ds_metadata['num_series'],
        custom_ds_metadata['num_steps'],
        custom_ds_metadata['prediction_length']
    )

    target, feat_dynamic_real, feat_static_cat = data_out

    train_ds = ListDataset(
        [
            {
                FieldName.TARGET: target,
                FieldName.START: start,
                FieldName.FEAT_DYNAMIC_REAL: [fdr],
                FieldName.FEAT_STATIC_CAT: [fsc]
            }
            for (target, start, fdr, fsc) in zip(
            target[:, :-custom_ds_metadata['prediction_length']],
            custom_ds_metadata['start'],
            feat_dynamic_real[:, :-custom_ds_metadata['prediction_length']],
            feat_static_cat
        )
        ],
        freq=custom_ds_metadata['freq']
    )
    test_ds = ListDataset(
        [
            {
                FieldName.TARGET: target,
                FieldName.START: start,
                FieldName.FEAT_DYNAMIC_REAL: [fdr],
                FieldName.FEAT_STATIC_CAT: [fsc]
            }
            for (target, start, fdr, fsc) in zip(
            target,
            custom_ds_metadata['start'],
            feat_dynamic_real,
            feat_static_cat)
        ],
        freq=custom_ds_metadata['freq']
    )
    print(train_ds.list_data)
    train_entry = next(iter(train_ds))
    print(train_entry.keys())
    test_entry = next(iter(test_ds))
    estimator = DeepVAREstimator(freq="5min",
                                 prediction_length=12,
                                 # input_size=19,
                                 target_dim=42,
                                 # trainer=Trainer(epochs=5,
                                 #                 batch_size=128,
                                 #                 # device=device,
                                 #                 )
                                 )
    num_worker = int(multiprocessing.cpu_count() / 2)
    predictor = estimator.train(training_data=train_ds, num_workers=num_worker)
    print("训练结束")
