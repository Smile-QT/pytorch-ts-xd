# -*- coding: utf-8 -*-
"""
@Time       : 2022/03/13 10:05
@Author     : Spring
@FileName   : train.py
@Description: 
"""
import datetime
import json
from pathlib import Path

import torch
from gluonts.dataset.multivariate_grouper import MultivariateGrouper

from gluonts.evaluation import make_evaluation_predictions, Evaluator
from matplotlib import pyplot as plt
from pts.model.lstnet import LSTNetEstimator

from pts.model.deepvar import DeepVAREstimator

from common import mkdir, get_custom_dataset, plot_prob_forecasts, plot_dataset
from pts import Trainer
from pts.model.deepar import DeepAREstimator
from pts.model.time_grad import TimeGradEstimator
import warnings

warnings.filterwarnings("ignore")
if __name__ == '__main__':
    from warnings import simplefilter

    simplefilter(action="ignore", category=FutureWarning)
    simplefilter(action="ignore", category=UserWarning)
    # 定义实验名称
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = "exp-" + current_time
    # 创建保存结果目录
    mkdir("results/" + exp_name)
    mkdir("models/" + exp_name)
    # 使用 cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import multiprocessing

    num_worker = int(multiprocessing.cpu_count() / 2)
    print("num_worker:", num_worker)
    # 获取自定义数据集
    train_data, test_data = get_custom_dataset(dataset_file="datasets/monthly-sunspots.csv", freq="5min",
                                               prediction_length=12,
                                               column="Sunspots")
    # 画数据集图
    plot_dataset(train_data, test_data, exp_name)
    # 定义模型

    estimator = DeepAREstimator(freq="5min",
                                prediction_length=12,
                                input_size=19,
                                trainer=Trainer(epochs=5,
                                                batch_size=128,
                                                device=device,
                                                )
                                )

    predictor = estimator.train(training_data=train_data, num_workers=num_worker)
    print("训练结束")
    # 保存模型
    predictor.serialize(Path("models/" + exp_name))
    # 开始预测
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_data,  # test dataset
        predictor=predictor,  # predictor
        num_samples=100,  # number of sample paths we want for evaluation
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)
    # 预测结果
    forecast_entry = forecasts[0]
    ts_entry = tss[0]
    # 画预测结果图
    plot_prob_forecasts(ts_entry, forecast_entry, exp_name)
    # 评价指标
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_data))
    metrics_data = json.dumps(agg_metrics, indent=4)

    print(metrics_data)
    with open("results/" + exp_name + "/metrics.json", 'w') as json_file:
        json_file.write(metrics_data + "\n")
    item_metrics.plot(x='MSIS', y='MASE', kind='scatter')
    plt.grid(which="both")
    plt.savefig("results/" + exp_name + "/metrics.jpg")
    # plt.show()
