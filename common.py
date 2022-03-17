# -*- coding: utf-8 -*-
"""
@Time       : 2022/03/13 11:21
@Author     : Spring
@FileName   : common.py
@Description: 
"""
import pandas as pd

from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName

from gluonts.dataset.util import to_pandas
from matplotlib import pyplot as plt


def plot_prob_forecasts(ts_entry, forecast_entry, exp_name="exp"):
    """
    画预测图像
    :param ts_entry:
    :param forecast_entry:
    :return:
    """
    plot_length = 150
    prediction_intervals = (50.0, 90.0)
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
    forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.savefig("results/" + exp_name + "/forecast.jpg")
    # plt.show()


def get_custom_dataset(dataset_file, freq, prediction_length, column):
    """
    获取自定义 csv 数据集
    :param dataset_file:
    :param freq:
    :param prediction_length:
    :param column:
    :return:
    """
    pd_csv = pd.read_csv(dataset_file, header=0, index_col=0, parse_dates=True)
    fields = pd_csv.columns.values
    print(fields)
    # 自定义数据集转换
    train_data = ListDataset(

        [{"start": pd_csv.index[0], "target": pd_csv[column][:-prediction_length]}],
        freq=freq
    )
    test_data = ListDataset(
        [{"start": pd_csv.index[0], "target": pd_csv[column][:]}],
        freq=freq
    )
    return train_data, test_data


def get_multi_var_custom_dataset(dataset_file, freq, prediction_length, ):
    """
    获取自定义 csv 数据集
    :param dataset_file:
    :param freq:
    :param prediction_length:
    :param column:
    :return:
    """
    pd_csv = pd.read_csv(dataset_file, header=0, index_col=0, parse_dates=True)
    # pd_csv = pd.read_table(dataset_file, sep=",", index_col=0, parse_dates=True, decimal=',')
    fields = pd_csv.columns.values
    print(fields)
    pd_csv = pd_csv.T
    print(pd_csv)
    # ts_code = pd_csv["index"].astype("category").cat.codes.values
    ts_code = [[i] for i in range(len(fields))]
    # print(ts_code)
    num_series = 3
    start_train = pd.Timestamp("2021-01-01 00:00:00", freq=freq)
    data = [row.tolist() for _, row in pd_csv.iterrows()]

    # :
    # print(index)  # 输出每行的索引值
    # print(row.tolist())
    # 自定义数据集转换
    train_data = ListDataset([
        {
            FieldName.TARGET: target,
            FieldName.START: start_train,
            FieldName.FEAT_STATIC_CAT: fsc
        } for (target, fsc) in zip(data, ts_code[0:num_series])
    ], freq=freq)
    # train_data = ListDataset(
    #     [{
    #         "start": pd_csv.index[0],
    #         "target": pd_csv[column][:-prediction_length],
    #         "feat_static_cat": ts_code[0:num_series]
    #     }],
    #     freq=freq
    # )
    # print(train_data.list_data)
    # test_data = ListDataset(
    #     [{"start": pd_csv.index[0], "target": pd_csv[column][:]}],
    #     freq=freq
    # )
    test_data = []
    return train_data, test_data


def plot_dataset(train_entry, test_entry, exp_name="exp", is_train=True):
    """
    画原始数据图
    :param exp_name:
    :param data:
    :return:
    """
    test_series = to_pandas(next(iter(test_entry)))
    train_series = to_pandas(next(iter(train_entry)))

    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 7))

    train_series.plot(ax=ax[0])
    ax[0].grid(which="both")
    ax[0].legend(["train series"], loc="upper left")

    test_series.plot(ax=ax[1])
    ax[1].axvline(train_series.index[-1], color='r')  # end of train dataset
    ax[1].grid(which="both")
    ax[1].legend(["test series", "end of train series"], loc="upper left")
    plt.savefig("results/" + exp_name + "/train_test.jpg")


def mkdir(path):
    """
    创建文件夹
    :param path:
    :return:
    """
    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)

        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False
