# скрыть предупреждения от библиотеки tensorflow
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import conn
import statistics
import pandas as pd
import numpy as np
import datetime
from keras.models import Sequential
from keras.layers import Dense, Activation
from pca import pca
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from keras import models
from plotly.subplots import make_subplots

""" Метод главных компонент """


def startPcaAnalysis(data):
    pcamod = pca(n_components=0.95)
    result = pcamod.fit_transform(data)
    fig, ax = pcamod.plot()
    fig.show()

    tr = result['outliers'].y_bool
    topfeat = result['topfeat']
    return topfeat


""" Нормализация данных """


def normalizeData(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()

        if max_value != min_value:
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
        else:
            del result[feature_name]

    return result


""" Нормализация выходных проверочных данных """


def normalizeTestOutputData(df):
    result = df.copy()
    denorm = []
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()

        if max_value != min_value:
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
            d = [min_value, max_value]
            denorm.append(d)
        else:
            result[feature_name] = 0
            d = [min_value, max_value]
            denorm.append(d)
    return result, denorm


""" приведение данных из относительных значений в абсолютные"""


def denormalizeData(df, denorm):
    result = df.copy()
    i = 0
    for column in df.columns:
        result[column] = df[column] * (int(denorm[i][1]) - int(denorm[i][0])) + int(denorm[i][0])
        i = i + 1
    return result


""" Конветируем полученный массив из бд в массив numpy """


def convertData(data):
    result = np.zeros((len(data), len(data[0]) - 2))
    names = []
    years = []

    for i in range(len(data)):
        j = 0
        for el in data[i]:
            if i == 0 and el != "id" and el != "year":
                names.append(el)
            if el == "year":
                years.append(data[i][el])

            if el != "id" and el != "year":
                result[i][j - 2] = data[i][el]

            j += 1

    resultDF = pd.DataFrame(result, index=years, columns=names)

    return resultDF


""" получение статистических величин использующихся при конвертировании признаков базы данных """


# возвращает моду массива
def getStatisticModa(elems):
    # приводим к целым числам
    elems[:] = (round(i) for i in elems)

    elem_counts = {}
    for e in elems:
        if e not in elem_counts:
            elem_counts[e] = 1
        else:
            elem_counts[e] += 1

    # Проходимся по словарю и ищем максимальное количество повторений
    maxp = 0
    mode_elem = None
    for k, v in elem_counts.items():
        if maxp < v:
            maxp = v
            mode_elem = k
    return mode_elem


# возвращает среднее массива
def getStatisticMean(elems):
    sum = count = 0
    for elem in elems:
        sum = sum + elem
        count = count + 1

    mean_elem = sum / count

    return mean_elem


# возвращает дисперсионный разброс массива
def getStatisticDisp(elems):
    diffs = 0
    avg = sum(elems) / len(elems)
    for n in elems:
        diffs += (n - avg) ** (2)
    return (diffs / (len(elems) - 1)) ** (0.5)


def transpoteAllInputData():
    connection = conn.getConnection()
    cursor = connection.cursor()

    years = [2006, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

    big_query = "INSERT INTO `input_data` (`id`, `year`, " \
                "`level_min_1`, `level_max_1`, `level_mean_1`, `level_median_1`, `level_disp_1`, `level_moda_1`, `T_min_1`, `T_max_1`, `T_mean_1`, `T_median_1`, `T_disp_1`, `T_moda_1`, `P0_min_1`, `P0_max_1`, `P0_mean_1`, `P0_median_1`, `P0_disp_1`, `P0_moda_1`, `P_min_1`, `P_max_1`, `P_mean_1`, `P_median_1`, `P_disp_1`, `P_moda_1`, `U_min_1`, `U_max_1`, `U_mean_1`, `U_median_1`, `U_disp_1`, `U_moda_1`, `N_min_1`, `N_max_1`, `N_mean_1`, `N_median_1`, `N_disp_1`, `N_moda_1`, `Tmin_min_1`, `Tmin_max_1`, `Tmin_mean_1`, `Tmin_median_1`, `Tmin_disp_1`, `Tmin_moda_1`, `Tmax_min_1`, `Tmax_max_1`, `Tmax_mean_1`, `Tmax_median_1`, `Tmax_disp_1`, `Tmax_moda_1`, `Nh_min_1`, `Nh_max_1`, `Nh_mean_1`, `Nh_median_1`, `Nh_disp_1`, `Nh_moda_1`, `RRR_max_1`, `RRR_mean_1`, `RRR_median_1`, `RRR_disp_1`, `RRR_moda_1`, `sss_min_1`, `sss_max_1`, `sss_mean_1`, `sss_median_1`, `sss_disp_1`, `sss_moda_1`, " + \
                "`level_min_2`, `level_max_2`, `level_mean_2`, `level_median_2`, `level_disp_2`, `level_moda_2`, `T_min_2`, `T_max_2`, `T_mean_2`, `T_median_2`, `T_disp_2`, `T_moda_2`, `P0_min_2`, `P0_max_2`, `P0_mean_2`, `P0_median_2`, `P0_disp_2`, `P0_moda_2`, `P_min_2`, `P_max_2`, `P_mean_2`, `P_median_2`, `P_disp_2`, `P_moda_2`, `U_min_2`, `U_max_2`, `U_mean_2`, `U_median_2`, `U_disp_2`, `U_moda_2`, `N_min_2`, `N_max_2`, `N_mean_2`, `N_median_2`, `N_disp_2`, `N_moda_2`, `Tmin_min_2`, `Tmin_max_2`, `Tmin_mean_2`, `Tmin_median_2`, `Tmin_disp_2`, `Tmin_moda_2`, `Tmax_min_2`, `Tmax_max_2`, `Tmax_mean_2`, `Tmax_median_2`, `Tmax_disp_2`, `Tmax_moda_2`, `Nh_min_2`, `Nh_max_2`, `Nh_mean_2`, `Nh_median_2`, `Nh_disp_2`, `Nh_moda_2`, `RRR_max_2`, `RRR_mean_2`, `RRR_median_2`, `RRR_disp_2`, `RRR_moda_2`, `sss_min_2`, `sss_max_2`, `sss_mean_2`, `sss_median_2`, `sss_disp_2`, `sss_moda_2`, " + \
                "`level_min_3`, `level_max_3`, `level_mean_3`, `level_median_3`, `level_disp_3`, `level_moda_3`, `T_min_3`, `T_max_3`, `T_mean_3`, `T_median_3`, `T_disp_3`, `T_moda_3`, `P0_min_3`, `P0_max_3`, `P0_mean_3`, `P0_median_3`, `P0_disp_3`, `P0_moda_3`, `P_min_3`, `P_max_3`, `P_mean_3`, `P_median_3`, `P_disp_3`, `P_moda_3`, `U_min_3`, `U_max_3`, `U_mean_3`, `U_median_3`, `U_disp_3`, `U_moda_3`, `N_min_3`, `N_max_3`, `N_mean_3`, `N_median_3`, `N_disp_3`, `N_moda_3`, `Tmin_min_3`, `Tmin_max_3`, `Tmin_mean_3`, `Tmin_median_3`, `Tmin_disp_3`, `Tmin_moda_3`, `Tmax_min_3`, `Tmax_max_3`, `Tmax_mean_3`, `Tmax_median_3`, `Tmax_disp_3`, `Tmax_moda_3`, `Nh_min_3`, `Nh_max_3`, `Nh_mean_3`, `Nh_median_3`, `Nh_disp_3`, `Nh_moda_3`, `RRR_max_3`, `RRR_mean_3`, `RRR_median_3`, `RRR_disp_3`, `RRR_moda_3`, `sss_min_3`, `sss_max_3`, `sss_mean_3`, `sss_median_3`, `sss_disp_3`, `sss_moda_3`, " + \
                "`level_min_4`, `level_max_4`, `level_mean_4`, `level_median_4`, `level_disp_4`, `level_moda_4`, `T_min_4`, `T_max_4`, `T_mean_4`, `T_median_4`, `T_disp_4`, `T_moda_4`, `P0_min_4`, `P0_max_4`, `P0_mean_4`, `P0_median_4`, `P0_disp_4`, `P0_moda_4`, `P_min_4`, `P_max_4`, `P_mean_4`, `P_median_4`, `P_disp_4`, `P_moda_4`, `U_min_4`, `U_max_4`, `U_mean_4`, `U_median_4`, `U_disp_4`, `U_moda_4`, `N_min_4`, `N_max_4`, `N_mean_4`, `N_median_4`, `N_disp_4`, `N_moda_4`, `Tmin_min_4`, `Tmin_max_4`, `Tmin_mean_4`, `Tmin_median_4`, `Tmin_disp_4`, `Tmin_moda_4`, `Tmax_min_4`, `Tmax_max_4`, `Tmax_mean_4`, `Tmax_median_4`, `Tmax_disp_4`, `Tmax_moda_4`, `Nh_min_4`, `Nh_max_4`, `Nh_mean_4`, `Nh_median_4`, `Nh_disp_4`, `Nh_moda_4`, `RRR_max_4`, `RRR_mean_4`, `RRR_median_4`, `RRR_disp_4`, `RRR_moda_4`, `sss_min_4`, `sss_max_4`, `sss_mean_4`, `sss_median_4`, `sss_disp_4`, `sss_moda_4`, " + \
                "`level_min_5`, `level_max_5`, `level_mean_5`, `level_median_5`, `level_disp_5`, `level_moda_5`, `T_min_5`, `T_max_5`, `T_mean_5`, `T_median_5`, `T_disp_5`, `T_moda_5`, `P0_min_5`, `P0_max_5`, `P0_mean_5`, `P0_median_5`, `P0_disp_5`, `P0_moda_5`, `P_min_5`, `P_max_5`, `P_mean_5`, `P_median_5`, `P_disp_5`, `P_moda_5`, `U_min_5`, `U_max_5`, `U_mean_5`, `U_median_5`, `U_disp_5`, `U_moda_5`, `N_min_5`, `N_max_5`, `N_mean_5`, `N_median_5`, `N_disp_5`, `N_moda_5`, `Tmin_min_5`, `Tmin_max_5`, `Tmin_mean_5`, `Tmin_median_5`, `Tmin_disp_5`, `Tmin_moda_5`, `Tmax_min_5`, `Tmax_max_5`, `Tmax_mean_5`, `Tmax_median_5`, `Tmax_disp_5`, `Tmax_moda_5`, `Nh_min_5`, `Nh_max_5`, `Nh_mean_5`, `Nh_median_5`, `Nh_disp_5`, `Nh_moda_5`, `RRR_max_5`, `RRR_mean_5`, `RRR_median_5`, `RRR_disp_5`, `RRR_moda_5`, `sss_min_5`, `sss_max_5`, `sss_mean_5`, `sss_median_5`, `sss_disp_5`, `sss_moda_5`, " + \
                "`level_min_6`, `level_max_6`, `level_mean_6`, `level_median_6`, `level_disp_6`, `level_moda_6`, `T_min_6`, `T_max_6`, `T_mean_6`, `T_median_6`, `T_disp_6`, `T_moda_6`, `P0_min_6`, `P0_max_6`, `P0_mean_6`, `P0_median_6`, `P0_disp_6`, `P0_moda_6`, `P_min_6`, `P_max_6`, `P_mean_6`, `P_median_6`, `P_disp_6`, `P_moda_6`, `U_min_6`, `U_max_6`, `U_mean_6`, `U_median_6`, `U_disp_6`, `U_moda_6`, `N_min_6`, `N_max_6`, `N_mean_6`, `N_median_6`, `N_disp_6`, `N_moda_6`, `Tmin_min_6`, `Tmin_max_6`, `Tmin_mean_6`, `Tmin_median_6`, `Tmin_disp_6`, `Tmin_moda_6`, `Tmax_min_6`, `Tmax_max_6`, `Tmax_mean_6`, `Tmax_median_6`, `Tmax_disp_6`, `Tmax_moda_6`, `Nh_min_6`, `Nh_max_6`, `Nh_mean_6`, `Nh_median_6`, `Nh_disp_6`, `Nh_moda_6`, `RRR_max_6`, `RRR_mean_6`, `RRR_median_6`, `RRR_disp_6`, `RRR_moda_6`, `sss_min_6`, `sss_max_6`, `sss_mean_6`, `sss_median_6`, `sss_disp_6`, `sss_moda_6`, " + \
                "`level_min_7`, `level_max_7`, `level_mean_7`, `level_median_7`, `level_disp_7`, `level_moda_7`, `T_min_7`, `T_max_7`, `T_mean_7`, `T_median_7`, `T_disp_7`, `T_moda_7`, `P0_min_7`, `P0_max_7`, `P0_mean_7`, `P0_median_7`, `P0_disp_7`, `P0_moda_7`, `P_min_7`, `P_max_7`, `P_mean_7`, `P_median_7`, `P_disp_7`, `P_moda_7`, `U_min_7`, `U_max_7`, `U_mean_7`, `U_median_7`, `U_disp_7`, `U_moda_7`, `N_min_7`, `N_max_7`, `N_mean_7`, `N_median_7`, `N_disp_7`, `N_moda_7`, `Tmin_min_7`, `Tmin_max_7`, `Tmin_mean_7`, `Tmin_median_7`, `Tmin_disp_7`, `Tmin_moda_7`, `Tmax_min_7`, `Tmax_max_7`, `Tmax_mean_7`, `Tmax_median_7`, `Tmax_disp_7`, `Tmax_moda_7`, `Nh_min_7`, `Nh_max_7`, `Nh_mean_7`, `Nh_median_7`, `Nh_disp_7`, `Nh_moda_7`, `RRR_max_7`, `RRR_mean_7`, `RRR_median_7`, `RRR_disp_7`, `RRR_moda_7`, `sss_min_7`, `sss_max_7`, `sss_mean_7`, `sss_median_7`, `sss_disp_7`, `sss_moda_7`, " + \
                "`level_min_8`, `level_max_8`, `level_mean_8`, `level_median_8`, `level_disp_8`, `level_moda_8`, `T_min_8`, `T_max_8`, `T_mean_8`, `T_median_8`, `T_disp_8`, `T_moda_8`, `P0_min_8`, `P0_max_8`, `P0_mean_8`, `P0_median_8`, `P0_disp_8`, `P0_moda_8`, `P_min_8`, `P_max_8`, `P_mean_8`, `P_median_8`, `P_disp_8`, `P_moda_8`, `U_min_8`, `U_max_8`, `U_mean_8`, `U_median_8`, `U_disp_8`, `U_moda_8`, `N_min_8`, `N_max_8`, `N_mean_8`, `N_median_8`, `N_disp_8`, `N_moda_8`, `Tmin_min_8`, `Tmin_max_8`, `Tmin_mean_8`, `Tmin_median_8`, `Tmin_disp_8`, `Tmin_moda_8`, `Tmax_min_8`, `Tmax_max_8`, `Tmax_mean_8`, `Tmax_median_8`, `Tmax_disp_8`, `Tmax_moda_8`, `Nh_min_8`, `Nh_max_8`, `Nh_mean_8`, `Nh_median_8`, `Nh_disp_8`, `Nh_moda_8`, `RRR_max_8`, `RRR_mean_8`, `RRR_median_8`, `RRR_disp_8`, `RRR_moda_8`, `sss_min_8`, `sss_max_8`, `sss_mean_8`, `sss_median_8`, `sss_disp_8`, `sss_moda_8`, " + \
                "`level_min_9`, `level_max_9`, `level_mean_9`, `level_median_9`, `level_disp_9`, `level_moda_9`, `T_min_9`, `T_max_9`, `T_mean_9`, `T_median_9`, `T_disp_9`, `T_moda_9`, `P0_min_9`, `P0_max_9`, `P0_mean_9`, `P0_median_9`, `P0_disp_9`, `P0_moda_9`, `P_min_9`, `P_max_9`, `P_mean_9`, `P_median_9`, `P_disp_9`, `P_moda_9`, `U_min_9`, `U_max_9`, `U_mean_9`, `U_median_9`, `U_disp_9`, `U_moda_9`, `N_min_9`, `N_max_9`, `N_mean_9`, `N_median_9`, `N_disp_9`, `N_moda_9`, `Tmin_min_9`, `Tmin_max_9`, `Tmin_mean_9`, `Tmin_median_9`, `Tmin_disp_9`, `Tmin_moda_9`, `Tmax_min_9`, `Tmax_max_9`, `Tmax_mean_9`, `Tmax_median_9`, `Tmax_disp_9`, `Tmax_moda_9`, `Nh_min_9`, `Nh_max_9`, `Nh_mean_9`, `Nh_median_9`, `Nh_disp_9`, `Nh_moda_9`, `RRR_max_9`, `RRR_mean_9`, `RRR_median_9`, `RRR_disp_9`, `RRR_moda_9`, `sss_min_9`, `sss_max_9`, `sss_mean_9`, `sss_median_9`, `sss_disp_9`, `sss_moda_9`, " + \
                "`level_min_10`, `level_max_10`, `level_mean_10`, `level_median_10`, `level_disp_10`, `level_moda_10`, `T_min_10`, `T_max_10`, `T_mean_10`, `T_median_10`, `T_disp_10`, `T_moda_10`, `P0_min_10`, `P0_max_10`, `P0_mean_10`, `P0_median_10`, `P0_disp_10`, `P0_moda_10`, `P_min_10`, `P_max_10`, `P_mean_10`, `P_median_10`, `P_disp_10`, `P_moda_10`, `U_min_10`, `U_max_10`, `U_mean_10`, `U_median_10`, `U_disp_10`, `U_moda_10`, `N_min_10`, `N_max_10`, `N_mean_10`, `N_median_10`, `N_disp_10`, `N_moda_10`, `Tmin_min_10`, `Tmin_max_10`, `Tmin_mean_10`, `Tmin_median_10`, `Tmin_disp_10`, `Tmin_moda_10`, `Tmax_min_10`, `Tmax_max_10`, `Tmax_mean_10`, `Tmax_median_10`, `Tmax_disp_10`, `Tmax_moda_10`, `Nh_min_10`, `Nh_max_10`, `Nh_mean_10`, `Nh_median_10`, `Nh_disp_10`, `Nh_moda_10`, `RRR_max_10`, `RRR_mean_10`, `RRR_median_10`, `RRR_disp_10`, `RRR_moda_10`, `sss_min_10`, `sss_max_10`, `sss_mean_10`, `sss_median_10`, `sss_disp_10`, `sss_moda_10`, " + \
                "`level_min_11`, `level_max_11`, `level_mean_11`, `level_median_11`, `level_disp_11`, `level_moda_11`, `T_min_11`, `T_max_11`, `T_mean_11`, `T_median_11`, `T_disp_11`, `T_moda_11`, `P0_min_11`, `P0_max_11`, `P0_mean_11`, `P0_median_11`, `P0_disp_11`, `P0_moda_11`, `P_min_11`, `P_max_11`, `P_mean_11`, `P_median_11`, `P_disp_11`, `P_moda_11`, `U_min_11`, `U_max_11`, `U_mean_11`, `U_median_11`, `U_disp_11`, `U_moda_11`, `N_min_11`, `N_max_11`, `N_mean_11`, `N_median_11`, `N_disp_11`, `N_moda_11`, `Tmin_min_11`, `Tmin_max_11`, `Tmin_mean_11`, `Tmin_median_11`, `Tmin_disp_11`, `Tmin_moda_11`, `Tmax_min_11`, `Tmax_max_11`, `Tmax_mean_11`, `Tmax_median_11`, `Tmax_disp_11`, `Tmax_moda_11`, `Nh_min_11`, `Nh_max_11`, `Nh_mean_11`, `Nh_median_11`, `Nh_disp_11`, `Nh_moda_11`, `RRR_max_11`, `RRR_mean_11`, `RRR_median_11`, `RRR_disp_11`, `RRR_moda_11`, `sss_min_11`, `sss_max_11`, `sss_mean_11`, `sss_median_11`, `sss_disp_11`, `sss_moda_11`, " + \
                "`level_min_12`, `level_max_12`, `level_mean_12`, `level_median_12`, `level_disp_12`, `level_moda_12`, `T_min_12`, `T_max_12`, `T_mean_12`, `T_median_12`, `T_disp_12`, `T_moda_12`, `P0_min_12`, `P0_max_12`, `P0_mean_12`, `P0_median_12`, `P0_disp_12`, `P0_moda_12`, `P_min_12`, `P_max_12`, `P_mean_12`, `P_median_12`, `P_disp_12`, `P_moda_12`, `U_min_12`, `U_max_12`, `U_mean_12`, `U_median_12`, `U_disp_12`, `U_moda_12`, `N_min_12`, `N_max_12`, `N_mean_12`, `N_median_12`, `N_disp_12`, `N_moda_12`, `Tmin_min_12`, `Tmin_max_12`, `Tmin_mean_12`, `Tmin_median_12`, `Tmin_disp_12`, `Tmin_moda_12`, `Tmax_min_12`, `Tmax_max_12`, `Tmax_mean_12`, `Tmax_median_12`, `Tmax_disp_12`, `Tmax_moda_12`, `Nh_min_12`, `Nh_max_12`, `Nh_mean_12`, `Nh_median_12`, `Nh_disp_12`, `Nh_moda_12`, `RRR_max_12`, `RRR_mean_12`, `RRR_median_12`, `RRR_disp_12`, `RRR_moda_12`, `sss_min_12`, `sss_max_12`, `sss_mean_12`, `sss_median_12`, `sss_disp_12`, `sss_moda_12`) "

    for year in years:
        i = 0
        level_min = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        T_min = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        P0_min = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        P_min = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        U_min = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        N_min = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Tmin_min = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Tmax_min = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Nh_min = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        sss_min = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        level_max = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        T_max = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        P0_max = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        P_max = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        U_max = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        N_max = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Tmin_max = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Tmax_max = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Nh_max = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        RRR_max = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        sss_max = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        level_mean = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        level_median = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        level_disp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        level_moda = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        T_mean = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        T_median = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        T_disp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        T_moda = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        P0_mean = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        P0_median = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        P0_disp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        P0_moda = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        P_mean = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        P_median = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        P_disp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        P_moda = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        U_mean = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        U_median = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        U_disp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        U_moda = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        N_mean = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        N_median = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        N_disp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        N_moda = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Tmin_mean = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Tmin_median = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Tmin_disp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Tmin_moda = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Tmax_mean = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Tmax_median = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Tmax_disp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Tmax_moda = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Nh_mean = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Nh_median = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Nh_disp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Nh_moda = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        RRR_mean = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        RRR_median = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        RRR_disp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        RRR_moda = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        sss_mean = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        sss_median = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        sss_disp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        sss_moda = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for month in months:
            query = "SELECT * FROM input_data_full WHERE input_data_full.date LIKE '" + str(year) + "-" + str(
                month) + "-%' "
            cursor.execute(query)
            results_month = cursor.fetchall()

            # получаем одномерные массивы для вычислений
            level_s = [x['level'] for x in results_month]
            T_s = [x['T'] for x in results_month]
            P0_s = [x['P0'] for x in results_month]
            P_s = [x['P'] for x in results_month]
            U_s = [x['U'] for x in results_month]
            N_s = [x['N'] for x in results_month]
            Tmin_s = [x['Tmin'] for x in results_month]
            Tmax_s = [x['Tmax'] for x in results_month]
            Nh_s = [x['Nh'] for x in results_month]
            RRR_s = [x['RRR'] for x in results_month]
            sss_s = [x['sss'] for x in results_month]

            level_min[i] = min(level_s)
            T_min[i] = min(T_s)
            P0_min[i] = min(P0_s)
            P_min[i] = min(P_s)
            U_min[i] = min(U_s)
            N_min[i] = min(N_s)
            Tmin_min[i] = min(Tmin_s)
            Tmax_min[i] = min(Tmax_s)
            Nh_min[i] = min(Nh_s)
            sss_min[i] = min(sss_s)

            level_max[i] = max(level_s)
            T_max[i] = max(T_s)
            P0_max[i] = max(P0_s)
            P_max[i] = max(P_s)
            U_max[i] = max(U_s)
            N_max[i] = max(N_s)
            Tmin_max[i] = max(Tmin_s)
            Tmax_max[i] = max(Tmax_s)
            Nh_max[i] = max(Nh_s)
            RRR_max[i] = max(RRR_s)
            sss_max[i] = max(sss_s)

            level_mean[i] = getStatisticMean(level_s)
            T_mean[i] = getStatisticMean(T_s)
            P0_mean[i] = getStatisticMean(P0_s)
            P_mean[i] = getStatisticMean(P_s)
            U_mean[i] = getStatisticMean(U_s)
            N_mean[i] = getStatisticMean(N_s)
            Tmin_mean[i] = getStatisticMean(Tmin_s)
            Tmax_mean[i] = getStatisticMean(Tmax_s)
            Nh_mean[i] = getStatisticMean(Nh_s)
            RRR_mean[i] = getStatisticMean(RRR_s)
            sss_mean[i] = getStatisticMean(sss_s)

            level_moda[i] = getStatisticModa(level_s)
            T_moda[i] = getStatisticModa(T_s)
            P0_moda[i] = getStatisticModa(P0_s)
            P_moda[i] = getStatisticModa(P_s)
            U_moda[i] = getStatisticModa(U_s)
            N_moda[i] = getStatisticModa(N_s)
            Tmin_moda[i] = getStatisticModa(Tmin_s)
            Tmax_moda[i] = getStatisticModa(Tmax_s)
            Nh_moda[i] = getStatisticModa(Nh_s)
            RRR_moda[i] = getStatisticModa(RRR_s)
            sss_moda[i] = getStatisticModa(sss_s)

            level_median[i] = statistics.median_high(level_s)
            T_median[i] = statistics.median_high(T_s)
            P0_median[i] = statistics.median_high(P0_s)
            P_median[i] = statistics.median_high(P_s)
            U_median[i] = statistics.median_high(U_s)
            N_median[i] = statistics.median_high(N_s)
            Tmin_median[i] = statistics.median_high(Tmin_s)
            Tmax_median[i] = statistics.median_high(Tmax_s)
            Nh_median[i] = statistics.median_high(Nh_s)
            RRR_median[i] = statistics.median_high(RRR_s)
            sss_median[i] = statistics.median_high(sss_s)

            level_disp[i] = getStatisticDisp(level_s)
            T_disp[i] = getStatisticDisp(T_s)
            P0_disp[i] = getStatisticDisp(P0_s)
            P_disp[i] = getStatisticDisp(P_s)
            U_disp[i] = getStatisticDisp(U_s)
            N_disp[i] = getStatisticDisp(N_s)
            Tmin_disp[i] = getStatisticDisp(Tmin_s)
            Tmax_disp[i] = getStatisticDisp(Tmax_s)
            Nh_disp[i] = getStatisticDisp(Nh_s)
            RRR_disp[i] = getStatisticDisp(RRR_s)
            sss_disp[i] = getStatisticDisp(sss_s)

            i = i + 1

        sql_tmp = "VALUES (NULL, '" + str(year) + ""

        for i in range(12):
            sql_tmp = sql_tmp + "', '" + str(level_min[i]) + "', '" + str(level_max[i]) + "', '" + str(
                level_mean[i]) + "', '" + str(level_median[i]) + "', '" + str(level_disp[i]) + "', '" + str(
                level_moda[i]) + "', '" \
                      + str(T_min[i]) + "', '" + str(T_max[i]) + "', '" + str(T_mean[i]) + "', '" + str(
                T_median[i]) + "', '" + str(T_disp[i]) + "', '" + str(T_moda[i]) + "', '" \
                      + str(P0_min[i]) + "', '" + str(P0_max[i]) + "', '" + str(P0_mean[i]) + "', '" + str(
                P0_median[i]) + "', '" + str(P0_disp[i]) + "', '" + str(P0_moda[i]) + "', '" \
                      + str(P_min[i]) + "', '" + str(P_max[i]) + "', '" + str(P_mean[i]) + "', '" + str(
                P_median[i]) + "', '" + str(P_disp[i]) + "', '" + str(P_moda[i]) + "', '" \
                      + str(U_min[i]) + "', '" + str(U_max[i]) + "', '" + str(U_mean[i]) + "', '" + str(
                U_median[i]) + "', '" + str(U_disp[i]) + "', '" + str(U_moda[i]) + "', '" \
                      + str(N_min[i]) + "', '" + str(N_max[i]) + "', '" + str(N_mean[i]) + "', '" + str(
                N_median[i]) + "', '" + str(N_disp[i]) + "', '" + str(N_moda[i]) + "', '" \
                      + str(Tmin_min[i]) + "', '" + str(Tmin_max[i]) + "', '" + str(Tmin_mean[i]) + "', '" + str(
                Tmin_median[i]) + "', '" + str(Tmin_disp[i]) + "', '" + str(Tmin_moda[i]) + "', '" \
                      + str(Tmax_min[i]) + "', '" + str(Tmax_max[i]) + "', '" + str(Tmax_mean[i]) + "', '" + str(
                Tmax_median[i]) + "', '" + str(Tmax_disp[i]) + "', '" + str(Tmax_moda[i]) + "', '" \
                      + str(Nh_min[i]) + "', '" + str(Nh_max[i]) + "', '" + str(Nh_mean[i]) + "', '" + str(
                Nh_median[i]) + "', '" + str(Nh_disp[i]) + "', '" + str(Nh_moda[i]) + "', '" \
                      + str(RRR_max[i]) + "', '" + str(RRR_mean[i]) + "', '" + str(RRR_median[i]) + "', '" + str(
                RRR_disp[i]) + "', '" + str(RRR_moda[i]) + "', '" \
                      + str(sss_min[i]) + "', '" + str(sss_max[i]) + "', '" + str(sss_mean[i]) + "', '" + str(
                sss_median[i]) + "', '" + str(sss_disp[i]) + "', '" + str(sss_moda[i])

        sql_tmp = sql_tmp + "'); "
        final_query = big_query + sql_tmp

        # для запуска при ошибки коннекта
        # print(final_query)
        cursor.execute(final_query)
        cursor.fetchall()

        # закрываем курсор и соединение с бд
        cursor.close()
        connection.close()

    return 1


def build_model():
    model = Sequential()
    model.add(Dense(units=708, activation='relu'))
    model.add(Dense(units=390, activation='relu'))
    model.add(Dense(units=390, activation='relu'))
    model.add(Dense(units=52, activation='relu'))
    model.add(Dense(units=13, activation='sigmoid'))

    # activation : elu, sigmoid, relu, softmax
    # loss       : mse, binary_crossentropy(0.4), poisson(0.7), logcosh(0.01), hinge
    # optimizer  : sgd(стох град спуск), rmsprop(sgd с импульсом), adam (изменение скорости)
    # metrics    : accuracy

    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    return model


def getTrainData():
    connection = conn.getConnection()
    cursor = connection.cursor()

    # запрос - получение данных
    cursor.execute("SELECT * FROM input_data where `year` NOT LIKE '2012' ORDER BY id DESC")
    results_in = cursor.fetchall()
    cursor.execute("SELECT * FROM output_data where `year` NOT LIKE '2012' ORDER BY id DESC")
    results_out = cursor.fetchall()
    # формирование обучающих входных и выходных данных; создание датасета, конвертирование
    train_input = convertData(results_in)
    train_output = convertData(results_out)
    # нормализация данных и удаление константных значений
    train_input = normalizeData(train_input)
    train_output = normalizeData(train_output)

    # закрываем курсор и соединение с бд
    cursor.close()
    connection.close()

    return [train_input, train_output]


def getMainData(train_input, train_output):
    connection = conn.getConnection()
    cursor = connection.cursor()

    cursor.execute("SELECT * FROM output_data where `year` LIKE '2012' ORDER BY id DESC")
    test_results_out = cursor.fetchall()
    cursor.execute("SELECT * FROM input_data where `year` LIKE '2012' ORDER BY id DESC")
    test_results_in = cursor.fetchall()
    # формирование тестовых входных и выходных данных; создание датасета
    test_input = convertData(test_results_in)
    test_output = convertData(test_results_out)
    # склеиваем данные с обучающими для нормализации (потом данные снова разделяются)
    new_input = pd.concat([train_input, test_input])
    new_output = pd.concat([train_output, test_output])
    # передаем на нормаизацию весь датафрейм для получения относительных величин,
    test_input = normalizeData(new_input)
    test_output, denorm = normalizeTestOutputData(new_output)
    # после чего оставляем только последнюю строку - запрашиваемый год
    test_input = test_input[len(test_input) - 1:]
    test_output = test_output[len(test_output) - 1:]

    # закрываем курсор и соединение с бд
    cursor.close()
    connection.close()

    return [test_input, test_output, denorm]


def getGrade(train_input, train_output, epochs):
    # гиперпараметры
    global val_mse
    validation_split = 0.2
    all_scores = []
    all_mae_histories = []
    years = [[2006, 2010], [2011, 2013], [2014, 2015], [2016, 2017], [2018, 2019]]
    step = 0

    # Оценка решения методом перекрестной проверки по K блокам
    for qur_test_years in years:
        print('этап обработки ', step + 1, "; Время: ", str(datetime.datetime.now().time()))
        step = step + 1
        # список лет для обучения
        qur_train_years = years.copy()
        qur_train_years.remove(qur_test_years)
        # преобразуем в одномерный массив
        qur_train_years = [a for b in qur_train_years for a in b]

        # временные обучающие данные
        qur_train_input = pd.DataFrame(train_input, index=qur_train_years)
        qur_train_output = pd.DataFrame(train_output, index=qur_train_years)
        # временные проверочные данные
        qur_test_input = pd.DataFrame(train_input, index=qur_test_years)
        qur_test_output = pd.DataFrame(train_output, index=qur_test_years)
        # построение модели
        model = build_model()
        history = model.fit(
            x=qur_train_input,
            y=qur_train_output,
            epochs=epochs,
            validation_split=validation_split,
            verbose=0
        )
        # получение оценок
        val_mse, val_mae = model.evaluate(qur_test_input, qur_test_output, verbose=0)
        mae_history = history.history['loss']
        all_mae_histories.append(mae_history)
        all_scores.append(val_mae)

    return [all_scores, val_mse, all_mae_histories, epochs]


def getGradePlots(all_scores, val_mse, all_mae_histories, epochs):
    print("Оценка по MAE по K блокам: ", all_scores)
    print("Оценка по MSE по K блокам: ", val_mse)
    print("Средняя оценка по MAE по K блокам: ", np.mean(all_scores))
    # графики оценок
    average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(epochs)]
    plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
    plt.xlabel("Эпохи")
    plt.ylabel("Оценка по MAE(средняя абсолютной разницы) ")
    plt.show()


def makeFinalModel(test_input, test_output, epochs):
    # построение итоговой модели
    model = build_model()
    model.fit(
        x=test_input,
        y=test_output,
        epochs=epochs,
        verbose=0
    )
    test_mse_score, test_mae_score = model.evaluate(test_input, test_output)
    predicted = model.predict(test_input)
    model.save('model/last_model')

    return test_mse_score, test_mae_score, predicted


def loadFinalModel(test_input, test_output):
    # загрузка итоговой модели
    model = models.load_model('model/last_model')
    test_mse_score, test_mae_score = model.evaluate(test_input, test_output)
    predicted = model.predict(test_input)

    return test_mse_score, test_mae_score, predicted


def getFinalResult(test_mae_score, test_mse_score, predicted, test_output, denorm):
    # вывод оценок
    print("test_mae_score : ", test_mae_score)
    print("test_mse_score : ", test_mse_score)

    # вывод результата
    predicted = pd.DataFrame(predicted)
    print(predicted)
    print(test_output)

    rus_columns = ['Донская Царица', 'Новоцимлянский', 'оз.Анушкино', 'оз.Бугаково', 'оз.Некрасово', 'оз.Нижнее',
                   'оз. Среднее', 'Ромашкинский залив', 'Мокро-Соленовская б.', 'Сухо-Соленовская б.',
                   'Уст. реки Мышкова',
                   'Чирской залив', 'Балабановский залив']
    predicted.columns = rus_columns
    test_output.columns = rus_columns
    test_output_T = test_output.copy()
    predicted_T = predicted.copy()

    itog = test_output.copy()
    for column_name in itog.columns:
        itog[column_name][2012] = abs(test_output_T[column_name][2012] - predicted_T[column_name][0]) / \
                                  test_output_T[column_name][2012]

    sum = 0
    i = 0
    for column_name in itog.columns:
        sum = sum + itog[column_name][2012]
        i = i + 1

    sred = sum / i
    print(sred)

    # денормализуем данные
    test_output = denormalizeData(test_output, denorm)
    predicted = denormalizeData(predicted, denorm)

    # вывод графиков
    test_output_trace = go.Bar(x=test_output.columns, y=[a for a in predicted.iloc[0]], width=0.35, offset=0.18,
                               name="Реальные данные")
    predicted_trace = go.Bar(x=predicted.columns, y=[a for a in test_output.iloc[0]], width=0.35,
                             name="Предсказанные данные")
    fig = make_subplots(specs=[[{"secondary_y": True}]], row_titles=["Зарастание, км&#178;"],
                        column_titles=["Водные ресурсы"])
    fig.add_trace(test_output_trace)
    fig.add_trace(predicted_trace)
    fig['layout'].update(title='Сравнение данных, спрогнозированных моделью за 2012 год.', xaxis=dict(
        tickangle=-90))

    # fig.show()
    fig.write_image("output_graphs/new_graph.png")

