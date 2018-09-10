"""
Модуль для работы с наборами данных.
Требование к набору данных(или его генератору):
Должен возвращать 2(два) массива: X и y.
X - массив M x N - набор свойств объектов,
y - массив 1 x N - набор классов объектов
"""

import sklearn.datasets
import urllib.request as r
import numpy as np
from inspect import getmembers, isfunction, signature
from os.path import join, exists, abspath, dirname
from os import makedirs

def get_dataset_dict():
    """Получить список методов. Индекс 0 - имя метода, индекс 1 - тело метода"""
    method_list = getmembers(sklearn.datasets, isfunction) + getmembers(sklearn.datasets.samples_generator)
    method_list.append(["get_from_url", get_from_url])
    method_list.append(["get_from_file", get_from_file])

    return {method[0]:method[1] for method in method_list
            if method[0].startswith('make_') or method[0].startswith('get_from_')}

def get_gen_dict():
    """Получить сигнатуру методов."""
    return {k:signature(v)
            for k, v in get_dataset_dict().items()}


def get_from_url(url:str, dataset_name:str='toy', delimiter:str='\t', cache:bool=True):
    """Загружает файл, кэширует при надобности"""
    dir = dirname(abspath(__file__)) + '\\datasets'
    if not exists(dir):
        makedirs(dir)
    file_name = join(dir, dataset_name + '.bin')
    if cache:
        if not exists(file_name):
            download(url, file_name)
        return get_from_file(file_name, delimiter)
    else:
        arr = np.genfromtxt(url, delimiter = delimiter)
        return get_from_array(arr)

def get_from_array(arr):
    sh = np.shape(arr)
    cols_num = sh[1]
    return arr[:, 0: cols_num - 1], arr[:, cols_num - 1]

def get_from_file(file_name:str, delimiter:str):
    arr = np.genfromtxt(file_name, dtype=float, delimiter = delimiter)
    return get_from_array(arr)

def download(url:str, file_name:str):
    """Загружает файл по заданному URL"""
    with r.urlopen(url) as response, open(file_name, 'wb') as file:
        data = response.read()
        file.write(data)