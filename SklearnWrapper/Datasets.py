"""
Модуль для работы с наборами данных.
Требование к набору данных(или его генератору):
Должен возвращать 2(два) массива: X и y.
X - массив M x N - набор свойств объектов,
y - массив 1 x N - набор классов объектов
"""

import sklearn.datasets
from inspect import getmembers, isfunction, signature

def get_dataset_dict():
    """Получить список методов. Индекс 0 - имя метода, индекс 1 - тело метода"""
    method_list = getmembers(sklearn.datasets, isfunction) + getmembers(sklearn.datasets.samples_generator)

    return {method[0]:method[1] for method in method_list
            if method[0].startswith('make_')}

def get_gen_dict():
    """Получить сигнатуру методов."""
    return {k:signature(v)
            for k, v in get_dataset_dict().items()}
