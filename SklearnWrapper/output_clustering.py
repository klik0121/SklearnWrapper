# Реализует логирование для методов-кластеризаторов
from datetime import datetime
from itertools import groupby

def write_perf_params(y, y_true):
    """ Получает параметры оценивания кластеризации"""
    a,b,c,d = (0,0,0,0)
    y_zip_t = zip(y, y_true)
    # Для каждой пары элементов в y проверяется, соответствует ли их
    # разбиение по кластерам предложенному экспертом (эталонному)
    # На основании кол-ва соответствующих / несоответствующих эталону
    # пар рассчитываются оценочные коэффициенты
    for pt1, pt1_t in y_zip_t:
        for pt2, pt2_t in y_zip_t:
            if pt1 is not pt2 and pt1 == pt2:
                if pt1_t == pt2_t:
                    # Пара в кластере, эталон в кластере
                    a = a + 1
                else:
                    # Пара в кластере, эталон не в кластере
                    b = b + 1
            else:
                if pt1_t == pt2_t:
                    # Пара не в кластере, эталон в кластере
                    c = c + 1
                else:
                    # Пара не в кластере, эталон не в кластере
                    d = d + 1
    print('P: ' + str(a / (a + b)))
    print('R: ' + str(a / (a + c)))
    print('E: ' + str((b + c) / (a + b + c + d)))
    print('A: ' + str((a + d) / (a + b + c + d)))

def write_clusters(y, n_clusters = 0):
    # Вывести номера кластеров с находящимися в них элементами
    if n_clusters > 0:
        # Поскольку количество кластеров задаётся изначально,
        # достаточно вывести номера соответствующих элементов
        for i in range(n_clusters):
            print('Cluster {}:'.format(i))
            for j in [index for index, value in enumerate(y) if value == i]:
                print(j)
    else:
        # В противном случае, элементы группируются по кластерам
        y_srt = sorted(range(y), key = lambda k: y[k])
        clusters = groupby(y_srt, key = lambda k: y[k])
        for i, cluster in enumerate(clusters):
            print('Cluster {}:'.format(i))
            for point in cluster:
                print(point)

def output(class_name, y, y_true = None, n_clusters = 0):
    """Выводит список кластеров и номера содержащихся в них объектов.
    Параметры:
    class_name - название класса, вызвавшего метод;
    y - результаты, полученные классификатором. Формат: список, состоящий из номеров кластеров;
    y_true - эталонные результаты, необязательный параметр для оценки качества разбиения;
    n_clusters - количество кластеров для кластеризаторов с задаваемым ограничением на кол-во кластеров"""
    print('//{0}, log time: {1}'.format(class_name, datetime.today().replace(microsecond=0)))
    print('//Predicted clusters:')
    write_clusters(y, n_clusters)
    print('//Clustering quality estimation:')
    if y_true is not None:
        write_perf_params(y, y_true)
    print('//END OF LOG')
