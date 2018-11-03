import sys, traceback
from pathlib import Path
import os
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from Datasets import get_gen_dict, get_dataset_dict
from MethodWrapper import MethodWrapper
from clustering import *
from classification import *
from ast import literal_eval
from contextlib import redirect_stdout
import numpy as np

class WrapperForm(QWidget):

    def __init__(self):
        super().__init__()
        self.wrappers = MethodWrapper.wrappers
        self.initUI()
        self.onActivated(list(self.wrappers.keys())[0])
        self.onActivatedData(list(get_gen_dict().keys())[0])
        self.cwd = os.getcwd()
        self.log_path = Path(sys.argv[0]).parent / "error.log"
        if os.path.exists(self.log_path):
            os.remove(self.log_path)

    def get_args(self, table):
        """Извлекает параметры из таблицы, формируя словарь"""
        args = {}
        for i in range(table.rowCount()):
            value = table.item(i, 1).text()
            try:
                value = literal_eval(value)
            except SyntaxError as e: pass
            except ValueError as e: pass
            args[table.item(i, 0).text()] = value
        return args

    def init_dataset(self):
        """Собирает список аргументов, затем запускает выбранный метод"""
        args = self.get_args(self.table_dataset)
        try:
            self.dataset = get_dataset_dict()[self.current_method](**args)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.error(str(e))
            self.log_except(e)


    def execute(self, button):
        """Задаёт параметры классификатора, затем запускает его"""
        self.instance.__dict__ = self.get_args(self.table_method)
        # Если метод создаёт свои файлы,
        # они будут помещены в папку output в корне проекта
        out_path = Path(sys.argv[0]).parent / "output"
        if not os.path.exists(out_path): os.makedirs(out_path)
        os.chdir(out_path)
        f = open(type(self.instance).__name__ +
            '-output.txt', 'w')
        with redirect_stdout(f):
            if hasattr(self, 'dataset'):
                try:
                    self.instance.execute(self.dataset)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    self.error(str(e))
                    self.log_except(e)
            else:
                self.error("Отсутствует набор данных")
        os.chdir(self.cwd)

    def log_except(self, e):
        """Записать ошибку и стек вызова в файл error.log"""
        f = open(self.log_path, 'w')
        traceback.print_exc(file=f)
        f.close()

    def error(self, message):
        """Сообщение об ошибке"""
        msg = QMessageBox()
        msg.setText(message)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Ошибка!")
        msg.setStandardButtons(QMessageBox.Ok)
        retval = msg.exec_()

    def create_combo(self, items, y, con):
        """Список методов, строка 0"""
        combo = QComboBox()
        combo.addItems(items)
        combo.activated[str].connect(con)
        self.mainLayout.addWidget(combo, 0, y)

    def create_param_table(self, x):
        """Таблица параметров, занимает строки с 1 по 8"""
        table = QTableWidget(self)
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Параметр", "Значение"])
        self.mainLayout.addWidget(table, 1, x, 8, 1)
        return table

    def create_button(self, label, cnt, x):
        """Кнопка "Go", строка 9"""
        btn = QPushButton(label)
        btn.clicked.connect(cnt)
        self.mainLayout.addWidget(btn, 9, x)

    def addHorizontalButton(self, layout, label, cnt):
        btn = QPushButton(label)
        btn.clicked.connect(cnt)
        layout.addWidget(btn)

    def createLayout(self):
        """Инициализирует и размещает элементы формы"""
        # Тип расположения элементов - таблица
        self.mainLayout = QGridLayout()

        # Методы наборов данных
        self.create_combo(get_gen_dict().keys(), 0, self.onActivatedData)
        self.table_dataset = self.create_param_table(0)
        #self.create_button('Create dataset', self.init_dataset, 0, 1)
        #self.create_button('Save dataset', self.saveDataset, 1, 1)

        hblayout = QHBoxLayout()
        self.addHorizontalButton(hblayout, 'Create dataset', self.init_dataset)
        self.addHorizontalButton(hblayout, 'Load dataset', self.loadDataset)
        self.addHorizontalButton(hblayout, 'Save dataset', self.saveDataset)
        self.mainLayout.addLayout(hblayout, 9, 0)


        # Методы анализа данных
        self.create_combo(self.wrappers.keys(), 1, self.onActivated)
        self.table_method = self.create_param_table(1)
        self.create_button('Go', self.execute, 1)

        # Фиксированный размер формы
        self.mainLayout.setSizeConstraint(3)
        self.setLayout(self.mainLayout)

    def saveDataset(self):
        """Сохранение набора данных в файл"""
        if self.dataset:
            fileName = QFileDialog.getSaveFileName(
                self, "Save File", "",
                "Comma-separated values file (*.csv)")[0]
            if fileName:
                np.savetxt(fileName, np.c_[self.dataset], delimiter = "\t")

    def get_from_array(self, arr):
        sh = np.shape(arr)
        cols_num = sh[1]
        return arr[:, 0: cols_num - 1], arr[:, cols_num - 1]

    def loadDataset(self):
        fileName = QFileDialog.getOpenFileName(
            self, "Open File", "",
            "Comma-separated values file (*.csv)")[0]
        if fileName:
            self.dataset = self.get_from_array(
            np.genfromtxt(fileName, dtype=float, delimiter = '\t'))

    def initUI(self):

        self.setGeometry(300, 300, 300, 300)
        self.setWindowTitle(list(self.wrappers.keys())[0])

        self.createLayout()
        self.show()

    def onActivated(self, text):
        """Заполнить таблицу параметров"""
        self.instance = self.wrappers[text]()
        self.table_method.clearContents()
        self.setWindowTitle(text)
        dict = self.instance.__dict__
        self.table_method.setRowCount(len(dict))
        i = 0
        for k, v in dict.items():
            self.table_method.setItem(i, 0, QTableWidgetItem(k))
            self.table_method.setItem(i, 1, QTableWidgetItem(str(v)))
            i+=1

    def onActivatedData(self, text):
        """Заполнить таблицу параметров"""
        self.current_method = text
        table = self.table_dataset
        table.clearContents()
        params = get_gen_dict()[text].parameters.values()
        l = len(params)
        table.setRowCount(l)
        for i, k in zip(range(l), params):
            table.setItem(i, 0, QTableWidgetItem(k.name))
            table.setItem(i, 1, QTableWidgetItem(
            str(k.default) if k.default is not k.empty else ""))
