import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from MethodWrapper import MethodWrapper
from clustering import DBSCANWrapper, AffinityPropagationWrapper, SCWrapper, GMWrapper
from classification import MLPWrapper, GNBWrapper, KRRWrapper, LRWrapper


class WrapperForm(QWidget):

    def __init__(self):
        super().__init__()
        self.wrappers = MethodWrapper.wrappers
        self.initUI()
        self.onActivated(list(self.wrappers.keys())[0])

    def execute(self, button):
        for i in range(0, self.table.rowCount()):
            param_name = self.table.item(i, 0).text()
            func_name = "set_" + param_name
            if hasattr(self.instance, func_name):
                try:
                    func = getattr(self.instance, func_name)
                    func(self.table.item(i, 1).text())
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    self.error("Не удалось установить значение " + param_name + ": " + str(e))
            else:
                self.error("Невозможно установить значение параметра " + param_name + ".")
                return
        try:
            self.instance.execute()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.error(str(e))

    def error(self, message):
        msg = QMessageBox()
        msg.setText(message)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Ошибка!")
        msg.setStandardButtons(QMessageBox.Ok)
        retval = msg.exec_()

    def createLayout(self):
        self.mainLayout = QGridLayout()
        self.mainLayout.SizeConstraint
        self.setLayout(self.mainLayout)

        self.table = QTableWidget(self)
        self.table.setColumnCount(2)
        self.mainLayout.addWidget(self.table, 1, 0, 8, 0)

        btn = QPushButton('Go')
        btn.resize(btn.sizeHint())
        btn.move(50, 50)
        btn.clicked.connect(self.execute)
        self.mainLayout.addWidget(btn, 9, 0)

        combo = QComboBox()
        combo.addItems(self.wrappers.keys())
        combo.move(10, 10)
        combo.activated[str].connect(self.onActivated)
        self.mainLayout.addWidget(combo, 0, 0)

    def initUI(self):

        self.setGeometry(300, 300, 300, 300)
        self.setWindowTitle(list(self.wrappers.keys())[0])

        self.createLayout()
        self.show()

    def onActivated(self, text):
        self.instance = self.wrappers[text]()
        self.table.clear()
        self.table.setHorizontalHeaderLabels(["Параметр", "Значение"])
        self.setWindowTitle(text)
        dict = self.instance.__dict__
        self.table.setRowCount(len(dict))
        i = 0
        for k, v in dict.items():
            self.table.setItem(i, 0, QTableWidgetItem(k))
            self.table.setItem(i, 1, QTableWidgetItem(str(v)))
            i+=1
