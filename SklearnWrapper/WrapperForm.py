import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from MethodWrapper import MethodWrapper
from clustering import DBSCANWrapper


class WrapperForm(QWidget):

    def __init__(self):
        super().__init__()
        self.wrappers = MethodWrapper.wrappers        
        self.initUI()        
        self.onActivated(list(self.wrappers.keys())[0])

    def execute(self, button):
        self.instance.execute()

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
        dict = self.instance.__dict__
        self.table.setRowCount(len(dict))
        i = 0
        for k, v in dict.items():
            self.table.setItem(i, 0, QTableWidgetItem(k))
            self.table.setItem(i, 1, QTableWidgetItem(str(v)))
            i+=1           



