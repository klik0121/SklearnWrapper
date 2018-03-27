import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from WrapperForm import WrapperForm

app = QApplication(sys.argv)
form = WrapperForm()
sys.exit(app.exec_())
