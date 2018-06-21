# SklearnWrapper

Оболочка для sklearn на Python.

Для запуска проекта необходим Python 3.6 и следующие модули:
* cycler==0.10.0
* kiwisolver==1.0.1
* matplotlib==2.2.2
* numpy==1.14.3
* pyparsing==2.2.0
* PyQt5==5.10.1
* python-dateutil==2.7.3
* pytz==2018.4
* scikit-learn==0.19.1
* scipy==1.1.0
* sip==4.19.8
* six==1.11.0
* sklearn==0.0

Все вышеперечисленные модули, за исключением PyQt5 и sklearn, входят в комплектацию последней версии Anaconda.

Cтруктура решения:
  * MethodWrapper: абстрактный класс. Каждый наследник представляет из себя оболочку для конкретного метода.
  * WrapperForm: GUI, подключает методы.

Реализованные методы:
  * Классификация:
    * Multi-Layer Perceptron
    * Decision Trees
    * Gaussian Naive Bayes
    * Kernel Ridge Regression
    * Linear Regression
  * Кластеризация:
    * DBSCAN
    * Affinity Propagation
    * Spectral Clustering
    * Gaussian Mixture

Как пользоваться:
  * Чтобы добавить свою обёртку:
    * Добавить класс с реализацией
      * Конструктор без параметров
      * Унаследован от MethodWrapper
      * Реализует execute так, как вам кажется нужным
      * Параметры обёртываемого метода являются полями экземпляра класса
      * *На будущее*: источник данных задаётся в параметрах

Что делать:
  * Реализовывать свои методы
  * *Добавить в интерфейс возможность выбирать источник данных из файла*
