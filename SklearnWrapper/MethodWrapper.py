from abc import ABC
from sklearn.metrics import confusion_matrix, adjusted_rand_score
import numpy as np

class MethodWrapper(object):
    """Abstract class for wrapping one of the sklearn methods"""
    wrappers = {}

    def __init__(self):
        self.animation_delay = -1

    @classmethod
    def __init_subclass__(cls, name:str, **kwargs):
        cls.wrappers[name] = cls

    def get_stats(self, correct, computed):
        """Computes stats for classification/clustering based on confusion matrix."""
        computed_relabeled = self.relabel(correct, computed)
        relabeled_labels = list(computed_relabeled)
        clusters = set(correct).union(set(relabeled_labels))
        #we just don't care about clusters we don't have in correct set
        confusionMatrix = confusion_matrix(correct, computed_relabeled, labels=list(set(correct)))
        stats = Stats()
        stats.confusionMatrix = confusionMatrix
        stats.compute()
        return stats
    
    def relabel(self, correct, computed):
        clusters = set(correct)
        computed_clusters = set(computed)
        used_clusters=set()
        unmached_clusters=[]
        for k in computed_clusters:
            if k >= 0:
                unmached_clusters.append(k)
        unmached_clusters = sorted(unmached_clusters)
        result = np.full(np.shape(computed), -1)
        for target_cluster in clusters:
            if target_cluster in used_clusters:
                continue
            max = 0
            selected_cluster = -1
            #here we want to find the best cluster to relabel
            #source_cluster is the one from computed ones
            #target_cluster is the one from true ones
            for source_cluster in unmached_clusters:
                tmp1 = np.full(np.shape(computed), - 1)
                tmp1[computed == source_cluster] = source_cluster
                tmp2 = np.full(np.shape(correct), - 1)
                tmp2[correct == target_cluster] = target_cluster
                #we evaluate ARI score between those clusters considering others as equal
                score = abs(adjusted_rand_score(tmp2, tmp1))
                if score > max:
                    max = score
                    selected_cluster = source_cluster
            #here we assume that target_cluster is mapped to selected_cluster
            if selected_cluster !=-1:
                unmached_clusters.remove(selected_cluster)
                used_clusters.add(target_cluster)
                result[computed == selected_cluster] = target_cluster
        return result
        

class Stats(object):
    def __init__(self):
        self.confusionMatrix = np.empty((1,1))

    def compute(self):
        self.TP = np.diag(self.confusionMatrix) #true posititve
        self.FP = [] #false positive
        self.FN = [] #false negative
        self.TN = [] #true negative
        for i in range(self.confusionMatrix.shape[0]):
            self.FP.append(sum(self.confusionMatrix[:,i]) - self.confusionMatrix[i,i])
            self.FN.append(sum(self.confusionMatrix[i,:]) - self.confusionMatrix[i,i])
            temp = np.delete(self.confusionMatrix, i, 0) #delete ith row
            temp = np.delete(temp, i, 1) #delete ith column
            self.TN.append(sum(sum(temp)))
        self.ClassPresicions = np.divide(self.TP, (np.add(self.TP, self.FP)))
        self.ClassPresicions[np.isnan(self.ClassPresicions)] = 1
        self.AveragePresicion = np.average(self.ClassPresicions)
        self.ClassCompleteness = np.divide(self.TP, np.add(self.TP, self.FN))
        self.ClassCompleteness[np.isnan(self.ClassCompleteness)] = 1
        self.AverageCompleteness = np.average(self.ClassCompleteness)
        self.ClassError = np.divide(np.add(self.FP, self.FN),(np.add(np.add(np.add(self.TP, self.FP), self.TN), self.FN)))
        self.AverageError = np.average(self.ClassError)
        self.ClassCorrectness = np.ones(len(self.ClassError)) - self.ClassError
        self.AverageCorrectness = np.average(self.ClassCorrectness)
        self.ClassIntegral = np.add(np.add(self.ClassPresicions, self.ClassCompleteness), self.ClassCorrectness)
        self.AverageIntegral = np.average(self.ClassIntegral)

    def get_formatted(self):
        np.set_printoptions(linewidth=np.Inf)
        formatStr = "Матрица ошибок (Confusion matrix): \n\r {0} \n\r "
        formatStr += "Истинный положительный результат по классам (TP): \n\r {1} \n\r "
        formatStr += "Ложный положительный результат по классам (FP): \n\r {2} \n\r "
        formatStr += "Истинный отрицательный результат по классам (TN): \n\r {3} \n\r "
        formatStr += "Ложный отрицательный результат по классам (FN): \n\r {4} \n\r "
        formatStr += "Точность по классам (P = TP/(TP + FP)): \n\r {5} \n\r "
        formatStr += "Средняя точность: {6} \n\r "
        formatStr += "Полнота по классам (R = TP/(TP + FN)): \n\r {7} \n\r "
        formatStr += "Средняя полнота: {8} \n\r "
        formatStr += "Ошибка по классам (E = (FP + FN)/(TP + TN + FP + FN)): \n\r {9} \n\r "
        formatStr += "Средняя ошибка: {10} \n\r "
        formatStr += "Правильность по классам (A = 1 - E): \n\r {11} \n\r "
        formatStr += "Средняя правильность: {12} \n\r "
        formatStr += "Интегральная оценка по классам (I = P + R + A): \n\r {13} \n\r "
        formatStr += "Средняя интегральная оценка:{14}  \n\r "
        formatStr = formatStr.format(self.confusionMatrix, self.TP, self.FP, self.TN, self.FN, self.ClassPresicions, self.AveragePresicion, 
                      self.ClassCompleteness, self.AverageCompleteness, self.ClassError, self.AverageError,
                      self.ClassCorrectness, self.AverageCorrectness, self.ClassIntegral, self.AverageIntegral)
        return formatStr.encode(encoding='utf-8', errors='strict')
        
        
        
            
        
                
        
