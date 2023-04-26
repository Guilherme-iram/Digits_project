import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import accuracy_score


# _______________________________________ Reducao de dimensionalidade _______________________________________

def simetria_vertical_digito(matrixes):
    res = []
    for matrix in matrixes:
        soma = 0
        for i in range(0, 28):
            for j in range(0, 14):
                soma += abs(matrix[(28 * i) + j] - matrix[(28 * i) + 27 - j])
        res.append(soma / 255.0)
    return np.array(res)

    
def simetria_horizontal_digito(matrixes):
    res = []
    for matrix in matrixes:
        soma = 0
        for i in range(0, 14):
            for j in range(0, 28):
                soma += abs(matrix[(28 * i) + j] - matrix[(28 * (27 - i)) + j])
        res.append(soma / 255.0)
    return np.array(res)


def intensidade_digito(matrixes): 
    return np.array([sum(matrix) / 255.0 for matrix in matrixes])

# _____________________________________ Classificação de Dígitos _____________________________________

def calculate_y(x, w):
    return (-w[0] - w[1]*x) / w[2]


def plot_classification_digits(df, digits_list, titulo, ax=None,  W=[]):

    colors = ['red', 'gold', 'green', 'purple']
    
    if ax == None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        # possivelmente apagar essa linha
        fig = ax.get_figure()

    for i, label in enumerate(df.label.unique()):
        data = df[df.label == label]
        ax.scatter(data.intensidade, data.simetria, color=colors[i], label=label)

    if len(W) > 0:
        linestyles = ['dashed', 'dotted', 'dashdot', 'dotted']
        colors_w = ['black', 'blue', 'gray']
        for i, w in enumerate(W):
            x_values = np.array([df.intensidade.min(), df.intensidade.max()])
            y_values = calculate_y(x_values, w)
            ax.plot(x_values, y_values,
             color=colors_w[i],
              linestyle=linestyles[i],
               label=f'Reta {digits_list[i]}x{digits_list[i + 1:len(w) + 1]}')


    ax.legend()
    ax.set_xlabel('Intensidade')
    ax.set_ylabel('Simetria')
    ax.set_title(titulo)
    ax.set_xlim([40, 165])
    ax.set_ylim([55, 170])

# _______________________________________ Matriz de Confusão _______________________________________


def multiclass_confusion_matrix(y_true, y_pred):

    labels = sorted(np.unique(y_true))
    n = len(labels)

    cm = np.zeros((n, n), dtype='int')
    
    for i in range(n):
        for j in range(n):
            cm[i, j] = np.sum(np.logical_and(y_pred == labels[j], y_true == labels[i]))

    return cm


def confusion_matrix_plot(y_test, y_pred, ax=None):

    cm = multiclass_confusion_matrix(y_test, y_pred)
    labels = set(y_test)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt='d' ,linewidths=.5, square = True, cmap = 'Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f"Acurracy Score: {np.sum(np.diag(cm)) / np.sum(cm):.4f}", size = 15)
    ax.set_ylabel('Actual label', size = 12)
    ax.set_xlabel('Predicted label', size = 12)


# _______________________________________ Métrica Multiclasses _______________________________________

class Metrics_multiclass:

    def __init__(self, digits):
        self.digits = sorted(digits)


    def acurracy_multiclass(self, y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)


    def multiclass_error(self, y_true, y_pred):
        return 1 - accuracy_score(y_true, y_pred)


    def precision_multiclass(self, y_true, y_pred, label):
        cm = multiclass_confusion_matrix(y_true, y_pred)
        return cm[self.digits.index(label), self.digits.index(label)] \
            / np.sum(cm[:, self.digits.index(label)]) # Somas das colunas do digito


    def recall_multiclass(self, y_true, y_pred, label):
        cm = multiclass_confusion_matrix(y_true, y_pred)
        return cm[self.digits.index(label), self.digits.index(label)] /\
             np.sum(cm[self.digits.index(label), :])  # Soma da linha do digito


    def f1_score_multiclass(self, y_true, y_pred, label):
        return 2 * self.precision_multiclass(y_true, y_pred, label) * self.recall_multiclass(y_true, y_pred, label)\
            / (self.precision_multiclass(y_true, y_pred, label) + self.recall_multiclass(y_true, y_pred, label))


    def weighted_f1_score_multiclass(self, y_true, y_pred):
        return sum([self.f1_score_multiclass(y_true, y_pred, label) for label in self.digits]) / len(self.digits)


    def print_metrics_multiclass(self, y_true, y_pred):

        labels = self.digits
        n = len(labels)

        print("---------------------------------")
        print("RELATORIO CLASSIFICACAO MULTICLASS")
        print("_______________________________________________________")
        print(f"Acurracy: {self.acurracy_multiclass(y_true, y_pred):.4f}")
        print(f"Error de Classificacao: {self.multiclass_error(y_true, y_pred):.4f}")
        print("_______________________________________________________")
        for d in self.digits:
            print("_______________________________________________________")
            print(f"Precision para digito {d}: {self.precision_multiclass(y_true, y_pred, d):.4f}")
            print(f"Recall para digito {d}: {self.recall_multiclass(y_true, y_pred, d):.4f}")
            print(f"F1 Score para digito {d}: {self.f1_score_multiclass(y_true, y_pred, d):.4f}")
            print("_______________________________________________________")
        print(f"Weighted F1 Score: {self.weighted_f1_score_multiclass(y_true, y_pred):.4f}")
        print("---------------------------------")
        confusion_matrix_plot(y_true, y_pred)

# _____________________________________________________________________________________________________________________

