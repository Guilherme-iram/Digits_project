import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


def calculate_y(x, w):
    return (-w[0] - w[1]*x) / w[2]


def plot_classification_digits(df, digits_list, colors_list, titulo, W=[]):
    
    colors = {digits_list[i]: colors_list[i] for i in range(len(digits_list))}
    
    # y = calculate_y(df.intensidade, w)
    fig, ax = plt.subplots(figsize=(10, 6))
    for label in df.label.unique():
        data = df[df.label == label]
        ax.scatter(data.intensidade, data.simetria, color=colors[label], label=label)

    if len(W) > 0:
        linestyles = ['dashed', 'dotted', 'dashdot', 'dotted']
        for i, w in enumerate(W):
            x_values = np.array([df.intensidade.min(), df.intensidade.max()])
            y_values = calculate_y(x_values, w)
            ax.plot(x_values, y_values, color=colors[digits_list[i]], linestyle=linestyles[i], label=f'Retas de separação {digits_list[i]}')


    ax.legend()
    ax.set_xlabel('Intensidade')
    ax.set_ylabel('Simetria')
    ax.set_title(titulo)
    ax.set_xlim([40, 165])
    ax.set_ylim([55, 170]) 
    plt.show()

# _________________________________________________________________________________________________________________

def binary_error(y_true, y_pred):
    return np.sum(y_true != y_pred) / len(y_true)


def acuracy_confusion_matrix(VP: int, VN: int, FP: int, FN:int) -> float:
    return (VP+VN)/(VP+VN+FP+FN)

def plot_confusion_matrix(VP, VN, FP, FN):
    cm = np.array([[VP, FP], [FN, VN]])
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    ax.set(xlabel='Predicted', ylabel='Actual')
    ax.set_title('Confusion Matrix')
    plt.show()


def positive_precision(VP: int, FP: int) -> float:
    if VP + FP != 0:
        return VP / (VP + FP)
    else: 
        return VP


def negative_precision(VN: int, FN:int) -> float:
    if VN + FN != 0:
        return VN / (VN + FN)
    else: 
        return VN


def positive_recall(VP: int, FN: int) -> float:
    if VP + FN != 0:
        return VP / (VP + FN)
    else:
        return VP


def negative_recall(VN: int, FP: int) -> float:
    if VN + FP != 0:
        return VN / (VN + FP)
    else: 
        return VN


def positive_f1_score(VP: int, FP: int, FN: int) -> float:
    try:
        return 2 * positive_precision(VP, FP) * positive_recall(VP, FN)\
        /  (positive_precision(VP, FP) + positive_recall(VP, FN))
    except:
        return  2 * positive_precision(VP, FP) * positive_recall(VP, FN)


def negative_f1_score(VN: int, FP: int, FN: int) -> float:
    try:
        return 2 * negative_precision(VN, FN) * negative_recall(VN, FP)\
        /  (negative_precision(VN, FN) + negative_recall(VN, FP))
    except: 
        return 2 * negative_precision(VN, FN) * negative_recall(VN, FP)


def weighted_f1_score(VP: int, VN: int, FP: int, FN: int, one_label: int, zero_label: int) -> float:
    return (one_label * positive_f1_score(VP, FP, FN) + zero_label * negative_f1_score(VN, FP, FN)) / (one_label + zero_label)


def calculate_VP_VN_FP_FN(y_test, pred):
    VP, VN, FP, FN = 0, 0, 0, 0

    for i in range(len(y_test)):

        if(y_test[i] == 1):
            if(pred[i] == 1):
                VP += 1
            else:
                FN += 1
        else:
            if(pred[i] == 1):
                FP += 1
            else:
                VN += 1

    return VP, VN, FP, FN


def print_metrics(y_test, pred):
    count_label = Counter(y_test)
    print(count_label)
    one_label = count_label[1]
    zero_label = count_label[-1]
    VP, VN, FP, FN = calculate_VP_VN_FP_FN(y_test, pred)
    print("RELATORIO CLASSIFICACAO BINARIA")
    print("one_label: ", one_label)
    print("zero_label: ", zero_label)
    print("VP: " + str(VP))
    print("VN: " + str(VN))
    print("FP: " + str(FP))
    print("FN: " + str(FN))
    print("Binary Error:", binary_error(y_test, pred))
    print("Acurracy:", acuracy_confusion_matrix(VP, VN, FP, FN))
    print("positive precision: ", positive_precision(VP, FP))
    print("negative precision: ", negative_precision(VN, FN))
    print("positive recall: ", positive_recall(VP, FN))
    print("negative recall: ", negative_recall(VN, FP))
    print("positive f1 score: ", positive_f1_score(VP, FP, FN))
    print("negative f1 score: ", negative_f1_score(VN, FP, FN))
    print("weighted_f1_score: ", weighted_f1_score(VP, VN, FP, FN, one_label, zero_label))
    plot_confusion_matrix(VP, VN, FP, FN)
