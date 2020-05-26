import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from random import sample
import copy
from scipy.stats import kendalltau


# Мутация
def mutation(population_number, m, k):
    index_list = []
    for i in range(population_number):
        chromosome = sample(range(m - 1), k)
        chromosome.sort()
        index_list.append(chromosome)
    return index_list


# Проверка листа на одинаковость
def check_equal(index_list):
    new_list = []
    for element in index_list:
        if element not in new_list:
            new_list.append(element)
    return len(new_list) == 1


# Проверка хромосомы на дубликаты
def check_duplicates(chromosome):
    new_list = []
    for element in chromosome:
        if element not in new_list:
            new_list.append(element)
    return len(new_list) != len(chromosome)


def crossover(population_number, father_list, mother_list, index_list, temp_list):
    for i in range(population_number):
        j = father_list[i]
        z = mother_list[i]
        for k in range(len(index_list[i])):
            if k % 2 == 0:
                index_list[i][k] = temp_list[j][k]
            else:
                index_list[i][k] = temp_list[z][k]
    return index_list


def get_mothers(population_number, father_list, prob_list):
    mother_list = []
    for i in range(population_number):
        m = tournament_selection(population_number, prob_list)
        while m[0] == father_list[i]:
            m = tournament_selection(population_number, prob_list)
        mother_list.append(m[0])
    return mother_list


# Турнирная селекция
def tournament_selection(population_number, prob_list):
    resList = []
    for i in range(population_number):
        someList = sample(range(len(prob_list)), population_number // 2)
        res = 0
        maxNumber = -1
        for i in range(len(someList)):
            if (prob_list[someList[i]] > maxNumber):
                res = someList[i]
                maxNumber = prob_list[someList[i]]
        resList.append(res)
    return resList


# Нахождение классов исходя из трешхолда
def find_yy(X, i, threshold, side):
    col = X[:, i]
    yy_list = []
    for xx in col:
        if side == 1:
            if xx < threshold:
                yy_list.append(1)
            else:
                yy_list.append(2)
        else:
            if xx < threshold:
                yy_list.append(2)
            else:
                yy_list.append(1)
    return yy_list


# Finding best threshold of single Xi
def find_threshold_of_x(xx, col, y, num_of_pos, num_of_neg):
    threshold_list = []
    value_list1 = []
    value_list2 = []
    for j in range(1, xx.shape[0] - 1):
        TP1 = 0  # number of True Positive
        TN1 = 0  # number of True Negative
        TP2 = 0
        TN2 = 0
        for z in range(col.shape[0]):
            if col[z] < xx[j]:
                if y[z] == 1:
                    TP1 += 1
                else:
                    TN2 += 1
            else:
                if y[z] == 2:
                    TN1 += 1
                else:
                    TP2 += 1
        threshold_list.append(xx[j])
        value_list1.append(((TP1 / num_of_pos) + (TN1 / num_of_neg)) / 2)
        value_list2.append(((TP2 / num_of_pos) + (TN2 / num_of_neg)) / 2)
    if max(value_list1) > max(value_list2):
        threshold = threshold_list[value_list1.index(max(value_list1))]
        side = 1
    else:
        threshold = threshold_list[value_list2.index(max(value_list2))]
        side = 2
    return threshold, side


# Finding threshold of each Xi
def find_thresholds(chromosome, X_train, y_train):
    num_of_pos = np.sum(y_train == 1)  # number of positive objects (norma)
    num_of_neg = np.sum(y_train == 2)  # number of negative objects (pathology)
    threshold_list = []
    side_list = []
    for i in chromosome:
        col = X_train[:, i]
        xx = copy.deepcopy(col)  # make copy of Xi to not change initial list
        xx.sort()

        # get threshold of Xi, its value on train sample and side of threshold
        threshold, side = find_threshold_of_x(xx, col, y_train, num_of_pos, num_of_neg)

        threshold_list.append(threshold)
        side_list.append(side)
    return threshold_list, side_list


# Расчёт QCFE - Quality criterion of features ensemble (критерий качества ансамбля признаков)
def calculate_qcfe(chromosome, X_train, y_train, X_test, y_test, test_weight, alpha):
    threshold_list, side_list = find_thresholds(chromosome, X_train, y_train)
    yy_train_list = []
    yy_test_list = []
    j = 0
    for i in chromosome:
        yy_train = find_yy(X_train, i, threshold_list[j], side_list[j])
        yy_train_list.append(yy_train)
        yy_test = find_yy(X_test, i, threshold_list[j], side_list[j])
        yy_test_list.append(yy_test)
        j += 1

    # рассчёт первого числителя из формулы Павлова
    corr_list = []
    for i in range(len(yy_train_list)):
        corr_train, p_value = kendalltau(yy_train_list[i], list(y_train))
        corr_test, p_value = kendalltau(yy_test_list[i], list(y_test))
        corr_list.append((1 - test_weight) * abs(corr_train) + test_weight * abs(corr_test))
    first_value = (alpha * sum(corr_list)) / (2 * len(corr_list))  # степень зависимости признаков

    # рассчёт второго числителя из формулы Павлова
    corr_list = []
    i = 0
    j = 1
    while i != len(chromosome) - 1:
        z = j
        while z < len(chromosome):
            corr_train, p_value = kendalltau(yy_train_list[i], yy_train_list[z])
            corr_test, p_value = kendalltau(yy_test_list[i], yy_test_list[z])
            corr_list.append((1 - test_weight) * abs(corr_train) + test_weight * abs(corr_test))
            z += 1
        i += 1
        j += 1
    second_value = (1 - alpha) / (2 + (sum(corr_list) / len(corr_list)))  # степень независимости признаков

    return first_value + second_value


# Генерирование первой популяции
def generate_first_population(population_number, m, k):
    index_list = []
    for i in range(population_number):
        chromosome = sample(range(m - 1), k)
        chromosome.sort()
        index_list.append(chromosome)
    return index_list


def genetic_algorithm(data, col_names, k, population_number, test_weight, alpha):
    y = data['class'].values  # классы печени
    X = data.drop(['class'], axis=1).values  # признаки

    # разбиение выборки на обучение и тест
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=0)

    m = len(col_names)  # общее количество признаков

    # лист индексов, который будет первой популяцией наших особей
    index_list = generate_first_population(population_number, m, k)

    best_qcfe = -1
    best_qcfe_list = []
    bfe_list = []  # best features ensemble list

    # ГЕНЕТИЧЕСКИЙ АЛГОРИТМ
    for index in range(200):
        qcfe_list = []
        for chromosome in index_list:
            qcfe = calculate_qcfe(chromosome, X_train, y_train, X_test, y_test, test_weight, alpha)
            qcfe_list.append(qcfe)

        max_qcfe = -1
        max_index = -1
        for i in range(len(index_list)):
            if qcfe_list[i] > max_qcfe:
                max_qcfe = qcfe_list[i]
                max_index = i
        if max_qcfe > best_qcfe:
            best_qcfe = max_qcfe
            best_qcfe_list.append(best_qcfe)
            bfe_list.append(index_list[max_index])

        prob_list = []
        for qcfe in qcfe_list:
            prob_list.append(qcfe / sum(qcfe_list))

        # Селекция особей (отцов и матерей)
        father_list = tournament_selection(population_number, prob_list)
        mother_list = get_mothers(population_number, father_list, prob_list)

        # Кроссовер
        temp_list = copy.deepcopy(index_list)
        index_list = crossover(population_number, father_list, mother_list, index_list, temp_list)
        
        for chromosome in index_list:
            chromosome.sort()
        
        for m in range(len(index_list)):
            if check_duplicates(index_list[m]):
                index_list[m] = sample(range(n - 1), k)
                index_list[m].sort()

        if check_equal(index_list):
            index_list = mutation(population_number, m, k)

    return best_qcfe_list, bfe_list


## всего есть 3 варианта датчика ##
## 'convex' - конвексный датчик ##
## 'linear' - линейныый датчик ##
## 'balls' - усиленный линейный датчик (яичко) ##
sensor_type = 'linear'  # тип датчика

name_of_train = sensor_type + '(train).xlsx'  # выборка данных
data = pd.read_excel(name_of_train)  # загрузка данных
col_names = list(data.columns[:-1])  # названия признаков
k = 10  # размер ансамбля
population_number = 8  # количество особей в популяции
test_weight = 1  # вес тестовой выборки
alpha = 0.1  # вес зависимости признаков
best_qcfe_list, bfe_list = genetic_algorithm(data, col_names, k, population_number, test_weight, alpha)

# вывод результатов (наилучшие ансамбли признаков и их значений QCFE)
for i in range(len(best_qcfe_list)):
    text = ''
    for index in bfe_list[i]:
        text += col_names[index] + ';'
    print(text)
    print(best_qcfe_list[i])
    print()
