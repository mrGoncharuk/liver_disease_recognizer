# coding=utf-8
import numpy as np
import pandas as pd
import cv2
import os
import time
from multiprocessing import Pool



def getPixelMatrix(filename):
    arr = cv2.imread(filename, 0)
    return arr.astype(int)


def getGLCM(arr):
    a1 = []
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            a1.append(arr[i][j])
    a2 = []
    for i in range(1, len(a1)):
        a2.append(a1[i])
    del a1[-1]
    strA = []
    for i in range(len(a1)):
        strA.append(str(a1[i]) + str(a2[i]))
    a3 = []
    for i in range(len(strA)):
        a3.append(strA.count(strA[i]))
    df = pd.DataFrame({'x': a1, 'y': a2, 'z': a3})
    df = df.drop_duplicates()
    return df.sort_values(by=['x'])


def getX1X2X3(df):
    newList = []
    a1 = df['x'].tolist()
    a2 = df['y'].tolist()
    a3 = df['z'].tolist()
    max_index = a3.index(max(a3))
    result = int(a1[max_index]) * int(a2[max_index])
    newList.append(result)  # первый признак (x1)
    norm_value = pd.Series(a3)
    norm_value = (norm_value / sum(norm_value) * 10000).tolist()
    tp = np.array(a1)
    arr_index = np.where(tp == 53)[0]
    result2_array = [norm_value[j] for j in arr_index]
    try:
        result2 = sum(result2_array) / len(result2_array)
    except:
        result2 = 0
    newList.append(result2)  # второй признак (x2)
    result3 = int(max(a1)) * int(max(a2))
    newList.append(result3)  # третий признак (x3)
    return newList



def calculate_norma(load_folder1, excel_file_name):
    n = len(os.listdir(load_folder1))
    n = 3
    print(n)
    list_to_save = []
    # Norma
    for iteration in range(n):
        newList = []
        filename = load_folder1 + str(iteration + 1) + '.png'
        arr = getPixelMatrix(filename)  # матрица градаций серого
        df = getGLCM(arr)  # GLCM
        xList = getX1X2X3(df)
        newList.append(xList[0])
        newList.append(xList[1])
        newList.append(xList[2])
        newList.append(1)
        list_to_save.append(newList)
    
    df = pd.DataFrame(list_to_save)
    df.to_excel(excel_file_name + "_norm.xlsx", index=False)

    return list_to_save


def calculate_pathology(load_folder2, excel_file_name):
    n = len(os.listdir(load_folder2))
    n = 3
    print(n)
    list_to_save = []
    # Pathology
    for iteration in range(n):
        newList = []
        filename = load_folder2 + str(iteration + 1) + '.png'
        arr = getPixelMatrix(filename)  # матрица градаций серого
        df = getGLCM(arr)  # GLCM
        xList = getX1X2X3(df)
        newList.append(xList[0])
        newList.append(xList[1])
        newList.append(xList[2])
        newList.append(2)
        list_to_save.append(newList)
    df = pd.DataFrame(list_to_save)
    df.to_excel(excel_file_name + "_path.xlsx", index=False)
    return list_to_save

def getAll(load_folder1, load_folder2, excel_file_name):
    listToSave = []
    listToSave = calculate_norma(load_folder1, excel_file_name)
    listToSave += calculate_pathology(load_folder2, excel_file_name)
    # df = pd.DataFrame(listToSave)
    # df.to_excel(excel_file + ".xlsx", index=False)


start_time = time.time()
getAll("/home/mhoncharuk/Education/liver_disease_recognizer/ROI/Norma/CL/",
       "/home/mhoncharuk/Education/liver_disease_recognizer/ROI/Norma/CL/",
       "convex1")
# print("Time: ", time.time() - start_time)
# start_time = time.time()
# getAll("D:\\Studying\\KPI\\DataForPreprocessing\\Norma\\Linear\\",
#        "D:\\Studying\\KPI\\DataForPreprocessing\\Pathology\\Linear\\",
#        "linear1")
# print("Time: ", time.time() - start_time)
# start_time = time.time()
# getAll("D:\\Studying\\KPI\\DataForPreprocessing\\Norma\\Balls\\",
#        "D:\\Studying\\KPI\\DataForPreprocessing\\Pathology\\Balls\\",
#        "balls1")
# print("Time: ", time.time() - start_time)
