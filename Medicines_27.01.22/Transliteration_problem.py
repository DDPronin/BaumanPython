from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transliterate import get_translit_function
import numpy as np

# NM_CLI vs NM_FULL
# PROD vs GROUP_NM_RUS

import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


class Medicine:
    def __init__(self, CD_DT, ID, NM_CLI, PROD, CD_U,
                 NM_FULL, GROUP_NM_RUS):
        self.CD_DT = CD_DT
        self.ID = ID
        self.NM_CLI = NM_CLI
        self.PROD = PROD
        self.CD_U = CD_U
        self.NM_FULL = NM_FULL
        self.GROUP_NM_RUS = GROUP_NM_RUS

    def get_tokens(self):
        medicine_tokens = list()
        medicine_tokens.append(re.findall(r'[A-ZА-Я]+', self.NM_CLI))
        medicine_tokens.append(re.findall(r'[A-ZА-Я]+', self.NM_FULL))
        medicine_tokens.append(re.findall(r'[A-ZА-Я]+', self.PROD))
        medicine_tokens.append(re.findall(r'[A-ZА-Я]+', self.GROUP_NM_RUS))
        return medicine_tokens

    def get_numbers(self):
        medicine_numbers = list()
        medicine_numbers.append(re.findall(r'\d+', self.NM_CLI))
        medicine_numbers.append(re.findall(r'\d+', self.NM_FULL))
        medicine_numbers.append(re.findall(r'\d+', self.PROD))
        medicine_numbers.append(re.findall(r'\d+', self.GROUP_NM_RUS))
        return medicine_numbers


def read_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as dataFile:
        out = list()
        for line in dataFile:
            data_in_line = line.split('\t')
            medicine = Medicine(data_in_line[0], data_in_line[1], data_in_line[2],
                                data_in_line[3], data_in_line[4], data_in_line[5],
                                data_in_line[6])
            out.append(medicine)
    return out


def sigmoid(x):
    return 1 / (1 + np.exp(-3 * (20 * x - 1)))


def word_eq(in1, in2):
    if len(in1) == 0 or len(in2) == 0:
        return 0
    translit = get_translit_function('ru')
    in1 = list(map(lambda word: translit(word, reversed=False), in1))
    in2 = list(map(lambda word: translit(word, reversed=False), in2))
    for word1 in in1:
        for word2 in in2:
            if word1 == word2:
                return 1
    input_list = list()
    print(in1)
    print(in2)
    input_list.append(' '.join(in1))
    input_list.append(' '.join(in2))
    print(input_list)
    vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2, 4)).fit(input_list)
    in1 = vectorizer.transform(in1)
    in2 = vectorizer.transform(in2)
    matrix = cosine_similarity(in1, in2)
    summa = 0
    num = 0
    for line in matrix:
        for number in line:
            summa += number
            num += 1
    print(summa)
    print(num)
    return sigmoid(summa / num)


data = read_file('data.txt')

result_NM = []  # результаты сравнений обрабатываются отдельно
result_PROD = []

for medicine in data:
    medicine_tokens = medicine.get_tokens()
    medicine_numbers = medicine.get_numbers()

    # NM_CLI vs NM_FULL
    token_comp_NM = word_eq(medicine_tokens[0], medicine_tokens[1])
    result_NM.append([token_comp_NM, medicine])
    # Конечная оценка ср.ар. полученной пары значений с чисел и токенов

    # PROD vs GROUP_NM_RUS
    token_comp_PROD = word_eq(medicine_tokens[2], medicine_tokens[3])
    result_PROD.append([token_comp_PROD, medicine])
    # Конечная оценка ср.ар. полученной пары значений с чисел и токенов

result_NM.sort(key=lambda result_NM: result_NM[0], reverse=False)  # сорт. в обратном порядке
result_PROD.sort(key=lambda result_PROD: result_PROD[0], reverse=False)

for element in tqdm(result_NM):
    print(str(element[0]) + ' - ' + element[1].NM_CLI + ' vs ' + element[1].NM_FULL)
    # num1.append(element[0])
    with open('NM_eq_test.txt', "a", encoding="utf-8") as file:
        file.write(str(element[0]) + ' - ' + element[1].NM_CLI + ' vs ' + element[1].NM_FULL + '\n')
# graf(range(100000-1), num1)
# print(sum(num1)/99999)

# вывести результат второго PROD сравнения
# num2 = []
for element in result_PROD:
    print(str(element[0]) + ' - ' + element[1].PROD + ' vs ' + element[1].GROUP_NM_RUS)
    # num2.append(element[0])
    with open('PROD_eq_test.txt', "a", encoding="utf-8") as file:
        file.write(str(element[0]) + ' - ' + element[1].PROD + ' vs ' + element[1].GROUP_NM_RUS)
# graf(range(100000-1), num2)
# print(sum(num2)/99999)
