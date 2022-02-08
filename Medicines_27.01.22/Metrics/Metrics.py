# NM_CLI vs NM_FULL
# PROD vs GROUP_NM_RUS

import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from transliterate import get_translit_function
import Levenshtein
import numpy as np


class Medicine:
    def __init__(self, CD_DT = '', ID = '', NM_CLI = '', PROD = '', CD_U = '',
                 NM_FULL = '', GROUP_NM_RUS = ''):
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


def every_word_sigmoid(x):
    return 1 / (1 + np.exp(-3 * (3 * x - 1.8)))


def levenshtein_metric(in1, in2, mode):
    if len(in1) == 0 or len(in2) == 0:
        return 0
    else:
        translit = get_translit_function('ru')
        in1 = map(lambda word: translit(word, reversed=mode), in1)
        in2 = map(lambda word: translit(word, reversed=mode), in2)
        # in1, in2 = list(''.join(in1).split(' ')),  list(''.join(in2).split(' '))
        maximum = 0
        in1, in2 = list(in1), list(in2)
        maximum = 0
        for word1 in in1:
            for word2 in in2:
                if word1 == word2:
                    return 1
                elif len(word1) > 4 and len(word2) > 4:
                    result = 1 - (2 * Levenshtein.distance(word1, word2) /
                                  (len(word1) + len(word2)))
                    if result > maximum:
                        maximum = result
        return maximum


def every_word_metric(in1, in2, mode):
    if len(in1) == 0 or len(in2) == 0:
        return 0
    else:
        translit = get_translit_function('ru')
        in1 = map(lambda word: translit(word, reversed=mode), in1)
        in2 = map(lambda word: translit(word, reversed=mode), in2)
        # in1, in2 = list(''.join(in1).split(' ')),  list(''.join(in2).split(' '))
        maximum = 0
        in1, in2 = list(in1), list(in2)
        for word1 in in1:
            for word2 in in2:
                if word1 == word2:
                    return 1
                elif len(word1) > 4 and len(word2) > 4:
                    vectorizer = CountVectorizer(analyzer='char_wb',
                                                ngram_range=(2, 2)).fit([word1 + word2])
                    vec1 = vectorizer.transform([word1])
                    vec2 = vectorizer.transform([word2])
                    result = cosine_similarity(vec1, vec2)
                    if result > maximum:
                        maximum = result
        return every_word_sigmoid(maximum)


def word_eq_sigmoid(x):
    return 1 / (1 + np.exp(-3 * (20 * x - 1)))


def word_eq_filter(in1, in2, mode):
    if len(in1) == 0 or len(in2) == 0:
        return 0
    translit = get_translit_function('ru')
    in1 = list(map(lambda word: translit(word, reversed=mode), in1))
    in2 = list(map(lambda word: translit(word, reversed=mode), in2))
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
    return word_eq_sigmoid(summa / num)

data = read_file('data.txt')
result_NM = []
result_PROD = []
for medicine in data:
    medicine_tokens = medicine.get_tokens()
    print(medicine_tokens)
    every_word_metric_NM = word_eq_filter(medicine_tokens[0],
                                              medicine_tokens[1], False)
    every_word_metric_PROD = word_eq_filter(medicine_tokens[2],
                                              medicine_tokens[3], False)
    result_NM.append([every_word_metric_NM, medicine])
    result_PROD.append([every_word_metric_PROD, medicine])

result_NM.sort(key=lambda result_NM: result_NM[0], reverse=False)
result_PROD.sort(key=lambda result_PROD: result_PROD[0], reverse=False)

for element in tqdm(result_NM):
    print(str(element[0]) + ' - ' + element[1].NM_CLI + ' vs ' + element[1].NM_FULL)
    # num1.append(element[0])
    with open('NM_word_eq_metric.txt', "a", encoding="utf-8") as file:
        file.write(str(element[0]) + ' - ' + element[1].NM_CLI + ' vs ' + element[1].NM_FULL + '\n')

for element in tqdm(result_PROD):
    print(str(element[0]) + ' - ' + element[1].PROD + ' vs ' + element[1].GROUP_NM_RUS)
    # num2.append(element[0])
    with open('PROD_word_eq_metric.txt', "a", encoding="utf-8") as file:
        file.write(str(element[0]) + ' - ' + element[1].PROD + ' vs ' + element[1].GROUP_NM_RUS)
