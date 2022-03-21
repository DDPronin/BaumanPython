# NM_CLI vs NM_FULL
# PROD vs GROUP_NM_RUS

import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from transliterate import get_translit_function
import numpy as np
import Levenshtein


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
        medicine_numbers = [[], [], [], []]
        medicine_numbers[0] = (re.findall(r'\d+', self.NM_CLI))
        medicine_numbers[1] = (re.findall(r'\d+', self.NM_FULL))
        medicine_numbers[2] = (re.findall(r'\d+', self.PROD))
        medicine_numbers[3] = (re.findall(r'\d+', self.GROUP_NM_RUS))
        out = []
        preout = []
        for element in medicine_numbers:
            for number in element:
                number = int(number)
                if number != 0:
                    if number < 1:
                        while number < 1:
                            number = number * 10
                    if number > 1:
                        while number % 10 == 0:
                            number = number / 10
                    preout.append(str(int(number)))
            out.append(preout)
            preout = []
        return out

    def get_N_index(self):
        N_index = list()
        N_index.append(re.findall(r'[N№] ?\d+', self.NM_CLI))
        N_index.append(re.findall(r'[N№] ?\d+', self.NM_FULL))
        out = []
        for element in N_index:
            out.append(re.findall(r'\d+', str(element)))
        return out


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


def word_eq_sigmoid(x):
    return 1 / (1 + np.exp(-3 * (20 * x - 1)))


def every_word_sigmoid(x):
    return 1 / (1 + np.exp(-3 * (3 * x - 1.8)))


def number_comparation(in1, in2):
    set1 = set(in1)
    set2 = set(in2)
    print(in1, in2)
    if len(set1) == 0 or len(set2) == 0:
        return -1
    else:
        return 2 * len(set1 & set2) / (len(set1) + len(set2))


def cosine_comparation(in1, in2):
    if len(in1) == 0 or len(in2) == 0:
        return 0
    else:
        input_list = list()
        input_list.append(' '.join(in1))
        input_list.append(' '.join(in2))
        # print(input_list)
        vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2, 2)).fit_transform(input_list)
        vectors = vectorizer.toarray()
        return cosine_similarity(vectors)[0][1]


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
    # print(in1)
    # print(in2)
    input_list.append(' '.join(in1))
    input_list.append(' '.join(in2))
    # print(input_list)
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
    # print(summa)
    # print(num)
    return summa / num


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
                if word1 == word2 and len(word1) > 3 and len(word2) > 3:
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
                if word1 == word2 and len(word1) > 3 and len(word2) > 3:
                    return 1
                elif len(word1) > 4 and len(word2) > 4:
                    vectorizer = CountVectorizer(analyzer='char_wb',
                                                ngram_range=(2, 2)).fit([word1 + word2])
                    vec1 = vectorizer.transform([word1])
                    vec2 = vectorizer.transform([word2])
                    result = cosine_similarity(vec1, vec2)
                    if result > maximum:
                        maximum = result
        return maximum

data = read_file('data_learning.txt')


for medicine in tqdm(data):

    medicine_tokens = medicine.get_tokens()
    medicine_numbers = medicine.get_numbers()
    medicine_N_index = medicine.get_N_index()

    cosine_comp_NM = cosine_comparation(medicine_tokens[0], medicine_tokens[1])
    number_comp_NM = number_comparation(medicine_numbers[0], medicine_numbers[1])
    N_index_comp = number_comparation(medicine_N_index[0], medicine_N_index[1])
    word_eq_ru_NM = word_eq_filter(medicine_tokens[0], medicine_tokens[1], False)
    every_word_metric_NM = every_word_metric(medicine_tokens[0], medicine_tokens[1], False)
    levenshtein_metric_NM = levenshtein_metric(medicine_tokens[0], medicine_tokens[1], False)

    cosine_comp_PROD = cosine_comparation(medicine_tokens[2], medicine_tokens[3])
    word_eq_ru_PROD = word_eq_filter(medicine_tokens[2], medicine_tokens[3], False)
    every_word_metric_PROD = every_word_metric(medicine_tokens[2], medicine_tokens[3], False)
    levenshtein_metric_PROD = levenshtein_metric(medicine_tokens[2], medicine_tokens[3], False)

    if number_comp_NM == -1:
        number_comp_NM = cosine_comp_NM
    if N_index_comp == -1:
        N_index_comp = number_comp_NM

    with open('data_learning_filters_score_NM.txt', "a", encoding="utf-8") as file:
        file.write(str(float(cosine_comp_NM)) + '\t' + str(float(number_comp_NM)) + '\t' +
                    str(float(N_index_comp)) + '\t' + str(float(word_eq_ru_NM)) + '\t' +
                    str(float(every_word_metric_NM)) + '\t' + str(float(levenshtein_metric_NM)) + '\n')
    with open('data_learning_filters_score_PROD.txt', "a", encoding="utf-8") as file:
        file.write(str(float(cosine_comp_PROD)) + '\t' + str(float(word_eq_ru_PROD)) + '\t' +
                   str(float(every_word_metric_PROD)) + '\t' + str(float(levenshtein_metric_PROD)) + '\n')
