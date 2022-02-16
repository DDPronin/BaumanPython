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


def number_set_comparation(in1, in2):
    set1 = set(in1)
    set2 = set(in2)
    if len(set1) == 0 or len(set2) == 0:
        return -1  # отсутствие чисел не дает информации, считаем "вес" пустого поля за 0.5
    else:
        return 2 * len(set1 & set2) / (len(set1) + len(set2))


def cosine_comparation(in1, in2):
    processed_element1 = list()
    for token in in1:
        for i in range(len(token) - 1):
            processed_element1.append(token[i] + token[i + 1])
    processed_element2 = list()
    for token in in2:
        for i in range(len(token) - 1):
            processed_element2.append(token[i] + token[i + 1])
    if len(processed_element1) == 0 or len(processed_element2) == 0:
        return 0  # считаем незаполненные поля однозначной ошибкой
    else:
        input_list = list()
        input_list.append(' '.join(processed_element1))
        input_list.append(' '.join(processed_element2))
        print(input_list)  # разбиваем токены так
        vectorizer = CountVectorizer().fit_transform(input_list)
        vectors = vectorizer.toarray()
        return cosine_similarity(vectors)[0][1]


def cosine_comparation2(in1, in2):
    if len(in1) == 0 or len(in2) == 0:
        return 0
    else:
        input_list = list()
        input_list.append(' '.join(in1))
        input_list.append(' '.join(in2))
        print(input_list)
        vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2, 2)).fit_transform(input_list)
        vectors = vectorizer.toarray()
        return cosine_similarity(vectors)[0][1]


def word_eq_sigmoid(x):
    return 1 / (1 + np.exp(-3 * (20 * x - 1)))


def every_word_sigmoid(x):
    return 1 / (1 + np.exp(-3 * (3 * x - 1.8)))


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
        return every_word_sigmoid(maximum)


def graf(x, y):
    plt.axis([0, 100000, 0, 100000])
    plt.plot(x, y, 'ro')
    plt.show()


data = read_file('data.txt')

#weights_NM1 = [0.3, 0.2, 0.25, 0.25]
#weights_NM2 = [0.3, 0.2, 0.15, 0.15, 0.2]

weights_NM2 = [0, 0, 1, 0, 0]
weights_NM1 = [0, 1, 0, 0]

#weights_PROD1 = [0.2, 0.3, 0.2, 0.3]
#weights_PROD2 = [0.2, 0.1, 0.3, 0.2, 0.2]

weights_PROD2 = [0, 0, 1, 0, 0]
weights_PROD1 = [0, 1, 0, 0]

result_NM = []  # результаты сравнений обрабатываются отдельно
result_PROD = []  # можно создать какую-либо метрику для обобщения двух сравнений
result = []
for medicine in tqdm(data):

    iteration_sum_NM = 0
    iteration_sum_PROD = 0
    medicine_tokens = medicine.get_tokens()
    medicine_numbers = medicine.get_numbers()

    # NM_CLI vs NM_FULL
    token_comp_NM = cosine_comparation2(medicine_tokens[0], medicine_tokens[1])
    number_comp_NM = number_set_comparation(medicine_numbers[0], medicine_numbers[1])
    word_eq_ru_NM = word_eq_filter(medicine_tokens[0], medicine_tokens[1], False)
    every_word_metric_NM = every_word_metric(medicine_tokens[0], medicine_tokens[1], False)
    levenshtein_metric_NM = levenshtein_metric(medicine_tokens[0], medicine_tokens[1], False)
    if number_comp_NM == -1:
        iteration_sum_NM += token_comp_NM * weights_NM1[0] + \
                        word_eq_ru_NM * weights_NM1[1] + \
                        every_word_metric_NM * weights_NM1[2] + \
                        levenshtein_metric_NM * weights_NM1[3]
        result_NM.append([iteration_sum_NM, medicine])
    else:
        iteration_sum_NM += token_comp_NM * weights_NM2[0] + \
                         number_comp_NM * weights_NM2[1] + \
                         word_eq_ru_NM * weights_NM2[2] + \
                         every_word_metric_NM * weights_NM2[3] +\
                         levenshtein_metric_NM * weights_NM2[4]
        result_NM.append([iteration_sum_NM, medicine])
    # Конечная оценка ср.ар. полученной пары значений с чисел и токенов

    # PROD vs GROUP_NM_RUS
    token_comp_PROD = cosine_comparation2(medicine_tokens[2], medicine_tokens[3])
    number_comp_PROD = number_set_comparation(medicine_numbers[2], medicine_numbers[3])
    word_eq_ru_PROD = word_eq_filter(medicine_tokens[2], medicine_tokens[3], False)
    every_word_metric_PROD = every_word_metric(medicine_tokens[2], medicine_tokens[3], False)
    levenshtein_metric_PROD = levenshtein_metric(medicine_tokens[2], medicine_tokens[3], False)
    if number_comp_PROD == -1:
        iteration_sum_PROD = token_comp_PROD * 0.2 + \
                             word_eq_ru_PROD * 0.3 +\
                             every_word_metric_PROD * 0.2 + \
                             levenshtein_metric_PROD * 0.3
        result_PROD.append([iteration_sum_PROD,
                            medicine])
    else:
        iteration_sum_PROD = token_comp_PROD * 0.2 + \
                             number_comp_PROD * 0.1 + \
                             word_eq_ru_PROD * 0.3 + \
                             every_word_metric_PROD * 0.2 + \
                             levenshtein_metric_PROD * 0.2
        result_PROD.append([iteration_sum_PROD,
                            medicine])
        # Конечная оценка ср.ар. полученной пары значений с чисел и токенов
    result.append([(iteration_sum_NM + iteration_sum_PROD)/2, medicine])

result.sort(key=lambda result: result[0], reverse=False)
result_NM.sort(key=lambda result_NM: result_NM[0], reverse=False)  # сорт. в обратном порядке
result_PROD.sort(key=lambda result_PROD: result_PROD[0], reverse=False)  # чтобы сразу видеть ошибки

# вывести результат первого NM сравнения
#num1 = []
# for element in tqdm(result_NM):
#     print(str(element[0]) + ' - ' + element[1].NM_CLI + ' vs ' + element[1].NM_FULL)
#     #num1.append(element[0])
#     with open('NM_with_5_filters_final3.txt', "a", encoding="utf-8") as file:
#         file.write(str(float(element[0])) + '\t' + element[1].CD_DT + '\t' +
#                    element[1].ID + '\t' + element[1].NM_CLI + '\t' + element[1].CD_U + '\t' +
#                    element[1].NM_FULL + '\n')
# #graf(range(100000-1), num1)
# # print(sum(num1)/99999)
#
# # вывести результат второго PROD сравнения
# # num2 = []
# for element in tqdm(result_PROD):
#     print(str(element[0]) + ' - ' + element[1].PROD + ' vs ' + element[1].GROUP_NM_RUS)
#     # num2.append(element[0])
#     with open('PROD_with_5_filters_final3.txt', "a", encoding="utf-8") as file:
#         file.write(str(float(element[0])) + '\t' + element[1].CD_DT + '\t' +
#                    element[1].ID + '\t' + element[1].PROD + '\t' + element[1].CD_U + '\t' + element[1].GROUP_NM_RUS + '\n')
# # graf(range(100000-1), num2)
# # print(sum(num2)/99999)

for element in tqdm(result):
    with open('NM+PROD.txt', "a", encoding="utf-8") as file:
        file.write(str(float(element[0])) + '\t' + element[1].CD_DT + '\t' +
                   element[1].ID + '\t' + element[1].NM_CLI + '\t' + element[1].PROD + '\t' + element[1].CD_U +
                   '\t' + element[1].NM_FULL + '\t' + element[1].GROUP_NM_RUS)