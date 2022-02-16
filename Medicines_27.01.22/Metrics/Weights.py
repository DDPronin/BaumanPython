import itertools as it
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from transliterate import get_translit_function
import Levenshtein
import numpy as np

weights = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]

weights_combinations = list(it.combinations_with_replacement(weights, 5))

preresult = []
for combination in weights_combinations:
    if round(sum(combination), 3) == 1:
        preresult.append(list(it.permutations(combination, 5)))

result = []
for group in preresult:
    for combination in group:
        if combination not in result:
            result.append(combination)

# ------ #

class Medicine:
    def __init__(self, SCORE = '', CD_DT = '', ID = '', NM_CLI = '', PROD = '', CD_U = '',
                 NM_FULL = '', GROUP_NM_RUS = ''):
        self.SCORE = SCORE
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
            medicine = Medicine()
            medicine.SCORE = float(data_in_line[0])
            medicine.NM_CLI = data_in_line[1]
            medicine.NM_FULL = data_in_line[2]
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




data = read_file("learn_data.txt")

result_NM = []
for solution in tqdm(result):
    errors_NM = []
    for medicine in data:

        medicine_tokens = medicine.get_tokens()
        medicine_numbers = medicine.get_numbers()

        # NM_CLI vs NM_FULL
        token_comp_NM = cosine_comparation2(medicine_tokens[0], medicine_tokens[1])
        number_comp_NM = number_set_comparation(medicine_numbers[0], medicine_numbers[1])
        word_eq_ru_NM = word_eq_filter(medicine_tokens[0], medicine_tokens[1], False)
        every_word_metric_NM = every_word_metric(medicine_tokens[0], medicine_tokens[1], False)
        levenshtein_metric_NM = levenshtein_metric(medicine_tokens[0], medicine_tokens[1], False)
        if number_comp_NM == -1:
            errors_NM.append(abs(token_comp_NM * solution[0] + word_eq_ru_NM * solution[2] +
                              every_word_metric_NM * solution[3] + levenshtein_metric_NM * solution[4]
                              - medicine.SCORE))
        else:
            errors_NM.append(abs(token_comp_NM * solution[0] + number_comp_NM * solution[1] +
                              word_eq_ru_NM * solution[2] + every_word_metric_NM * solution[3] +
                              levenshtein_metric_NM * solution[4] - medicine.SCORE))
    result_NM.append([sum(errors_NM), solution])
result_NM.sort(key=lambda result_NM: result_NM[0], reverse=False)
for element in result_NM:
    with open('weights_res.txt', "a", encoding="utf-8") as file:
        file.write(str(element[0]) + ' --- ' + str(element[1]) + '\n')
