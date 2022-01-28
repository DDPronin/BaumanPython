# NM_CLI vs NM_FULL
# PROD vs GROUP_NM_RUS

import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
    if len(set1) + len(set2) == 0:
        return 0.5  # отсутствие чисел не дает информации, считаем "вес" пустого поля за 0.5
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


data = read_file('data.txt')

result_NM = []  # результаты сравнений обрабатываются отдельно
result_PROD = []  # можно создать какую-либо метрику для обобщения двух сравнений
for medicine in data:
    medicine_tokens = medicine.get_tokens()
    medicine_numbers = medicine.get_numbers()

    # NM_CLI vs NM_FULL
    token_comp_NM = cosine_comparation(medicine_tokens[0], medicine_tokens[1])
    number_comp_NM = number_set_comparation(medicine_numbers[0], medicine_numbers[1])
    result_NM.append([(token_comp_NM + number_comp_NM) / 2, medicine])
    # Конечная оценка ср.ар. полученной пары значений с чисел и токенов

    # PROD vs GROUP_NM_RUS
    token_comp_PROD = cosine_comparation(medicine_tokens[2], medicine_tokens[3])
    number_comp_PROD = number_set_comparation(medicine_numbers[2], medicine_numbers[3])
    result_PROD.append([(token_comp_PROD + number_comp_PROD) / 2, medicine])
    # Конечная оценка ср.ар. полученной пары значений с чисел и токенов

result_NM.sort(key=lambda result_NM: result_NM[0], reverse=True)  # сорт. в обратном порядке
result_PROD.sort(key=lambda result_PROD: result_NM[0], reverse=True)  # чтобы сразу видеть ошибки

# вывести результат первого NM сравнения
for element in result_NM:
    print(str(element[0]) + ' - ' + element[1].NM_CLI + ' vs ' + element[1].NM_FULL)

# вывести результат второго PROD сравнения
# for element in result_PROD:
#     print(str(element[0]) + ' - ' + element[1].PROD + ' vs ' + element[1].GROUP_NM_RUS)

