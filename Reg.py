import re

input_string = "АНАЛЬГИН 5 МГ"

with open('лекстроки.txt', 'r', encoding='utf-8') as dataFile:
    data = []
    for line in dataFile:
        data.append(line.rstrip('\n'))


def main(string):
    # value = re.findall(r'\d+[МГ]\w+', string)
    value = re.findall(r'\d+\s?МГ|\d+\s?МЛ|\d+\s?Г', string)  # для единиц массы и объема в СИ
    return value


num = 0
for element in data:
    num += 1
    print(num, main(element))

