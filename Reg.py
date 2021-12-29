import re

with open('лекстроки.txt', 'r', encoding='utf-8') as dataFile:
    data = []
    for line in dataFile:
        data.append(line.rstrip('\n'))


def main(string):  # для единиц массы и объема СИ
    value = re.findall(r'(\d+[.,]?\d*\s?/?)(МГ/МЛ|МКГ|МКЛ|Г|МИЛЛИГРАММ|ГРАММ|МГ|МЛ)', string)
    return value


num = 0
for element in data:
    num += 1
    out = main(element)
    if not out:
        print(num, ":", element)
        # строки, не содержащие массы и объема, выводятся в предоставленном виде
    else:
        print(num, ":", end=' ')
        for pair in out:
            pair = list(pair)
            pair[0] = pair[0].replace("/", "")
            pair[0] = pair[0].replace(" ", "")
            pair[0] = pair[0].replace(",", ".")
            print(pair[0] + " " + pair[1], end=' ')
        print()
        # численные части строк, содержащих массу и/или объем, приводятся к виду,
        # допускающему преобразование в численный тип (int/ float) и выводятся на экран

