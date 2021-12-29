import re

input_string = "АНАЛЬГИН 5 МГ"


def main(string):
    value = re.findall(r'\d\s\w+', string)
    return value


print(main(input_string))
