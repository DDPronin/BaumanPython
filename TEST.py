def ratio(a, b):
    return a / b



# n = int(input())
# factorial = 1
# if n != 0:
#     for i in range(1, n+1):
#         factorial *= i
# print(factorial)



# supply_value = 100
# supply = []
# element = int(input())
# while supply_value - element >= 0:
#     supply_value -= element
#     supply.append(element)
#     element = int(input())
# print(supply)


# number = int(input())
# if number % 2 == 0:
#     print(number / 2)
# elif number % 3 == 0:
#     print(number / 3)
# elif number % 5 == 0:
#     print(number / 5)
# else:
#     print(number)
#




def odd_sum(numbers):
    _sum = 0
    for number in numbers:
        if number % 2 == 1:
            _sum += number
    return _sum
