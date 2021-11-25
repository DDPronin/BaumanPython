a = float(input())
b = float(input())
c = float(input())
if a + b <= c or a + c <= b or c + b <= a:
    print('Треугольника не существует')
else:
    if a == b or b == c or c == a:
        print('Треугольник равнобедренный ')    
        if a ** 2 + b ** 2 == c ** 2 or a ** 2 + c ** 2 == b ** 2  or b ** 2 + c ** 2 == a ** 2:
            print('Треугольник прямоугольный ')
        elif a ** 2 + b ** 2 < c ** 2 or a ** 2 + c ** 2 < b ** 2  or b ** 2 + c ** 2 < a ** 2:
            print('Треугольник тупоугольный ')
        else:
            print('Треугольник остроугольный ')
    elif a == b and b == c:
        print('Треугольник равноcторонний ')
        print('Треугольник остроугольный ')
    else:
        if a ** 2 + b ** 2 == c ** 2 or a ** 2 + c ** 2 == b ** 2  or b ** 2 + c ** 2 == a ** 2:
            print('Треугольник прямоугольный ')
        elif a ** 2 + b ** 2 < c ** 2 or a ** 2 + c ** 2 < b ** 2  or b ** 2 + c ** 2 < a ** 2:
            print('Треугольник тупоугольный ')
        else:
            print('Треугольник остроугольный ')
