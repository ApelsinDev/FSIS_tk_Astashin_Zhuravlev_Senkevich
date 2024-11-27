
import numpy as np
from itertools import combinations
import random as rnd
import math

# Функция для получения базисного порядка
def get_basis_order(n, m):
    # Генерация всех чисел от 0 до n-1 в двоичном представлении
    ans = []
    for i in range(n):
        binary = f'{i:b}'[::-1]  # Перевод в двоичное представление и разворот
        if len(binary) < m:
            binary += '0' * (m - len(binary))  # Добавление недостающих нулей
        ans.append(binary)
    return ans

# Функция для получения порядка векторов
def get_vectors_order(r, m):
    elements = list(range(m))
    ans = []
    # Генерация всех комбинаций длиной от 0 до r включительно
    for i in range(r + 1):
        combinations_list = sorted(list(combinations(elements, i)), reverse=True)
        for combination in combinations_list:
            ans.append(list(combination))
    return ans

# Функция для формирования порождающей матрицы кода Рида-Маллера
def get_rm_G_matr(r, m):
    n = 2 ** m  # Длина кода
    basis_order = get_basis_order(n, m)  # Базисный порядок
    vectors_order = get_vectors_order(r, m)  # Порядок векторов
    g_matr = np.zeros((len(vectors_order), n), dtype=int)

    # Формирование матрицы
    for i in range(g_matr.shape[0]):
        for j in range(g_matr.shape[1]):
            flag = True
            for indx in vectors_order[i]:
                if basis_order[j][indx] == '1':
                    g_matr[i][j] = 0
                    flag = False
                    break
            if flag:
                g_matr[i][j] = 1

    print("Промежуточный вывод: порождающая матрица кода Рида-Маллера")
    print(g_matr)
    print("Промежуточный вывод: базисный порядок")
    print(basis_order)
    print("Промежуточный вывод: порядок векторов")
    print(vectors_order)

    return g_matr, basis_order, vectors_order

# Функция для нахождения дополнения множества
def get_complement(m, I):
    Zm = list(range(m))
    return [i for i in Zm if i not in I]

# Функция для вычисления множества Hj
def get_Hj(g_matr, basis_order, vectors_order, Jc, m):
    Hj = []
    J = list(Jc)
    if J == list(range(m)):
        str_var = vectors_order.index([])
    else:
        str_var = vectors_order.index(J)
    for i in range(len(g_matr[str_var])):
        if g_matr[str_var][i] == 1:
            Hj.append(basis_order[i])
    return Hj

# Функция для вычисления вектора V
def get_V(Jc, basis_order, hj):
    v = []
    for pos in basis_order:
        flag = True
        for j in Jc:
            if pos[j] != hj[j]:
                v.append(0)
                flag = False
                break
        if flag:
            v.append(1)
    return v

# Функция для мажоритарного декодирования
def get_Mj(W, m, basis_order, r, g_matr, vectors_order):
    M = {}
    for I in range(r, -1, -1):
        if I == r:
            w = W
        else:
            for key in sorted(M):
                if len(key) == I + 1 and M[key] == 1:
                    _w = w
                    w = []
                    v = g_matr[vectors_order.index(list(key))]
                    for e in range(len(_w)):
                        w.append((_w[e] + v[e]) % 2)
                    break
        J = sorted(list(combinations(range(m), I)))
        for j in J:
            Jc = get_complement(m, j)
            Hj = get_Hj(g_matr, basis_order, vectors_order, j, m)
            count1 = 0
            count0 = 0
            for hj in Hj:
                V = get_V(Jc, basis_order, hj)

                temp = []
                s = 0
                for k in range(len(V)):
                    temp.append((V[k] or w[k]))
                    s += temp[-1] if temp[-1] == 1 else 0
                if Jc == list(range(m)):
                    M[j] = 0
                    break
                if ((s + 1) % 2) == 1:
                    count1 += 1
                else:
                    count0 += 1

                if count1 > 2 ** (m - I - 1):
                    M[j] = 1
                    break
                elif count0 > 2 ** (m - I - 1):
                    M[j] = 0
                    break
    return M

# Функция для добавления ошибок и проверки декодирования
def get_err_word(g_matr, r, basis_order, vectors_order, t):
    m = int(math.log2(g_matr.shape[1]))
    row = g_matr.shape[0]

    idx = rnd.randint(0, row - 1)
    word = np.array(g_matr[idx][:row])
    w = np.dot(word, g_matr) % 2
    print(f"Исходное сообщение: {word}")
    print(f"Отправленное сообщение: {w}")
    for i in range(t):
        w[i] += 1
        w[i] %= 2
    print(f"Принятое сообщение с ошибкой: {w}")
    M = get_Mj(w, m, basis_order, r, g_matr, vectors_order)
    u = []
    for i, j in M.items():
        u.append(j)
    u = u[::-1]
    print(f"Изменённое сообщение после преобразования: {u}")
    try:
        print(f"Декодированное сообщение: {np.dot(u, g_matr) % 2}")
    except:
        print("Произошла ошибка, необходимо повторно отправить сообщение")

# Тестирование кода Рида-Маллера
r, m = 2, 4
rm_g_matr, basis_order, vectors_order = get_rm_G_matr(r, m)
t_list = [1, 2]

for t in t_list:
    print(f"[1mЭкспериментальная проверка декодирования кода Рида-Маллера RM({r}, {m}) при t = {t}[0m")
    get_err_word(rm_g_matr, r, basis_order, vectors_order, t)
    if t != t_list[-1]:
        print("")
