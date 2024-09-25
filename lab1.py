import numpy as np
from itertools import combinations


# Задание 1.1: Функция приведения матрицы к ступенчатому виду (REF)
def ref(matrix):
    mat = np.array(matrix)
    n_rows, n_cols = mat.shape
    lead = 0
    for r in range(n_rows):
        if lead >= n_cols:
            return mat
        i = r
        while mat[i, lead] == 0:
            i += 1
            if i == n_rows:
                i = r
                lead += 1
                if lead == n_cols:
                    return mat
        # Меняем строки местами
        mat[[i, r]] = mat[[r, i]]

        # Обрабатываем строки ниже текущей
        for i in range(r + 1, n_rows):
            if mat[i, lead] != 0:
                mat[i] = (mat[i] + mat[r]) % 2
        lead += 1
    return mat


# Задание 1.2: Функция приведения матрицы к приведённому ступенчатому виду (RREF)
def rref(mat):
    mat = ref(mat)
    n_rows, n_cols = mat.shape

    for r in range(n_rows - 1, -1, -1):
        lead = np.argmax(mat[r] != 0)
        if mat[r, lead] != 0:
            for i in range(r - 1, -1, -1):
                if mat[i, lead] != 0:
                    mat[i] = (mat[i] + mat[r]) % 2
    while not any(mat[n_rows - 1]):
        mat = mat[:-1, :]
        n_rows -= 1
    return mat


# Задание 1.3: Ведущие столбцы и создание сокращённой матрицы
def find_lead_columns(matrix):
    lead_columns = []
    for r in range(len(matrix)):
        row = matrix[r]
        for i, val in enumerate(row):
            if val == 1:
                lead_columns.append(i)
                break
    return lead_columns


# Удаление ведущих столбцов
def remove_lead_columns(matrix, lead_columns):
    mat = np.array(matrix)
    reduced_matrix = np.delete(mat, lead_columns, axis=1)
    return reduced_matrix


# Задание 1.3.4: Формирование проверочной матрицы H
def form_H_matrix(X, lead_columns, n_cols):
    n_rows = np.shape(X)[1]
    H = np.zeros((n_cols, n_rows), dtype=int)
    I = np.eye(6, dtype=int)

    H[lead_columns, :] = X
    not_lead = [i for i in range(n_cols) if i not in lead_columns]
    H[not_lead, :] = I
    return H


# Основная функция выполнения всех шагов лабораторной работы
def LinearCode(mat):
    # Задание 1.3.1: Приведение матрицы к ступенчатому виду
    G_star = rref(mat)
    print("G* (RREF матрица) =")
    print(G_star)

    # Задание 1.3.2: Нахождение ведущих столбцов
    lead_columns = find_lead_columns(G_star)
    print(f"lead = {lead_columns}")

    # Задание 1.3.3: Удаление ведущих столбцов
    X = remove_lead_columns(G_star, lead_columns)
    print("Сокращённая матрица X =")
    print(X)

    # Задание 1.3.4: Формирование проверочной матрицы H
    n_cols = np.shape(mat)[1]
    H = form_H_matrix(X, lead_columns, n_cols)
    print("Проверочная матрица H =")
    print(H)

    return H


# Генерация всех кодовых слов через сложение строк
def generate_codewords_from_combinations(G):
    rows = G.shape[0]
    codewords = set()

    for r in range(1, rows + 1):
        for comb in combinations(range(rows), r):
            codeword = np.bitwise_xor.reduce(G[list(comb)], axis=0)
            codewords.add(tuple(codeword))

    codewords.add(tuple(np.zeros(G.shape[1], dtype=int)))
    return np.array(list(codewords))


# Генерация кодовых слов умножением двоичных слов на G
def generate_codewords_binary_multiplication(G):
    k = G.shape[0]
    n = G.shape[1]
    codewords = []

    for i in range(2 ** k):
        binary_word = np.array(list(np.binary_repr(i, k)), dtype=int)
        codeword = np.dot(binary_word, G) % 2
        codewords.append(codeword)

    return np.array(codewords)


# Проверка кодового слова с помощью матрицы H
def check_codeword(codeword, H):
    return np.dot(codeword, H) % 2


# Вычисление кодового расстояния
def calculate_code_distance(codewords):
    min_distance = float('inf')

    for i in range(len(codewords)):
        for j in range(i + 1, len(codewords)):
            distance = np.sum(np.bitwise_xor(codewords[i], codewords[j]))
            if distance > 0:
                min_distance = min(min_distance, distance)
    return min_distance


# Выполнение всех шагов с ошибками
def LinearCodeWithErrors(mat):
    G_star = rref(mat)
    lead_columns = find_lead_columns(G_star)
    X = remove_lead_columns(G_star, lead_columns)
    n_cols = np.shape(mat)[1]
    H = form_H_matrix(X, lead_columns, n_cols)

    print("G* (RREF матрица) =")
    print(G_star)
    print(f"lead = {lead_columns}")
    print("Сокращённая матрица X =")
    print(X)
    print("Проверочная матрица H =")
    print(H)

    # Генерация кодовых слов через сложение строк
    codewords_1 = generate_codewords_from_combinations(G_star)
    print("Все кодовые слова (способ 1):")
    print(codewords_1)

    # Генерация кодовых слов умножением двоичных слов на G
    codewords_2 = generate_codewords_binary_multiplication(G_star)
    print("Все кодовые слова (способ 2):")
    print(codewords_2)

    assert set(map(tuple, codewords_1)) == set(map(tuple, codewords_2)), "Наборы кодовых слов не совпадают!"

    for codeword in codewords_1:
        result = check_codeword(codeword, H)
        assert np.all(result == 0), f"Ошибка: кодовое слово {codeword} не прошло проверку матрицей H"

    print("Все кодовые слова прошли проверку матрицей H.")

    # Вычисление кодового расстояния
    d = calculate_code_distance(codewords_1)
    t = (d - 1) // 2 if d > 1 else 1
    print(f"Кодовое расстояние d = {d}")
    print(f"Кратность обнаруживаемой ошибки t = {t}")

    # Проверка ошибки кратности t
    e1 = np.zeros(n_cols, dtype=int)
    e1[2] = 1
    v = codewords_1[4]
    v_e1 = (v + e1) % 2
    print(f"v + e1 = {v_e1}")
    print(f"(v + e1)@H = {check_codeword(v_e1, H)} - ошибка")

    # Проверка ошибки кратности t+1
    e2 = np.zeros(n_cols, dtype=int)
    e2[6] = 1
    e2[9] = 1
    v_e2 = (v + e2) % 2
    print(f"v + e2 = {v_e2}")
    print(f"(v + e2)@H = {check_codeword(v_e2, H)} - без ошибки")

    return H


# Пример использования
matrix = [[1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
          [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
          [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
          [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1],
          [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
          [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]]

result = LinearCodeWithErrors(matrix)