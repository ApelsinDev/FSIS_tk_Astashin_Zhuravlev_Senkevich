import random


def create_identity_matrix(size):
    return [[1 if i == j else 0 for j in range(size)] for i in range(size)]


def concatenate_horizontal(matrix1, matrix2):
    return [row1 + row2 for row1, row2 in zip(matrix1, matrix2)]


def concatenate_vertical(matrix1, matrix2):
    return matrix1 + matrix2


def generate_x_vectors(n, k):
    vectors = []
    vector = [0] * (n - k)
    while len(vectors) < k:
        for i in reversed(range(len(vector))):
            if vector[i] == 0:
                vector[i] = 1
                vectors.append(vector[:])
                break
            else:
                vector[i] = 0
    return vectors


def generate_hamming_matrix(r):
    n = 2 ** r - 1
    k = 2 ** r - r - 1
    return concatenate_horizontal(create_identity_matrix(k), generate_x_vectors(n, k))


def generate_parity_matrix(r):
    n = 2 ** r - 1
    k = 2 ** r - r - 1
    return concatenate_vertical(generate_x_vectors(n, k), create_identity_matrix(n - k))


def generate_syndrome_table(h_matrix):
    syndrome_table = {}
    for i in range(len(h_matrix[0])):
        error_vector = [0] * len(h_matrix[0])
        error_vector[i] = 1
        syndrome = vector_multiply_matrix(error_vector, h_matrix)
        syndrome_table[tuple(syndrome)] = error_vector
    return syndrome_table


def vector_multiply_matrix(vector, matrix):
    return [sum(v * m for v, m in zip(vector, col)) % 2 for col in zip(*matrix)]


def generate_extended_hamming_matrix(r):
    g_matrix = generate_hamming_matrix(r)
    for row in g_matrix:
        parity = sum(row) % 2
        row.append(parity)
    return g_matrix


def generate_extended_parity_matrix(r):
    h_matrix = generate_parity_matrix(r)
    additional_row = [0] * len(h_matrix[0])
    h_matrix.append(additional_row)
    for row in h_matrix:
        row.append(1)
    return h_matrix


def generate_error_vector(length, errors):
    vector = [0] * length
    positions = random.sample(range(length), errors)
    for pos in positions:
        vector[pos] = 1
    return vector


def correct_error(h_matrix, received_word):
    syndrome = vector_multiply_matrix(received_word, h_matrix)
    syndrome_table = generate_syndrome_table(h_matrix)
    if tuple(syndrome) in syndrome_table:
        error_vector = syndrome_table[tuple(syndrome)]
        corrected_word = [(bit + err) % 2 for bit, err in zip(received_word, error_vector)]
        return corrected_word, syndrome
    return received_word, syndrome


def investigate_hamming_code(r, extended=False):
    if extended:
        g_matrix = generate_extended_hamming_matrix(r)
        h_matrix = generate_extended_parity_matrix(r)
        max_errors = 4
        print("\nИсследование расширенного кода Хэмминга")
    else:
        g_matrix = generate_hamming_matrix(r)
        h_matrix = generate_parity_matrix(r)
        max_errors = 3
        print("\nИсследование стандартного кода Хэмминга")

    print("\nПорождающая матрица G:")
    for row in g_matrix:
        print(row)

    print("\nПроверочная матрица H:")
    for row in h_matrix:
        print(row)

    u_vectors = [list(col) for col in zip(*g_matrix)]
    codeword = u_vectors[random.randint(0, len(u_vectors) - 1)]
    print("\nСгенерированное кодовое слово:")
    print(codeword)

    # Проверка на ошибки (1 до max_errors)
    for errors in range(1, max_errors + 1):
        if errors > len(codeword):
            break
        print(f"\nПроверка для {errors} ошибок:")

        error_vector = generate_error_vector(len(codeword), errors)
        print(f"Вектор ошибок: {error_vector}")

        received_word = [(bit + err) % 2 for bit, err in zip(codeword, error_vector)]
        print(f"Кодовое слово с ошибками: {received_word}")

        corrected_word, syndrome = correct_error(h_matrix, received_word)
        print(f"Синдром: {syndrome}")
        print(f"Исправленное кодовое слово: {corrected_word}")

        final_syndrome = vector_multiply_matrix(corrected_word, h_matrix)
        print(f"Синдром после коррекции (должен быть [0,...,0]): {final_syndrome}")


# Примеры вызова функций
# Исследование стандартного кода Хэмминга для одно-, двух- и трехкратных ошибок
investigate_hamming_code(2)
investigate_hamming_code(3)
investigate_hamming_code(4)

# Исследование расширенного кода Хэмминга для одно-, двух-, трех- и четырехкратных ошибок
investigate_hamming_code(2, extended=True)
investigate_hamming_code(3, extended=True)
investigate_hamming_code(4, extended=True)

