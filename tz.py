from collections import Counter


def find_w_distances(seq):
    w_indices = [i for i, c in enumerate(seq) if c == 'W']
    distances = []
    for i in range(len(w_indices) - 1):
        if w_indices[i + 1] - w_indices[i] > 1:
            distances.append((w_indices[i + 1] - w_indices[i], w_indices[i]))
    return distances


def max_seq(seq):
    max_length = 0
    current_length = 0
    for char in seq:
        if char == 'W':
            current_length += 1
            max_length = max(max_length, current_length)
        else:
            current_length = 0
    return max_length


def min_replacements(a, b, seq, answer):
    seq = list(seq)
    # print(max_seq(seq))
    if a == b:
        return a - Counter(seq)['W']
    elif max_seq(seq) >= b:
        return answer
    else:
        # all_w = [i for i, j in enumerate(seq) if j == 'W']
        all_w = sorted(find_w_distances(seq), key=lambda x: x[0])
        seq[all_w[0][1] + 1] = 'W'

        answer += 1
        # print(seq, all_w)
        return min_replacements(a, b, seq, answer)


n = int(input('Количество наборов: '))
for i in range(n):
    a, b = input().split()
    line = input('line: ')
    if len(line) != int(a):
        print('Длина не совпадает')
        exit(1)
    print(min_replacements(int(a), int(b), line, 0))
