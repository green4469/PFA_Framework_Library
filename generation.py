import random
def sum_to_one(n):
    values = [0.0, 1.0] + [random.random() for _ in range(n - 1)]
    values.sort()
    return [values[i+1] - values[i] for i in range(n)]
print(sum_to_one(3))
print(sum(sum_to_one(3)))