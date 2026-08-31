import math
import random
from typing import List, Union

def dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

def norm(a: List[float]) -> float:
    return math.sqrt(sum(x * x for x in a))

def normalize(a: List[float]) -> List[float]:
    n = norm(a)
    if n < 1e-9:
        return [0.0] * len(a)
    return [x / n for x in a]

def l2_dist(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def sub(a: List[float], b: List[float]) -> List[float]:
    return [x - y for x, y in zip(a, b)]

def add(a: List[float], b: List[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]

def scale(a: List[float], s: float) -> List[float]:
    return [x * s for x in a]

def cosine_similarity(a: List[float], b: List[float]) -> float:
    na = norm(a)
    nb = norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return dot(a, b) / (na * nb)

def relu(x: float) -> float:
    return max(0.0, x)

def tanh(x: float) -> float:
    return math.tanh(x)

def softmax(logits: List[float]) -> List[float]:
    max_l = max(logits)
    exp_l = [math.exp(x - max_l) for x in logits]
    sum_e = sum(exp_l)
    return [x / sum_e for x in exp_l]

def matmul_vec(W: List[List[float]], x: List[float], bias: List[float]) -> List[float]:
    # W shape: [out_features, in_features]
    return [sum(w_row[j] * x[j] for j in range(len(x))) + bias[i] for i, w_row in enumerate(W)]
