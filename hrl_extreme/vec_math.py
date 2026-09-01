import math
import random
from typing import List, Tuple, Union

def dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

def norm(a: List[float]) -> float:
    return math.sqrt(max(0.0, sum(x * x for x in a)))

def normalize(a: List[float]) -> List[float]:
    n = norm(a)
    if n < 1e-9:
        return [0.0] * len(a)
    return [x / n for x in a]

def l2_dist(a: List[float], b: List[float]) -> float:
    return math.sqrt(max(0.0, sum((x - y) ** 2 for x, y in zip(a, b))))

def sub(a: List[float], b: List[float]) -> List[float]:
    return [x - y for x, y in zip(a, b)]

def add(a: List[float], b: List[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]

def scale(a: List[float], s: float) -> List[float]:
    return [x * s for x in a]

def clip(x: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, x))

def clip_vec(a: List[float], min_val: float, max_val: float) -> List[float]:
    return [clip(x, min_val, max_val) for x in a]

def cosine_similarity(a: List[float], b: List[float]) -> float:
    na = norm(a)
    nb = norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return dot(a, b) / (na * nb)

def relu(x: float) -> float:
    return max(0.0, x)

def relu_grad(x: float) -> float:
    return 1.0 if x > 0 else 0.0

def tanh(x: float) -> float:
    return math.tanh(x)

def tanh_grad(y: float) -> float:
    # y = tanh(x)
    return 1.0 - y * y

def softmax(logits: List[float]) -> List[float]:
    max_l = max(logits)
    exp_l = [math.exp(x - max_l) for x in logits]
    sum_e = sum(exp_l)
    return [x / max(sum_e, 1e-12) for x in exp_l]

def matmul_vec(W: List[List[float]], x: List[float], bias: List[float]) -> List[float]:
    # W shape: [out_features, in_features]
    return [sum(w_row[j] * x[j] for j in range(len(x))) + bias[i] for i, w_row in enumerate(W)]

def gaussian_sample(mean: List[float], std: List[float]) -> List[float]:
    return [random.gauss(m, max(1e-5, s)) for m, s in zip(mean, std)]

def gaussian_log_prob(action: List[float], mean: List[float], std: List[float]) -> float:
    # Log probability density under diagonal Gaussian
    log_prob = 0.0
    for a, m, s in zip(action, mean, std):
        s_safe = max(1e-5, s)
        var = s_safe ** 2
        log_scale = math.log(s_safe)
        diff = a - m
        log_p = -0.5 * (diff ** 2) / var - log_scale - 0.5 * math.log(2.0 * math.pi)
        log_prob += log_p
    return log_prob

def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[List[float], List[float]]:
    """
    Generalized Advantage Estimation (GAE).
    Returns (advantages, returns).
    """
    n = len(rewards)
    advantages = [0.0] * n
    last_adv = 0.0
    for t in reversed(range(n)):
        next_non_terminal = 1.0 - float(dones[t])
        next_val = values[t + 1] if (t + 1 < len(values)) else 0.0
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        last_adv = delta + gamma * lam * next_non_terminal * last_adv
        advantages[t] = last_adv
    returns = [adv + v for adv, v in zip(advantages, values[:n])]
    return advantages, returns
