import random
import pandas as pd
import numpy as np
from sympy import isprime, primerange

def prime_factors(n):
    i = 2
    f = set()
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            f.add(i)
    if n > 1:
        f.add(n)
    return list(f)

def find_generator(p):
    from sympy import isprime
    if not isprime(p):
        raise ValueError("p must be prime.")
    phi = p - 1
    facs = prime_factors(phi)
    for g in range(2, p):
        flag = False
        for factor in facs:
            if pow(g, phi // factor, p) == 1:
                flag = True
                break
        if not flag:
            return g
    raise ValueError(f"No generator found for p={p}.")

def generate_dataset(p, num_samples):
    g = find_generator(p)
    s = random.randint(1, p - 1)
    half_range = (p - 1) // 2
    data, labels = [], []
    for _ in range(num_samples):
        y = random.randint(1, p - 1)
        x = pow(g, y, p)
        label = 1 if s <= y < s + half_range else -1
        data.append(x)
        labels.append(label)
    return pd.DataFrame({'x': data, 'y': labels}), g, s
