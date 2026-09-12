def embedding_distance(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right or len(left) != len(right):
        return 999.0
    total = 0.0
    for a, b in zip(left, right):
        delta = float(a) - float(b)
        total += delta * delta
    return total ** 0.5
