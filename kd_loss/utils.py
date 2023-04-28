def pair_minus(x):
    """
    :param x: [*, N]
    :return: [*, N, N]
    """
    # A[i, j] = x[i] - x[j]
    return x.unsqueeze(-1) - x.unsqueeze(-2)