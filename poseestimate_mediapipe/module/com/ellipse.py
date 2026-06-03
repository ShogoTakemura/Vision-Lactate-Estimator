import numpy as np

# 短軸 : 2b, 長軸 : 2a


def inner_func(theta: float, lng: float, short: float) -> float:
    epsiron_sq = 1 - (short*short) / (lng*lng)
    cosval = np.cos(theta)
    return np.sqrt(1 - epsiron_sq * cosval * cosval)


def target_func():
    pass


def integral(start: float, end: float, divine: int) -> float:
    span = (end - start) / divine
    delta = np.arange(start, end, span)
    vals = [inner_func(d_theta, lng=60, short=35) *
            d_theta for d_theta in delta]
    return np.sum(vals)


def main():
    sum_val = integral(0.0, np.pi / 2.0, 10000)
    length = sum_val * 4.0 * 60.0
    print('ellipse length', length)


if __name__ == '__main__':
    main()
