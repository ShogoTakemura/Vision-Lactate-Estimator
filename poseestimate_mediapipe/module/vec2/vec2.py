#2次元ベクトル操作

from math import sqrt


vec2 = tuple[float, float]


def distance(first: vec2, second: vec2) -> float:#2つのベクトル間のユークリッド距離を計算
    sub = (vc_f - vc_s for vc_f, vc_s in zip(first, second))
    square = (val * val for val in sub)
    return sqrt(sum(square))


def direc(root: vec2, edge: vec2) -> vec2:#方向ベクトルの計算
    return tuple([ed - rt for ed, rt in zip(edge, root)])


def multiple(vector: vec2, coeff: float) -> vec2:
    return tuple(map(lambda component: component * coeff, vector))


def magni(vector: vec2) -> float:
    return sqrt(sum([item*item for item in vector]))


def unit(vector: vec2) -> vec2:
    magnitude = magni(vector)
    return multiple(vector, 1.0/magnitude)


def add(first: vec2, second: vec2) -> vec2:
    return tuple([vc1 + vc2 for vc1, vc2 in zip(first, second)])


def average(first: vec2, second: vec2) -> vec2:
    return tuple([(vc_f + vc_s) / 2.0 for vc_f, vc_s in zip(first, second)])


def inner_divine(vec_one: vec2, vec_two: vec2, ratio1: float, ratio2: float) -> vec2:
    length = ratio1 + ratio2
    vec_pc_y = (vec_one[0] * ratio2 + vec_two[0] * ratio1) / length
    vec_pc_z = (vec_one[1] * ratio2 + vec_two[1] * ratio1) / length
    return (vec_pc_y, vec_pc_z)


def normal(vector: vec2) -> vec2:
    return (-vector[1], vector[0])
