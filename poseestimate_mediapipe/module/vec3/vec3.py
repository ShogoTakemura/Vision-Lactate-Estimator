

from math import sqrt


vec3 = tuple[float, float, float]


def direc_vec3(root: vec3, edge: vec3) -> vec3:
    return tuple([ed - rt for ed, rt in zip(edge, root)])


def multi_vec3(vector: vec3, coeff: float) -> vec3:
    return tuple(map(lambda component: component * coeff, vector))


def magni_vec3(vector: vec3) -> float:
    return sqrt(sum([item*item for item in vector]))


def unit_vec3(vector: vec3) -> vec3:
    magnitude = magni_vec3(vector)
    return multi_vec3(vector, 1.0/magnitude)


def add_vec3(first: vec3, second: vec3) -> vec3:
    return tuple([vc1 + vc2 for vc1, vc2 in zip(first, second)])


def ave_vec3(first: vec3, second: vec3) -> vec3:
    return tuple([(vc_f + vc_s) / 2.0 for vc_f, vc_s in zip(first, second)])
