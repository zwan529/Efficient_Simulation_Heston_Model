

import numpy as np
from scipy import stats, optimize
def TG_obj_func(r, phi):
    normal_pdf = stats.norm.pdf(r)
    normal_cdf = stats.norm.cdf(r)
    return r * normal_pdf + normal_cdf * ( 1 + r ** 2) - (1 + phi) * (normal_pdf + r * normal_cdf ) ** 2 

def TG_solver(phi):
    ## Solve TG_obj_func given a value of phi
    return optimize.newton(lambda x: TG_obj_func(x, phi), x0 = 4, maxiter = 100)

def TG_mean(r):
    numerator = r
    denominator = stats.norm.pdf(r) + r * stats.norm.cdf(r)
    return numerator / denominator

def TG_var(r, phi):
    ## r, phi: scalar or narray
    numerator = 1 / np.sqrt(phi)
    denomunator = stats.norm.pdf(r) + r * stats.norm.cdf(r)
    return numerator / denomunator


def TG_map(start,end,points):
    space = np.linspace(start,end,points)
    loop_res = [TG_solver(x) for x in space]
    ## create np array
    result = np.array(loop_res)
    return (result, space)


def testpt(phi):
    r = TG_solver(phi)
    print(r)
    print((TG_mean(r), TG_var(r,phi)))