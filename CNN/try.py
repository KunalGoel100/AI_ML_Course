import numpy as np
from sympy.codegen.cnodes import sizeof

a = np.array([
    [1,1,1],
    [2,-2,2],
    [3,3,3]
])

b = np.array([
    [2,2],
    [2,2]
])

print(a[0:2,0:2].max())