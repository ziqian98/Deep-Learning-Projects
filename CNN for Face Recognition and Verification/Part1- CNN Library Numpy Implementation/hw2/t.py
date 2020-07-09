import numpy as np

A = np.arange(30.).reshape(2,3,5)
print(A)

print("***")
B = np.arange(24.).reshape(3,4,2)
print(B)
print("***")
C = np.tensordot(A,B, axes = ([0,1],[2,0]))
print(C)