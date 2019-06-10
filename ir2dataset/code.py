import numpy as np
import csv
from numpy import linalg as LA
import time
import math
from collections import OrderedDict

no_of_movies=100
no_of_users=610
Matrix = [[0 for x in range(no_of_movies)]for y in range(no_of_users)]
f=open('movies.csv',encoding="utf8")

k = csv.reader(f)

movieid=[]
l=0
for i in k:
	l=l+1
	if l!=1 and l<102:
		movieid.append(int(i[0]))

f.close()

g = open('ratings.csv', encoding="utf8")
m=csv.reader(g)
l=0
for i in m:
	l=l+1
	if l!=1:
		for x in range(no_of_movies):
			if int(i[1])== movieid[x]:
				Matrix[int(i[0])-1][x]=float(i[2])
g.close()
print(Matrix[0][69])

Matrix_Transpose= [[0 for x in range(no_of_users)]for y in range(no_of_movies)]

for i in range(no_of_users):
	for j in range(no_of_movies):
		Matrix_Transpose[j][i]=Matrix[i][j]

def matrixmult (A, B):
    C = [[0 for row in range(len(A))] for col in range(len(B[0]))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i][j] += A[i][k]*B[k][j]
    return C

Matrix_Product=matrixmult(Matrix,Matrix_Transpose)

def eigen_decomposition(M):
    """
    Returns Eigen values and corresponding eigen vectors arranged in descending order.

    @params:
    M: Input numpy matrix

    Output:
    Returns list - sorted_eigen_values, sorted_eigen_vectors
    sorted_eigen_values - list of sorted eigen_values
    sorted_eigen_vectors - numpy matrix containing eigen vectors
    """
    eigen_values, eigen_vectors = LA.eig(M)
    eigen_values = eigen_values.real  # Considering real parts only
    eigen_vectors = eigen_vectors.real  # Considering real parts only
    for i in range(len(list(eigen_values))):
        eigen_values[i] = round(eigen_values[i], 2)  # Rounding values of 2 digits
    for i in range(eigen_vectors.shape[0]):
        for j in range(eigen_vectors.shape[1]):
            eigen_vectors[i][j] = round(eigen_vectors[i][j], 2)  # Rounding values of 2 digits

    eigen = dict()
    for i in range(len(eigen_values)):
        if eigen_values[i] != 0:
            eigen[eigen_values[i]] = eigen_vectors[:, i]  # Removing zeros

    sorted_eigen_values = sorted(list(eigen.keys()), reverse=True)
    sorted_eigen_vectors = np.zeros_like(eigen_vectors)
    for i in range(len(sorted_eigen_values)):
        sorted_eigen_vectors[:, i] = eigen[sorted_eigen_values[i]]

    sorted_eigen_vectors = sorted_eigen_vectors[:, :len(sorted_eigen_values)]  # Removing zeroed eigen vectors

    return sorted_eigen_values, sorted_eigen_vectors

def svd(M, dimension_reduction=1.0):
    """
    Applies Singular Value Decomposition to input matrix M - minimum reconstruction
    error of M expressed as U, sigma and V such that M = U * sigma * V

    Supports dimensionality reduction where least values of sigma are removed along with
    their corresponding U columns and V rows.

    @params:
    M : Input numpy matrix M
    dimension_reduction: Reduce the dimensions. Recommended range: 0.8 - 1.0

    Output:
    Returns list - U, sigma, V
    sigma - singular values of M
    """
    try:
        assert dimension_reduction <= 1.0 or dimension_reduction == None
    except AssertionError as ae:
        return "Wrong dimension_reduction value"

    try:
        assert type(M) == np.ndarray
    except AssertionError as ae:
        return "Wrong Matrix type. (numpy.ndarray) required."

    eigen_values_u, U = eigen_decomposition(np.dot(M, M.T))
    eigen_values_v, V = eigen_decomposition(np.dot(M.T, M))

    V = V.T
    print(eigen_values_u)
    print(eigen_values_v)

    sigma = np.diag([i**0.5 for i in eigen_values_u])
    if dimension_reduction == 1.0 or dimension_reduction == None:
        return U, sigma, V
    else:
        total_sigma = np.sum(sigma ** 0.5)
        for i in range(sigma.shape[0]):
            sigma_sum = np.sum(sigma[:i+1, :i+1])
            if sigma_sum > dimension_reduction * total_sigma:
                sigma = sigma[:i, :i]
                U = U[:, :i]
                V = V[:i, :]
                return U, sigma, V
