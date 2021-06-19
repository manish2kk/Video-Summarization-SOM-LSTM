print('\n----SVD----\n')
from numpy import array
from numpy import diag
from numpy import dot
from numpy import zeros
from scipy.linalg import svd
# define a matrix
A = array([[1, 2], [3, 4], [5, 6]])
print('\nA =\n')
print(A)
# Singular-value decomposition
U, s, VT = svd(A)
print('\nU =\n')
print(U)
print('\ns =\n')
print(s)
print('\nVT =\n')
print(VT)

# create m x n Sigma matrix
Sigma = zeros((A.shape[0], A.shape[1]))
# populate Sigma with n x n diagonal matrix
Sigma[:A.shape[1], :A.shape[1]] = diag(s)
# reconstruct matrix
B = U.dot(Sigma.dot(VT))
print('\nU.s.VT =\n')
print(B)

print('\n----Pseudoinverse----\n')
from numpy.linalg import pinv
# define matrix
A = array([
	[0.1, 0.2],
	[0.3, 0.4],
	[0.5, 0.6],
	[0.7, 0.8]])
print('\nA =\n')
print(A)
# calculate pseudoinverse
B = pinv(A)
print('\nA+ =\n')
print(B)
U, s, VT = svd(A)
# reciprocals of s
d = 1.0 / s
# create m x n D matrix
D = zeros(A.shape)
# populate D with n x n diagonal matrix
D[:A.shape[1], :A.shape[1]] = diag(d)
# calculate pseudoinverse
B = VT.T.dot(D.T).dot(U.T)
print('\nA+ by SVD=\n')
print(B)
print('\nAA+ =\n')
print(A.dot(B))
print(B.dot(A))

print('\n-----dim red-----\n')
A = array([
	[1,2,3,4,5,6,7,8,9,10],
	[11,12,13,14,15,16,17,18,19,20],
	[21,22,23,24,25,26,27,28,29,30]])
print('\nA =\n')
print(A)
# Singular-value decomposition
U, s, VT = svd(A)
print('\nU =\n')
print(U)
print('\ns =\n')
print(s)
print('\nVT =\n')
print(VT)
# create m x n Sigma matrix
Sigma = zeros((A.shape[0], A.shape[1]))
# populate Sigma with n x n diagonal matrix
Sigma[:A.shape[0], :A.shape[0]] = diag(s)
# select
n_elements = 2
Sigma = Sigma[:, :n_elements]
VT = VT[:n_elements, :]
# reconstruct
B = U.dot(Sigma.dot(VT))
print('\nA =\n')
print(B)
# transform
T = U.dot(Sigma)
print('\nT =\n')
print(T)
T = A.dot(VT.T)
print('\nT =\n')
print(T)
