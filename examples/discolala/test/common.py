from disco.core import Disco
from numpy import random, float64
from scipy.sparse import coo_matrix
from matrixWrap import MatrixWrapper

disco = Disco("http://localhost:8989")
dfs = MatrixWrapper.DFS
raw = MatrixWrapper.RAW

def randomSparseMatrix(height, width, sparsity=0.7):
	"""
	Create a height-by-width coo matrix in double precision by default.
	The values in the matrix are randomly chosen from the range of (-1.0, 1.0).
	@param height Height of the matrix.
	@param width Width of the matrix.
	@param sparsity Density of the matrix in the range of [0-1] with 0=dense matrix and 1=matrix with all zeros.
	"""
	assert sparsity<=1 and sparsity>=0, "Invalid value for sparsity"
	rows = []
	cols = []
	vals = []
	for i in range(0, height):
		for j in range(0, width):
			if random.rand() >= sparsity:
				rows.append(i)
				cols.append(j)
				if random.rand() > 0.5:
					vals.append(random.rand())
				else:
					vals.append(-random.rand())
	return coo_matrix((vals, (rows,cols)), dtype=float64, dims=(height,width))

