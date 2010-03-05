import unittest
import os
from operations import *
from test.common import *
from disco.util import jobname

class TestTril(unittest.TestCase):
	def validate(self, k, A, U, tol=0.00000000001):
		A = A.todense()
		for i in range(0, U.shape[0]):
			for j in range(0, U.shape[1]):
				if i >= j - k: 
					assert abs(A[i,j]-U[i,j]) <= tol, "expecting %.14f but received %.14f for element (%d,%d)" % (A[i,j], U[i,j], i, j)
				else:
					assert U[i,j] == 0, "expecting 0 but received %.14f for element (%d,%d)" % (U[i,j], i, j)

	def test1(self):
		"""
		Test normal usage.
		"""
		random.seed(13)
		m, n = 44, 33
		k = -2
		sparsityA = 0.1
		protocolA = MatrixWrapper.RAW
		dfsDirA = None
		maxCoresA = 6
		maxCores = 5 
		# instantiate A
		A = randomSparseMatrix(m, n, sparsityA)
		Awrap = MatrixWrapper.wrapMatrix(A, protocolA, dfsDirA, maxCoresA)
		# get lower triangular matrix
		Uwrap = tril(disco, m, n, Awrap, k, maxCores)
		U = Uwrap.unwrapMatrix(m, n).todense()
		# clean up
		Uwrap.purge(disco)
		# validate
		self.validate(k, A, U)

if __name__ == "__main__":
	unittest.main()
