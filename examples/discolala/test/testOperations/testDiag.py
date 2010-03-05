import unittest
import os
from operations import *
from test.common import *
from disco.util import jobname

class TestDiag(unittest.TestCase):
	def validate(self, k, A, Zwrap, tol=0.00000000001):
		from numpy import diag, zeros
		A = A.todense()
		if A.shape[0] == 1:
			Z = Zwrap.unwrapMatrix(max(A.shape)+abs(k), max(A.shape)+abs(k)).todense()
			# diag of a vector is a matrix
			Zexp = diag([A[0,j] for j in range(0, A.shape[1])], k)
			D = Zexp - Z 
			for i in range(0, D.shape[0]):
				for j in range(0, D.shape[1]):
					assert abs(D[i,j]) <= tol, "expecting %.14f but received %.14f for element (%d,%d)" % (Zexp[i,j], Z[i,j], i, j)
		elif A.shape[1] == 1:
			Z = Zwrap.unwrapMatrix(max(A.shape)+abs(k), max(A.shape)+abs(k)).todense()
			# diag of a vector is a matrix
			Zexp = diag([A[i,0] for i in range(0, A.shape[0])], k)
			D = Zexp - Z
			for i in range(0, D.shape[0]):
				for j in range(0, D.shape[1]):
					assert abs(D[i,j]) <= tol, "expecting %.14f but received %.14f for element (%d,%d)" % (Zexp[i,j], Z[i,j], i, j)
		else:
			# diag of a matrix is a vector
			v = diag(A, k)
			# convert vector to a matrix
			Zexp = zeros((len(v), 1))
			Zexp[:,0] = v
			Z = Zwrap.unwrapMatrix(len(v),1).todense()
			D = Zexp - Z 
			for i in range(0, D.shape[0]):
				for j in range(0, D.shape[1]):
					assert abs(D[i,j]) <= tol, "expecting %.14f but received %.14f for element (%d,%d)" % (Zexp[i,j], Z[i,j], i, j)

	def test1(self):
		"""
		Test input to diag is a tall matrix with positive k.
		"""
		random.seed(13)
		m, n = 101, 51
		k = 1
		sparsityA = 0.3
		protocolA = MatrixWrapper.RAW
		dfsDirA = None
		maxCoresA = 13
		maxTotalBlocks = 3
		# instantiate A
		A = randomSparseMatrix(m, n, sparsityA)
		Awrap = MatrixWrapper.wrapMatrix(A, protocolA, dfsDirA, maxCoresA)
		# compute diagonal
		Zwrap = diag(disco, m, n, Awrap, k, maxTotalBlocks)
		# validate
		self.validate(k, A, Zwrap)
		# clean up
		disco.purge(jobname(Zwrap.urls[0]))

	def test2(self):
		"""
		Test input to diag is a wide matrix with negative k.
		"""
		random.seed(13)
		m, n = 33, 77
		k = -3 
		sparsityA = 0.3
		protocolA = MatrixWrapper.RAW
		dfsDirA = None
		maxCoresA = 13
		maxTotalBlocks = 4
		# instantiate A
		A = randomSparseMatrix(m, n, sparsityA)
		Awrap = MatrixWrapper.wrapMatrix(A, protocolA, dfsDirA, maxCoresA)
		# compute diagonal
		Zwrap = diag(disco, m, n, Awrap, k, maxTotalBlocks)
		# validate
		self.validate(k, A, Zwrap)
		# clean up
		disco.purge(jobname(Zwrap.urls[0]))

	def test3(self):
		"""
		Test input to diag is a col vector.
		"""
		random.seed(13)
		m, n = 123, 1
		k = 1 
		sparsityA = 0.2
		protocolA = MatrixWrapper.RAW
		dfsDirA = None
		maxCoresA = 4 
		maxTotalBlocks = 5
		# instantiate A
		A = randomSparseMatrix(m, n, sparsityA)
		Awrap = MatrixWrapper.wrapMatrix(A, protocolA, dfsDirA, maxCoresA)
		# compute diagonal
		Zwrap = diag(disco, m, n, Awrap, k, maxTotalBlocks)
		# validate
		self.validate(k, A, Zwrap)
		# clean up
		disco.purge(jobname(Zwrap.urls[0]))

	def test4(self):
		"""
		Test input to diag is a row vector.
		"""
		random.seed(13)
		m, n = 1, 111
		k = -6 
		sparsityA = 0.2
		protocolA = MatrixWrapper.RAW
		dfsDirA = None
		maxCoresA = 7 
		maxTotalBlocks = 6
		# instantiate A
		A = randomSparseMatrix(m, n, sparsityA)
		Awrap = MatrixWrapper.wrapMatrix(A, protocolA, dfsDirA, maxCoresA)
		# compute diagonal
		Zwrap = diag(disco, m, n, Awrap, k, maxTotalBlocks)
		# validate
		self.validate(k, A, Zwrap)
		# clean up
		disco.purge(jobname(Zwrap.urls[0]))

if __name__ == "__main__":
	unittest.main()
