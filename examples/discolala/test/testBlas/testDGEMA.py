import unittest
from blas import dgema
from test.common import *
from disco.util import jobname

class TestDGEMA(unittest.TestCase):
	def validate(self, alpha, beta, transA, transB, A, B, Z, tol=0.00000000001):
		if transA:
			A = A.transpose()
		if transB:
			B = B.transpose()
		Zexp = (alpha * A + beta * B).todense()
		D = Zexp - Z 
		for i in range(0, D.shape[0]):
			for j in range(0, D.shape[1]):
				assert abs(D[i,j]) <= tol, "expecting %.14f but received %.14f for element (%d,%d)" % (Zexp[i,j], Z[i,j], i,j)

	def test0(self):
		'''
		Test normal basic usage.
		'''
		random.seed(13)
		m, n = 30, 30
		transA, transB = False, False
		alpha, beta = 1.0, 1.0
		sparsityA, sparsityB = 0.3, 0.0
		protocolA, protocolB = MatrixWrapper.RAW, MatrixWrapper.RAW
		dfsDirA, dfsDirB = None, None
		maxCoresA, maxCoresB = 13, 11
		maxTotalBlocks = 10
		# instantiate A
		A = randomSparseMatrix(m, n, sparsityA)
		Awrap = MatrixWrapper.wrapMatrix(A, protocolA, dfsDirA, maxCoresA)
		# instantiate B
		B = randomSparseMatrix(m, n, sparsityB)
		Bwrap = MatrixWrapper.wrapMatrix(B, protocolB, dfsDirB, maxCoresB)
		# add 
		Zwrap = dgema(disco, transA, transB, m, n, alpha, Awrap, Bwrap, beta, maxTotalBlocks)
		Z = Zwrap.unwrapMatrix(m, n).todense()
		# clean up
		disco.purge(jobname(Zwrap.urls[0]))
		# validate
		self.validate(alpha, beta, transA, transB, A, B, Z)

	def test1(self):
		'''
		Test the case where size of matrix is smaller than the number of inputs for A and B.
		'''
		random.seed(13)
		m, n = 3, 3
		transA, transB = False, False
		alpha, beta = -1.2, 0.8
		sparsityA, sparsityB = 0.0, 0.0
		protocolA, protocolB = MatrixWrapper.RAW, MatrixWrapper.RAW
		dfsDirA, dfsDirB = None, None
		maxCoresA, maxCoresB = 13, 11
		maxTotalBlocks = 10
		# instantiate A
		A = randomSparseMatrix(m, n, sparsityA)
		Awrap = MatrixWrapper.wrapMatrix(A, protocolA, dfsDirA, maxCoresA)
		# instantiate B
		B = randomSparseMatrix(m, n, sparsityB)
		Bwrap = MatrixWrapper.wrapMatrix(B, protocolB, dfsDirB, maxCoresB)
		# add
		Zwrap = dgema(disco, transA, transB, m, n, alpha, Awrap, Bwrap, beta, maxTotalBlocks)
		Z = Zwrap.unwrapMatrix(m, n).todense()
		# clean up
		disco.purge(jobname(Zwrap.urls[0]))
		# validate
		self.validate(alpha, beta, transA, transB, A, B, Z)

	def test2(self):
		'''
		Test transpose. 
		'''
		random.seed(13)
		m, n = 33, 3
		transA, transB = True, True
		alpha, beta = 0.2, 0.4
		sparsityA, sparsityB = 0.1, 0.1
		protocolA, protocolB = MatrixWrapper.RAW, MatrixWrapper.RAW
		dfsDirA, dfsDirB = None, None
		maxCoresA, maxCoresB = 5, 5 
		maxTotalBlocks = 3
		# instantiate A
		A = randomSparseMatrix(n, m, sparsityA)
		Awrap = MatrixWrapper.wrapMatrix(A, protocolA, dfsDirA, maxCoresA)
		# instantiate B
		B = randomSparseMatrix(n, m, sparsityB)
		Bwrap = MatrixWrapper.wrapMatrix(B, protocolB, dfsDirB, maxCoresB)
		# add
		Zwrap = dgema(disco, transA, transB, m, n, alpha, Awrap, Bwrap, beta, maxTotalBlocks)
		Z = Zwrap.unwrapMatrix(m, n).todense()
		# clean up
		disco.purge(jobname(Zwrap.urls[0]))
		# validate
		self.validate(alpha, beta, transA, transB, A, B, Z)

	def test3(self):
		'''
		Test the case where alpha=0.
		'''
		random.seed(13)
		m, n = 33, 77 
		transA, transB = False, False
		alpha, beta = 0.0, 0.5
		sparsityA, sparsityB = 0.0, 0.0
		protocolA, protocolB = MatrixWrapper.RAW, MatrixWrapper.RAW
		dfsDirA, dfsDirB = None, None
		maxCoresA, maxCoresB = 7, 5
		maxTotalBlocks = 7 
		# instantiate A
		A = randomSparseMatrix(m, n, sparsityA)
		Awrap = MatrixWrapper.wrapMatrix(A, protocolA, dfsDirA, maxCoresA)
		# instantiate B
		B = randomSparseMatrix(m, n, sparsityB)
		Bwrap = MatrixWrapper.wrapMatrix(B, protocolB, dfsDirB, maxCoresB)
		# add
		Zwrap = dgema(disco, transA, transB, m, n, alpha, Awrap, Bwrap, beta, maxTotalBlocks)
		Z = Zwrap.unwrapMatrix(m, n).todense()
		# clean up
		disco.purge(jobname(Zwrap.urls[0]))
		# validate
		self.validate(alpha, beta, transA, transB, A, B, Z)

	def test4(self):
		'''
		Test the case where alpha=0 and beta=0.
		'''
		random.seed(13)
		m, n = 33, 33
		transA, transB = False, False
		alpha, beta = 0, 0
		sparsityA, sparsityB = 0.0, 0.0
		protocolA, protocolB = MatrixWrapper.RAW, MatrixWrapper.RAW
		dfsDirA, dfsDirB = None, None
		maxCoresA, maxCoresB = 7, 5
		maxTotalBlocks = 7 
		# instantiate A
		A = randomSparseMatrix(m, n, sparsityA)
		Awrap = MatrixWrapper.wrapMatrix(A, protocolA, dfsDirA, maxCoresA)
		# instantiate B
		B = randomSparseMatrix(m, n, sparsityB)
		Bwrap = MatrixWrapper.wrapMatrix(B, protocolB, dfsDirB, maxCoresB)
		# add
		Zwrap = dgema(disco, transA, transB, m, n, alpha, Awrap, Bwrap, beta, maxTotalBlocks)
		Z = Zwrap.unwrapMatrix(m, n).todense()
		# clean up
		disco.purge(jobname(Zwrap.urls[0]))
		# validate
		self.validate(alpha, beta, transA, transB, A, B, Z)

if __name__ == "__main__":
	unittest.main()
