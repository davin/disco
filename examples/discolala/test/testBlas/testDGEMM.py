import unittest
from disco.dfs.gluster import files
from blas import dgemm
from test.common import *
from disco.util import jobname

class TestDGEMM(unittest.TestCase):
	def validate(self, alpha, beta, transA, transB, A, B, C, Z, tol=0.00000000001):
		if transA:
			A = A.transpose()
		if transB:
			B = B.transpose()
		Zexp = (alpha*A*B + beta*C).todense()
		D = Zexp - Z 
		for i in range(0, D.shape[0]):
			for j in range(0, D.shape[1]):
				assert abs(D[i,j]) <= tol, "expecting %.14f but received %.14f for element (%d,%d)" % (Zexp[i,j], Z[i,j], i,j)

	def test0(self):
		'''
		Test normal basic usage.
		'''
		random.seed(13)
		m, k, n = 30, 30, 30
		transA, transB = False, False
		alpha, beta = 1.0, 1.0
		sparsityA, sparsityB, sparsityC = 0.0, 0.0, 0.0
		protocolA, protocolB, protocolC = MatrixWrapper.RAW, MatrixWrapper.RAW, MatrixWrapper.RAW
		dfsDirA, dfsDirB, dfsDirC = None, None, None
		maxCoresA, maxCoresB, maxCoresC = 13, 11, 7
		maxTotalBlocks = 10
		# instantiate A
		A = randomSparseMatrix(m, k, sparsityA)
		Awrap = MatrixWrapper.wrapMatrix(A, protocolA, dfsDirA, maxCoresA)
		# instantiate B
		B = randomSparseMatrix(k, n, sparsityB)
		Bwrap = MatrixWrapper.wrapMatrix(B, protocolB, dfsDirB, maxCoresB)
		# instantiate C
		C = randomSparseMatrix(m, n, sparsityC)
		Cwrap = MatrixWrapper.wrapMatrix(C, protocolC, dfsDirC, maxCoresC)
		# multiply
		Zwrap = dgemm(disco, transA, transB, m, n, k, alpha, Awrap, Bwrap, beta, Cwrap, maxTotalBlocks)
		Z = Zwrap.unwrapMatrix(m, n).todense()
		# clean up
		disco.purge(jobname(Zwrap.urls[0]))
		# validate
		self.validate(alpha, beta, transA, transB, A, B, C, Z)

	def test1(self):
		'''
		Test the case where size of matrix is smaller than the number of inputs for A, B and C.
		'''
		random.seed(13)
		m, k, n = 2, 1, 2
		transA, transB = False, False
		alpha, beta = 1.0, 1.0
		sparsityA, sparsityB, sparsityC = 0.0, 0.0, 0.0
		protocolA, protocolB, protocolC = MatrixWrapper.RAW, MatrixWrapper.RAW, MatrixWrapper.RAW
		dfsDirA, dfsDirB, dfsDirC = None, None, None
		maxCoresA, maxCoresB, maxCoresC = 13, 11, 7
		maxTotalBlocks = 10
		# instantiate A
		A = randomSparseMatrix(m, k, sparsityA)
		Awrap = MatrixWrapper.wrapMatrix(A, protocolA, dfsDirA, maxCoresA)
		# instantiate B
		B = randomSparseMatrix(k, n, sparsityB)
		Bwrap = MatrixWrapper.wrapMatrix(B, protocolB, dfsDirB, maxCoresB)
		# instantiate C
		C = randomSparseMatrix(m, n, sparsityC)
		Cwrap = MatrixWrapper.wrapMatrix(C, protocolC, dfsDirC, maxCoresC)
		# multiply
		Zwrap = dgemm(disco, transA, transB, m, n, k, alpha, Awrap, Bwrap, beta, Cwrap, maxTotalBlocks)
		Z = Zwrap.unwrapMatrix(m, n).todense()
		# clean up
		disco.purge(jobname(Zwrap.urls[0]))
		# validate
		self.validate(alpha, beta, transA, transB, A, B, C, Z)

	def test2(self):
		'''
		Test the case where C is an empty wrapper.
		'''
		random.seed(13)
		m, k, n = 22, 11, 21
		transA, transB = False, False
		alpha, beta = 1.0, 1.0
		sparsityA, sparsityB = 0.0, 0.0
		protocolA, protocolB = MatrixWrapper.RAW, MatrixWrapper.RAW
		dfsDirA, dfsDirB = None, None
		maxCoresA, maxCoresB = 13, 11
		maxTotalBlocks = 10
		# instantiate A
		A = randomSparseMatrix(m, k, sparsityA)
		Awrap = MatrixWrapper.wrapMatrix(A, protocolA, dfsDirA, maxCoresA)
		# instantiate B
		B = randomSparseMatrix(k, n, sparsityB)
		Bwrap = MatrixWrapper.wrapMatrix(B, protocolB, dfsDirB, maxCoresB)
		# instantiate C
		Cwrap = MatrixWrapper()
		# multiply
		Zwrap = dgemm(disco, transA, transB, m, n, k, alpha, Awrap, Bwrap, beta, Cwrap, maxTotalBlocks)
		Z = Zwrap.unwrapMatrix(m, n).todense()
		# clean up
		disco.purge(jobname(Zwrap.urls[0]))
		# validate
		self.validate(alpha, beta, transA, transB, A, B, C, Z)

	def test3(self):
		'''
		Test the case where alpha=0.
		'''
		random.seed(13)
		m, k, n = 11, 13, 23
		transA, transB = False, False
		alpha, beta = 0.0, 1.0
		sparsityA, sparsityB, sparsityC = 0.0, 0.0, 0.0
		protocolA, protocolB, protocolC = MatrixWrapper.RAW, MatrixWrapper.RAW, MatrixWrapper.RAW
		dfsDirA, dfsDirB, dfsDirC = None, None, None
		maxCoresA, maxCoresB, maxCoresC = 1, 3, 1
		maxTotalBlocks = 2 
		# instantiate A
		A = randomSparseMatrix(m, k, sparsityA)
		Awrap = MatrixWrapper.wrapMatrix(A, protocolA, dfsDirA, maxCoresA)
		# instantiate B
		B = randomSparseMatrix(k, n, sparsityB)
		Bwrap = MatrixWrapper.wrapMatrix(B, protocolB, dfsDirB, maxCoresB)
		# instantiate C
		C = randomSparseMatrix(m, n, sparsityC)
		Cwrap = MatrixWrapper.wrapMatrix(C, protocolC, dfsDirC, maxCoresC)
		# multiply
		Zwrap = dgemm(disco, transA, transB, m, n, k, alpha, Awrap, Bwrap, beta, Cwrap, maxTotalBlocks)
		Z = Zwrap.unwrapMatrix(m, n).todense()
		# clean up
		disco.purge(jobname(Zwrap.urls[0]))
		# validate
		self.validate(alpha, beta, transA, transB, A, B, C, Z)

	def test4(self):
		'''
		Test the rank-1 case where dim(A)=(m,1) and dim(B)=(1,n).
		'''
		random.seed(13)
		m, k, n = 301, 1, 223
		transA, transB = False, False
		alpha, beta = 2.27, -3.14
		sparsityA, sparsityB, sparsityC = 0.4, 0.3, 0.6
		protocolA, protocolB, protocolC = MatrixWrapper.RAW, MatrixWrapper.RAW, MatrixWrapper.RAW
		dfsDirA, dfsDirB, dfsDirC = None, None, None
		maxCoresA, maxCoresB, maxCoresC = 17, 11, 23 
		maxTotalBlocks = 13
		# instantiate A
		A = randomSparseMatrix(m, k, sparsityA)
		Awrap = MatrixWrapper.wrapMatrix(A, protocolA, dfsDirA, maxCoresA)
		# instantiate B
		B = randomSparseMatrix(k, n, sparsityB)
		Bwrap = MatrixWrapper.wrapMatrix(B, protocolB, dfsDirB, maxCoresB)
		# instantiate C
		C = randomSparseMatrix(m, n, sparsityC)
		Cwrap = MatrixWrapper.wrapMatrix(C, protocolC, dfsDirC, maxCoresC)
		# multiply
		Zwrap = dgemm(disco, transA, transB, m, n, k, alpha, Awrap, Bwrap, beta, Cwrap, maxTotalBlocks)
		Z = Zwrap.unwrapMatrix(m, n).todense()
		# clean up
		disco.purge(jobname(Zwrap.urls[0]))
		# validate
		self.validate(alpha, beta, transA, transB, A, B, C, Z)

	def test5(self):
		'''
		Test transpose.
		'''
		random.seed(13)
		m, k, n = 31, 11, 31 
		transA, transB = True, True
		alpha, beta = -3.1415, 0.5
		sparsityA, sparsityB, sparsityC = 0.2, 0.3, 0.4
		protocolA, protocolB, protocolC = MatrixWrapper.RAW, MatrixWrapper.RAW, MatrixWrapper.RAW
		dfsDirA, dfsDirB, dfsDirC = None, None, None
		maxCoresA, maxCoresB, maxCoresC = 13, 11, 7
		maxTotalBlocks = 10
		# instantiate A
		A = randomSparseMatrix(k, m, sparsityA)
		Awrap = MatrixWrapper.wrapMatrix(A, protocolA, dfsDirA, maxCoresA)
		# instantiate B
		B = randomSparseMatrix(n, k, sparsityB)
		Bwrap = MatrixWrapper.wrapMatrix(B, protocolB, dfsDirB, maxCoresB)
		# instantiate C
		C = randomSparseMatrix(m, n, sparsityC)
		Cwrap = MatrixWrapper.wrapMatrix(C, protocolC, dfsDirC, maxCoresC)
		# multiply
		Zwrap = dgemm(disco, transA, transB, m, n, k, alpha, Awrap, Bwrap, beta, Cwrap, maxTotalBlocks)
		Z = Zwrap.unwrapMatrix(m, n).todense()
		# clean up
		disco.purge(jobname(Zwrap.urls[0]))
		# validate
		self.validate(alpha, beta, transA, transB, A, B, C, Z)

if __name__ == "__main__":
	unittest.main()

