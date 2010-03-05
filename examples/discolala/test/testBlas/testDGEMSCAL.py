import unittest
from blas import dgemscal
from test.common import *
from disco.util import jobname

class TestDGEMSCAL(unittest.TestCase):
	def validate(self, transA, alpha, A, Z, tol=0.00000000001):
		if transA:
			A = A.transpose()
		Acoo = A.tocoo()
		Zexp = (alpha * A).todense()
		D = Zexp - Z 
		for i in range(0, D.shape[0]):
			for j in range(0, D.shape[1]):
				assert abs(D[i,j]) <= tol, "expecting %.14f but received %.14f for element (%d,%d)" % (Zexp[i,j], Z[i,j], i,j)

	def test0(self):
		'''
		Test normal basic usage.
		'''
		random.seed(13)
		m, n = 12, 340
		transA = False
		alpha = -3.14
		sparsityA = 0.3
		protocolA = MatrixWrapper.RAW
		dfsDirA = None
		maxCoresA = 13
		maxTotalBlocks = 10
		# instantiate A
		A = randomSparseMatrix(m, n, sparsityA)
		Awrap = MatrixWrapper.wrapMatrix(A, protocolA, dfsDirA, maxCoresA)
		# sum 
		Zwrap = dgemscal(disco, transA, m, n, alpha, Awrap, maxTotalBlocks)
		Z = Zwrap.unwrapMatrix(m, n).todense()
		# clean up
		disco.purge(jobname(Zwrap.urls[0]))
		# validate
		self.validate(transA, alpha, A, Z)

	def test1(self):
		'''
		Test transpose. 
		'''
		random.seed(13)
		m, n = 13, 123
		transA = True
		alpha = 3.14
		sparsityA = 0.8
		protocolA = MatrixWrapper.RAW
		dfsDirA = None
		maxCoresA = 13
		maxTotalBlocks = 10
		# instantiate A
		A = randomSparseMatrix(n, m, sparsityA)
		Awrap = MatrixWrapper.wrapMatrix(A, protocolA, dfsDirA, maxCoresA)
		# sum 
		Zwrap = dgemscal(disco, transA, m, n, alpha, Awrap, maxTotalBlocks)
		Z = Zwrap.unwrapMatrix(m, n).todense()
		# clean up
		disco.purge(jobname(Zwrap.urls[0]))
		# validate
		self.validate(transA, alpha, A, Z)

if __name__ == "__main__":
	unittest.main()


