import unittest
from blas import dgemsum
from test.common import *
from disco.util import jobname

class TestDGEMSUM(unittest.TestCase):
	def validate(self, transA, A, Z, tol=0.00000000001):
		if transA:
			A = A.transpose()
		Acoo = A.tocoo()
		colSums = [0] * A.shape[1]
		for j in range(0, len(Acoo.data)):
			colSums[Acoo.col[j]] += Acoo.data[j]
		for j in range(0, len(colSums)):
			assert abs(colSums[j]-Z[0,j]) <= tol, "expecting %.14f but received %.14f for element %d" % (colSums[j], Z[0,j], j)

	def test0(self):
		'''
		Test normal basic usage.
		'''
		random.seed(13)
		m, n = 12, 340
		transA = False
		sparsityA = 0.3
		protocolA = MatrixWrapper.RAW
		dfsDirA = None
		maxCoresA = 13
		maxTotalBlocks = 10
		# instantiate A
		A = randomSparseMatrix(m, n, sparsityA)
		Awrap = MatrixWrapper.wrapMatrix(A, protocolA, dfsDirA, maxCoresA)
		# sum 
		Zwrap = dgemsum(disco, transA, m, n, Awrap, maxTotalBlocks)
		Z = Zwrap.unwrapMatrix(1, n).todense()
		# clean up
		disco.purge(jobname(Zwrap.urls[0]))
		# validate
		self.validate(transA, A, Z)

	def test1(self):
		'''
		Test transpose. 
		'''
		random.seed(13)
		m, n = 13, 123
		transA = True 
		sparsityA = 0.8
		protocolA = MatrixWrapper.RAW
		dfsDirA = None
		maxCoresA = 13
		maxTotalBlocks = 10
		# instantiate A
		A = randomSparseMatrix(n, m, sparsityA)
		Awrap = MatrixWrapper.wrapMatrix(A, protocolA, dfsDirA, maxCoresA)
		# sum 
		Zwrap = dgemsum(disco, transA, m, n, Awrap, maxTotalBlocks)
		Z = Zwrap.unwrapMatrix(1, n).todense()
		# clean up
		disco.purge(jobname(Zwrap.urls[0]))
		# validate
		self.validate(transA, A, Z)

if __name__ == "__main__":
	unittest.main()


