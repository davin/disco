import unittest
from blas import dgemnrm1 
from test.common import *
from disco.util import jobname

class TestDGEMNRM1(unittest.TestCase):
	def validate(self, transA, A, val, tol=0.00000000001):
		if transA:
			A = A.transpose()
		Acoo = A.tocoo()
		colSums = [0] * A.shape[1]
		for j in range(0, len(Acoo.data)):
			colSums[Acoo.col[j]] += abs(Acoo.data[j])
		assert max(colSums) - val < tol, "expecting %.14f but received %.14f" % (max(colSums), val)

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
		# norm 1 
		val = dgemnrm1(disco, transA, m, n, Awrap, maxTotalBlocks)
		# validate
		self.validate(transA, A, val)

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
		# norm 1
		val = dgemnrm1(disco, transA, m, n, Awrap, maxTotalBlocks)
		# validate
		self.validate(transA, A, val)
	
	def test2(self):
		'''
		Test matrix with only zeros.
		'''
		random.seed(13)
		m, n = 12, 34
		transA = False
		sparsityA = 1.0
		protocolA = MatrixWrapper.RAW
		dfsDirA = None
		maxCoresA = 7 
		maxTotalBlocks = 5
		# instantiate A
		A = randomSparseMatrix(m, n, sparsityA)
		Awrap = MatrixWrapper.wrapMatrix(A, protocolA, dfsDirA, maxCoresA)
		# norm 1 
		val = dgemnrm1(disco, transA, m, n, Awrap, maxTotalBlocks)
		# validate
		self.validate(transA, A, val)

if __name__ == "__main__":
	unittest.main()
