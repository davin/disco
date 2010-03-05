import unittest
import os
from operations import *
from test.common import *
from disco.util import jobname

class TestCrop(unittest.TestCase):
	def validate(self, transA, minRowId, minColId, maxRowId, maxColId, A, Z, tol=0.00000000001):
		A = A.todense()
		if transA:
			A = A.transpose()
		for i in range(0, A.shape[0]):
			for j in range(0, A.shape[1]):
				if i >= minRowId and i <= maxRowId and j >= minColId and j <= maxColId:
					assert abs(A[i,j]-Z[i,j]) <= tol, "expecting %.14f but received %.14f for element (%d,%d)" % (A[i,j], Z[i,j], i, j)
				else:
					assert Z[i,j] == 0, "expecting 0 but received %.14f for element (%d,%d)" % (Z[i,j], i, j)

	def test1(self):
		"""
		Test normal usage.
		"""
		random.seed(13)
		m, n = 101, 51
		transA = False
		minRowId, minColId, maxRowId, maxColId = 10, 20, 30, 40
		sparsityA = 0.5
		protocolA = MatrixWrapper.RAW
		dfsDirA = None
		maxCoresA = 13
		maxCores = 3
		# instantiate A
		A = randomSparseMatrix(m, n, sparsityA)
		Awrap = MatrixWrapper.wrapMatrix(A, protocolA, dfsDirA, maxCoresA)
		# crop matrix 
		Zwrap = crop(disco, transA, m, n, Awrap, minRowId, minColId, maxRowId, maxColId, maxCores)
		Z = Zwrap.unwrapMatrix(m, n).todense()
		# clean up
		disco.purge(jobname(Zwrap.urls[0]))
		# validate
		self.validate(transA, minRowId, minColId, maxRowId, maxColId, A, Z)

	def test2(self):
		"""
		Test transpose. 
		"""
		random.seed(13)
		m, n = 101, 51
		transA = True 
		minRowId, minColId, maxRowId, maxColId = 10, 20, 30, 40
		sparsityA = 0.5
		protocolA = MatrixWrapper.RAW
		dfsDirA = None
		maxCoresA = 13
		maxCores = 3
		# instantiate A
		A = randomSparseMatrix(n, m, sparsityA)
		Awrap = MatrixWrapper.wrapMatrix(A, protocolA, dfsDirA, maxCoresA)
		# crop matrix 
		Zwrap = crop(disco, transA, m, n, Awrap, minRowId, minColId, maxRowId, maxColId, maxCores)
		Z = Zwrap.unwrapMatrix(m, n).todense()
		# clean up
		disco.purge(jobname(Zwrap.urls[0]))
		# validate
		self.validate(transA, minRowId, minColId, maxRowId, maxColId, A, Z)

if __name__ == "__main__":
	unittest.main()
