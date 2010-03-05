import unittest
import os
from operations import *
from test.common import *
from disco.util import jobname

class TestReplace(unittest.TestCase):
	def validate(self, minRowId, minColId, maxRowId, maxColId, A, Z, tol=0.00000000001):
		A = A.todense()
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
		minRowId, minColId, maxRowId, maxColId = 10, 20, 30, 40
		sparsityA = 0.5
		protocolA = MatrixWrapper.RAW
		dfsDirA = None
		maxCoresA = 13
		maxCores = 3
		# instantiate A
		A = randomSparseMatrix(m, n, sparsityA)
		Awrap = MatrixWrapper.wrapMatrix(A, protocolA, dfsDirA, maxCoresA)
		# replace elements in matrix
		args = {}
		args['minRowId'] = minRowId 
		args['maxRowId'] = maxRowId
		args['minColId'] = minColId
		args['maxColId'] = maxColId
		f = lambda args: args['v'] if args['i']>=args['minRowId'] and args['i']<=args['maxRowId'] and args['j']>=args['minColId'] and args['j']<=args['maxColId'] else 0
		Zwrap = replace(disco, m, n, Awrap, f, args, maxCores)
		Z = Zwrap.unwrapMatrix(m, n).todense()
		# clean up
		disco.purge(jobname(Zwrap.urls[0]))
		# validate
		self.validate(minRowId, minColId, maxRowId, maxColId, A, Z)

	def test2(self):
		"""
		Test input matrix with raw protocol.
		"""
		random.seed(13)
		n = 100
		sparsity = 0
		# setting cores to 1 will break from long raw input string
		cores = 2
		A = randomSparseMatrix(n, n, sparsity)
		Awrap = MatrixWrapper.wrapMatrix(A, MatrixWrapper.RAW, None, cores)
		# create upper triangular matrix
		f = lambda args: 0 if args['i']<args['j'] else args['v']
		Uwrap = replace(disco, n, n, Awrap, f, cores=cores)
		U = Uwrap.unwrapMatrix(n, n).todense()
		# validate
		A = A.todense()
		for i in range(0, A.shape[0]):
			for j in range(0, A.shape[1]):
				if i < j:
					assert U[i,j] == 0, "expecting 0 but received %.14f for element (%d,%d)" % (U[i,j], i, j)
				else:
					assert U[i,j] != 0, "expecting %.14f but received 0 for element (%d,%d)" % (A[i,j], i, j)

if __name__ == "__main__":
	unittest.main()
