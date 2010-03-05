import unittest
from lapack import *
from test.common import *
from operations import *
import scipy.linalg as linalg

class TestCg(unittest.TestCase):
	def validate(self, transA, n, A, b, x, tol):
		xExp = linalg.solve(A.todense(), b.todense())
		for i in range(0, n):
			assert abs(xExp[i]-x[i,0]) <= tol, "expecting %.14f for element %d but received %.14f" % (xExp[i], i, x[i])

	def test1(self):
		"""
		Test with a random postive definite matrix A.
		"""
		from numpy import arange, random
		from scipy import sparse
		from scipy.sparse import coo_matrix
		# create matrix A
		n = 500 
		sparsity = 0.3
		lb, ub = 0.0, 1.0 
		transA = False
		cores = 2
		print "initializing a random positive definite matrix..."
		sys.stdout.flush()
		import time
		startTime = time.time()
		"""
		random.seed(13)
		S = randomSparseMatrix(n, n, sparsity)
		Swrap = MatrixWrapper.wrapMatrix(S, MatrixWrapper.RAW, None, cores)
		"""
		Swrap = rand(disco, n, n, sparsity, lb, ub, cores)
		f = lambda args: 0 if args['i']<args['j'] else args['v']
		Twrap = replace(disco, n, n, Swrap, f, cores=cores)
		Awrap = dgema(disco, False, True, n, n, 0.5, Twrap, Twrap, 0.5, maxTotalBlocks=cores)
		A = Awrap.unwrapMatrix(n, n, dtype=float64)
		#Swrap.purge(disco)
		Twrap.purge(disco)
		# create vector b
		b = sparse.lil_matrix((n, 1))
		for i in range(0, n):
			b[i,0] = i
		bWrap = MatrixWrapper.wrapMatrix(b.tocoo(), MatrixWrapper.RAW)
		# find least squares solution
		import time
		print "solving for x..."
		startTime = time.time()
		xWrap = cg(disco, n, Awrap, bWrap, cores, tol=0.01)
		x = xWrap.unwrapMatrix(n, 1, dtype=float64).todense()
		print "total time for solving x: %f mins" % ((time.time()- startTime)/60.0)
		# validate
		self.validate(transA, n, A, b, x, tol=0.1)
		# clean up
		xWrap.purge(disco)

if __name__ == "__main__":
	unittest.main()
