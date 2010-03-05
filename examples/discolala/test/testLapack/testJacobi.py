import unittest
from lapack import *
from test.common import *
from operations import *
import scipy.linalg as linalg

class TestJacobi(unittest.TestCase):
	def validate(self, transA, n, A, b, x, tol):
		xExp = linalg.solve(A.todense(), b.todense())
		for i in range(0, n):
			assert abs(xExp[i]-x[i,0]) <= tol, "expecting %.14f for element %d but received %.14f" % (xExp[i], i, x[i])

	def test2(self):
		"""
		Test with a diagonally dominant upper triangular matrix A.
		"""
		from numpy import arange, random, ones
		from scipy import sparse
		from scipy.sparse import coo_matrix
		# create matrix A
		print "initializing an upper triangular matrix..."
		n = 100
		transA = False
		cores = 2
		A = sparse.lil_matrix((n,n))
		A.setdiag(ones(n))
		for i in range(0, n):
			for j in range(i, n):
				if i == j:
					A[i,j] = 99 
				else:
					A[i,j] = (33 - j + i) % 3
		Awrap = MatrixWrapper.wrapMatrix(A.tocoo(), MatrixWrapper.RAW)
		# create vector b
		b = sparse.lil_matrix((n, 1))
		for i in range(0, n):
			b[i,0] = i
		bWrap = MatrixWrapper.wrapMatrix(b.tocoo(), MatrixWrapper.RAW)
		# find least squares solution
		import time
		print "solving for x..."
		startTime = time.time()
		xWrap = jacobi(disco, transA, n, Awrap, bWrap, cores, tol=0.0001)
		x = xWrap.unwrapMatrix(n, 1, dtype=float64).todense()
		print "total time for solving x: %f mins" % ((time.time()- startTime)/60.0)
		# validate
		self.validate(transA, n, A, b, x, tol=0.01)
		# clean up
		xWrap.purge(disco)

	def test1(self):
		"""
		Test with a random diagonally dominant postive definite matrix A.
		"""
		from numpy import arange, random
		from scipy import sparse
		from scipy.sparse import coo_matrix
		# create matrix A
		n = 100
		sparsity = 0
		lb, ub = 1, 10
		transA = False
		cores = 2
		print "initializing a random positive definite matrix..."
		sys.stdout.flush()
		import time
		startTime = time.time()
		Dwrap = rand(disco, n, n, sparsity, lb, ub, cores)
		f = lambda args: 0 if args['i']<['j'] else args['v']
		Cwrap = replace(disco, n, n, Dwrap, f, cores)
		#nrm1 = dgemnrm1(disco, False, n, n, Cwrap, cores)
		g = lambda args: 1000000 if args['i']==args['j'] else args['v']
		Bwrap = replace(disco, n, n, Cwrap, g, cores)
		Awrap = dgemm(disco, True, False, n, n, n, 1.0, Bwrap, Bwrap, 0, MatrixWrapper(), maxTotalBlocks=cores)
		A = Awrap.unwrapMatrix(n, n, dtype=float64)
		Bwrap.purge(disco)
		Cwrap.purge(disco)
		Dwrap.purge(disco)
		# create vector b
		b = sparse.lil_matrix((n, 1))
		for i in range(0, n):
			b[i,0] = i
		bWrap = MatrixWrapper.wrapMatrix(b.tocoo(), MatrixWrapper.RAW)
		# find least squares solution
		import time
		print "solving for x..."
		startTime = time.time()
		xWrap = jacobi(disco, transA, n, Awrap, bWrap, cores, tol=0.01)
		x = xWrap.unwrapMatrix(n, 1, dtype=float64).todense()
		print "total time for solving x: %f mins" % ((time.time()- startTime)/60.0)
		# validate
		self.validate(transA, n, A, b, x, tol=0.1)
		# clean up
		xWrap.purge(disco)

if __name__ == "__main__":
	unittest.main()
