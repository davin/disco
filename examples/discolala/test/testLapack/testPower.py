import unittest
from lapack import *
from test.common import *
from operations import *
import scipy.linalg as linalg

class TestPower(unittest.TestCase):
	def validate(self, transA, n, A, v, tol):
		# compute expected dominant eigenvector in 1-norm
		print "computing expected dominant eigenvector..."
		sys.stdout.flush()
		if transA:
			la, v_exp = linalg.eig(A.transpose().todense())
		else:
			la, v_exp = linalg.eig(A.todense())
		la = map(abs, list(la))
		idx = la.index(max(la))
		v1_exp = map(abs, list(v_exp[:,idx]))
		norm1 = sum([abs(i) for i in v1_exp])
		v1_exp = [i/norm1 for i in v1_exp]
		# compare expected vs. actual
		for i in range(0, n):
			diff = abs(v[i,0] - v1_exp[i])
			assert abs(diff) <= tol, "diff of %.14f exceeds threshold" % diff

	def test1(self):
		# instantiate dense positive semi-definite matrix
		tol = 0.1
		n = 100
		transA = False
		sparsity = 0.0
		lb, ub = -10, 10
		cores = 2 
		print "initializing a random positive semi-definite matrix..."
		sys.stdout.flush()
		import time
		startTime = time.time()
		Bwrap = rand(disco, n, n, sparsity, lb, ub, cores)
		Awrap = dgemm(disco, True, False, n, n, n, 1.0, Bwrap, Bwrap, 0, MatrixWrapper(), maxTotalBlocks=cores)
		Bwrap.purge(disco)
		A = Awrap.unwrapMatrix(n, n, dtype=float64)
		print "time elapsed: %f mins" % ((time.time()- startTime)/60.0)
		# compute eigenvector
		print "computing dominant eigenvector..."
		sys.stdout.flush()
		startTime = time.time()
		vWrap = power(disco, transA, n, Awrap, cores, tol/20)
		v = vWrap.unwrapMatrix(n, 1, dtype=float64).todense()
		print "time elapsed: %f mins" % ((time.time()- startTime)/60.0)
		sys.stdout.flush()
		# validate
		self.validate(transA, n, A, v, tol)
		# clean up
		Awrap.purge(disco)
		vWrap.purge(disco)

if __name__ == "__main__":
	unittest.main()

