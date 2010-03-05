import unittest
from operations import *
from test.common import *
from disco.util import jobname

class TestRand(unittest.TestCase):
	def validate(self, m, n, sparsity, lb, ub, M, tol):
		nnzExp = (1-sparsity) * m * n
		nnzAct = len(M.tocoo().data)
		# check sparsity
		assert abs(nnzAct-nnzExp) < tol, "expecting %d non-zero elements but received %d" % (nnzExp, nnzAct)
		M = M.todense()
		for i in range(0, m):
			for j in range(0, n):
				# check bound 
				assert M[i,j] >= lb and M[i,j] < ub, "element (%d,%d) value of %f exceeds [%f,%f)" % (i,j,M[i,j],lb,ub)

	def test1(self):
		"""
		Test normal usage.
		"""
		random.seed(13)
		from numpy import arange
		lb, ub = -3.0, 4.0
		m, n = 500, 333 
		sparsity = 0.5
		maxCores = 4
		# generate random matrix
		Mwrap = rand(disco, m, n, sparsity, lb, ub, maxCores)
		M = Mwrap.unwrapMatrix(m, n)
		# validate
		self.validate(m, n, sparsity, lb, ub, M, 0.05*m*n)
		# clean up
		disco.purge(jobname(Mwrap.urls[0]))

if __name__ == "__main__":
	unittest.main()
