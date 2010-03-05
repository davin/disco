import unittest
from operations import *
from test.common import *
from disco.util import jobname

class TestRandSym(unittest.TestCase):
	def validate(self, n, sparsity, lb, ub, M, tol):
		nnzExp = (1-sparsity) * n**2
		nnzAct = len(M.tocoo().data)
		# check sparsity
		assert abs(nnzAct-nnzExp) < tol, "expecting %d non-zero elements but received %d" % (nnzExp, nnzAct)
		M = M.todense()
		for i in range(0, n):
			for j in range(i, n):
				# check bound 
				assert M[i,j] >= lb and M[i,j] < ub, "element (%d,%d) value of %f exceeds [%f,%f)" % (i,j,M[i,j],lb,ub)
				# check symmetry
				assert M[i,j] == M[j,i], "element (%d,%d) != (%d,%d)" % (i,j,j,i)

	def test1(self):
		"""
		Test normal usage.
		"""
		random.seed(13)
		from numpy import arange
		lb, ub = -3.0, 4.0
		n = 500 
		sparsity = 0.5
		maxCores = 4
		# generate random matrix
		Mwrap = randSym(disco, n, sparsity, lb, ub, maxCores)
		M = Mwrap.unwrapMatrix(n, n)
		# validate
		self.validate(n, sparsity, lb, ub, M, 0.05*n**2)
		# clean up
		disco.purge(jobname(Mwrap.urls[0]))

if __name__ == "__main__":
	unittest.main()
