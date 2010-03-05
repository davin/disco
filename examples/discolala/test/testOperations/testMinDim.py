import unittest
import os
from operations import *
from test.common import *
from disco.util import jobname

class TestMinDim(unittest.TestCase):
	def validate(self, expDim, actDim):
		assert expDim == actDim, "expecting dimension (%d,%d) but received (%d,%d)" % (expDim[0], expDim[1], actDim[0], actDim[1])

	def test1(self):
		"""
		Test normal usage.
		"""
		random.seed(13)
		m, n = 101, 51
		sparsityA = 0.5
		protocolA = MatrixWrapper.RAW
		dfsDirA = None
		maxCoresA = 11
		# instantiate A
		A = randomSparseMatrix(m, n, sparsityA)
		Awrap = MatrixWrapper.wrapMatrix(A, protocolA, dfsDirA, maxCoresA)
		# get minimum dimension 
		dim = minDim(disco, Awrap)
		# validate
		self.validate((m,n), dim)

	def test2(self):
		"""
		Test empty matrix 
		"""
		random.seed(13)
		m, n = 99, 111
		sparsityA = 1 
		protocolA = MatrixWrapper.RAW
		dfsDirA = None
		maxCoresA = 5
		# instantiate A
		A = randomSparseMatrix(m, n, sparsityA)
		Awrap = MatrixWrapper.wrapMatrix(A, protocolA, dfsDirA, maxCoresA)
		# get minimum dimension 
		dim = minDim(disco, Awrap)
		# validate
		self.validate((0,0), dim)

if __name__ == "__main__":
	unittest.main()
