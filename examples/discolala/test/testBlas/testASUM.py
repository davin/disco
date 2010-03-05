import unittest
from blas import dasum 
from test.common import *
from disco.util import jobname

class TestASUM(unittest.TestCase):
	def validate(self, n, x, val, tol=0.000000001):
		x = x.todense()
		valExp = 0
		for i in range(0, n):
			if x.shape[0] >= x.shape[1]:
				valExp += abs(x[i,0])
			else:
				valExp += abs(x[0,i])
		assert abs(valExp - val) < tol, "expecting %.14f but received %.14f" % (valExp, val)

	def test0(self):
		'''
		Test row vector.
		'''
		random.seed(13)
		n = 234
		maxTotalBlocks = 10
		# instantiate x
		sparsityX = 0.1
		protocolX = MatrixWrapper.RAW
		dfsDirX = None
		maxCoresX = 13
		x = randomSparseMatrix(1, n, sparsityX)
		xWrap = MatrixWrapper.wrapMatrix(x, protocolX, dfsDirX, maxCoresX)
		# sum of absolute
		val = dasum(disco, n, xWrap, maxTotalBlocks)
		# validate
		self.validate(n, x, val)

	def test1(self):
		'''
		Test column vector.
		'''
		random.seed(13)
		n = 1
		maxTotalBlocks = 99
		# instantiate x
		sparsityX = 0.1
		protocolX = MatrixWrapper.RAW
		dfsDirX = None
		maxCoresX = 5
		x = randomSparseMatrix(n, 1, sparsityX)
		xWrap = MatrixWrapper.wrapMatrix(x, protocolX, dfsDirX, maxCoresX)
		# sum of absolute
		val = dasum(disco, n, xWrap, maxTotalBlocks)
		# validate
		self.validate(n, x, val)

if __name__ == "__main__":
	unittest.main()
