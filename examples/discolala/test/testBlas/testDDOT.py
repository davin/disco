import unittest
from blas import ddot 
from test.common import *
from disco.util import jobname

class TestDDOT(unittest.TestCase):
	def validate(self, n, x, y, val, tol=0.00000000001):
		x = x.todense()
		y = y.todense()
		valExp = 0
		for i in range(0, n):
			if x.shape[0] >= x.shape[1]:
				if y.shape[0] >= y.shape[1]:
					valExp += x[i,0] * y[i,0]
			if x.shape[0] < x.shape[1]:
				if y.shape[0] >= y.shape[1]:
					valExp += x[0,i] * y[i,0]
			if x.shape[0] >= x.shape[1]:
				if y.shape[0] < y.shape[1]:
					valExp += x[i,0] * y[0,i]
			if x.shape[0] < x.shape[1]:
				if y.shape[0] < y.shape[1]:
					valExp += x[0,i] * y[0,i]
		assert abs(valExp - val) < tol, "expecting %.14f but received %.14f" % (valExp, val)

	def test0(self):
		'''
		Test row vector dot col vector.
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
		# instantiate y
		sparsityY = 0.2
		protocolY = MatrixWrapper.RAW
		dfsDirY = None
		maxCoresY = 13
		y = randomSparseMatrix(n, 1, sparsityY)
		yWrap = MatrixWrapper.wrapMatrix(y, protocolY, dfsDirY, maxCoresY)
		# dot product 
		val = ddot(disco, n, xWrap, yWrap, maxTotalBlocks)
		# validate
		self.validate(n, x, y, val)

	def test1(self):
		'''
		Test row vector dot row vector.
		'''
		random.seed(13)
		n = 99
		maxTotalBlocks = 1
		# instantiate x
		sparsityX = 0.1
		protocolX = MatrixWrapper.RAW
		dfsDirX = None
		maxCoresX = 5
		x = randomSparseMatrix(1, n, sparsityX)
		xWrap = MatrixWrapper.wrapMatrix(x, protocolX, dfsDirX, maxCoresX)
		# instantiate y
		sparsityY = 0.2
		protocolY = MatrixWrapper.RAW
		dfsDirY = None
		maxCoresY = 7
		y = randomSparseMatrix(1, n, sparsityY)
		yWrap = MatrixWrapper.wrapMatrix(y, protocolY, dfsDirY, maxCoresY)
		# dot product 
		val = ddot(disco, n, xWrap, yWrap, maxTotalBlocks)
		# validate
		self.validate(n, x, y, val)

	def test2(self):
		'''
		Test col vector dot row vector.
		'''
		random.seed(13)
		n = 432
		maxTotalBlocks = 13
		# instantiate x
		sparsityX = 0.2
		protocolX = MatrixWrapper.RAW
		dfsDirX = None
		maxCoresX = 13
		x = randomSparseMatrix(n, 1, sparsityX)
		xWrap = MatrixWrapper.wrapMatrix(x, protocolX, dfsDirX, maxCoresX)
		# instantiate y
		sparsityY = 0.5
		protocolY = MatrixWrapper.RAW
		dfsDirY = None
		maxCoresY = 13
		y = randomSparseMatrix(1, n, sparsityY)
		yWrap = MatrixWrapper.wrapMatrix(y, protocolY, dfsDirY, maxCoresY)
		# dot product 
		val = ddot(disco, n, xWrap, yWrap, maxTotalBlocks)
		# validate
		self.validate(n, x, y, val)

	def test3(self):
		'''
		Test col vector dot col vector.
		'''
		random.seed(13)
		n = 34
		maxTotalBlocks = 6 
		# instantiate x
		sparsityX = 0.1
		protocolX = MatrixWrapper.RAW
		dfsDirX = None
		maxCoresX = 3
		x = randomSparseMatrix(n, 1, sparsityX)
		xWrap = MatrixWrapper.wrapMatrix(x, protocolX, dfsDirX, maxCoresX)
		# instantiate y
		sparsityY = 0.2
		protocolY = MatrixWrapper.RAW
		dfsDirY = None
		maxCoresY = 4
		y = randomSparseMatrix(n, 1, sparsityY)
		yWrap = MatrixWrapper.wrapMatrix(y, protocolY, dfsDirY, maxCoresY)
		# dot product 
		val = ddot(disco, n, xWrap, yWrap, maxTotalBlocks)
		# validate
		self.validate(n, x, y, val)

if __name__ == "__main__":
	unittest.main()
