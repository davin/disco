import unittest
import os
from operations import *
from test.common import *
from disco.util import jobname

class TestNnz(unittest.TestCase):
	def validate(self, A, total):
		expected = len(A.tocoo().data)
		assert total == expected, "expecting %d non-zero elements but received %d" % (total, expected)

	def test1(self):
		"""
		Test normal usage.
		"""
		random.seed(13)
		m, n = 101, 51
		sparsityA = 0.5
		protocolA = MatrixWrapper.RAW
		dfsDirA = None
		maxCoresA = 13
		# instantiate A
		A = randomSparseMatrix(m, n, sparsityA)
		Awrap = MatrixWrapper.wrapMatrix(A, protocolA, dfsDirA, maxCoresA)
		# count nnz 
		total = nnz(disco, m, n, Awrap)
		# validate
		self.validate(A, total)

	def test2(self):
		"""
		Test empty matrix. 
		"""
		random.seed(13)
		m, n = 11, 555
		sparsityA = 1 
		protocolA = MatrixWrapper.RAW
		dfsDirA = None
		maxCoresA = 3
		# instantiate A
		A = randomSparseMatrix(m, n, sparsityA)
		Awrap = MatrixWrapper.wrapMatrix(A, protocolA, dfsDirA, maxCoresA)
		# count nnz 
		total = nnz(disco, m, n, Awrap)
		# validate
		self.validate(A, total)

if __name__ == "__main__":
	unittest.main()
