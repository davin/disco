import unittest
import os
from operations import *
from test.common import *
from disco.util import jobname

class TestOnes(unittest.TestCase):
	def validate(self, m, n, Z, tol=0.00000000001):
		assert m == Z.shape[0] and n == Z.shape[1], "expecting dimension (%d,%d) but received (%d,%d)" % (m, n, Z.shape[0], Z.shape[1])
		for i in range(0, m):
			for j in range(0, n):
				assert Z[i,j] == 1, "expecting 1 but received %.14f for element (%d,%d)" % (Z[i,j], i, j)

	def test1(self):
		"""
		Test normal usage.
		"""
		random.seed(13)
		m, n = 101, 51
		maxCores = 3
		# generate 1- matrix 
		Zwrap = ones(disco, m, n, maxCores)
		Z = Zwrap.unwrapMatrix(m, n).todense()
		# clean up
		disco.purge(jobname(Zwrap.urls[0]))
		# validate
		self.validate(m, n, Z)

if __name__ == "__main__":
	unittest.main()
