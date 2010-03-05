import unittest
import os
from operations import *
from test.common import *
from disco.util import jobname

class TestEye(unittest.TestCase):
	def validate(self, m, n, Z, tol=0.00000000001):
		from numpy import eye
		Zexp = eye(m, n)
		D = Zexp - Z 
		for i in range(0, D.shape[0]):
			for j in range(0, D.shape[1]):
				assert abs(D[i,j]) <= tol, "expecting %.14f but received %.14f for element (%d,%d)" % (Zexp[i,j], Z[i,j], i, j)

	def test1(self):
		"""
		Test normal usage.
		"""
		random.seed(13)
		m, n = 101, 51
		maxCores = 3
		# generate identity matrix 
		Zwrap = eye(disco, m, n, maxCores)
		Z = Zwrap.unwrapMatrix(m, n).todense()
		# clean up
		disco.purge(jobname(Zwrap.urls[0]))
		# validate
		self.validate(m, n, Z)

if __name__ == "__main__":
	unittest.main()
