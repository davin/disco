import unittest
import testBlas.testAll as s
import testOperations.testAll as t
from testMatrixWrap import *

def suite():
	"""
	Test everything in one go.
	"""
	suite1 = unittest.makeSuite(TestMatrixWrap, 'test')
	suite2 = s.suite()
	suite3 = t.suite()
	return unittest.TestSuite((suite1, suite2, suite3))

if __name__ == "__main__":
	# run all tests
	runner = unittest.TextTestRunner()
	runner.run(suite())

