import unittest
from testCrop import *
from testDiag import *
from testEye import *
from testMinDim import *
from testNnz import *
from testOnes import *
from testRand import *
from testRandSym import *
from testReplace import *
from testTriu import *
from testTril import *

def suite():
	"""
	Test all matrix operations.
	"""
	suite1 = unittest.makeSuite(TestCrop, 'test')
	suite2 = unittest.makeSuite(TestDiag, 'test')
	suite3 = unittest.makeSuite(TestEye, 'test')
	suite4 = unittest.makeSuite(TestMinDim, 'test') 
	suite5 = unittest.makeSuite(TestNnz, 'test')
	suite6 = unittest.makeSuite(TestOnes, 'test')
	suite7 = unittest.makeSuite(TestRand, 'test')
	suite8 = unittest.makeSuite(TestRandSym, 'test')
	suite9 = unittest.makeSuite(TestReplace, 'test')
	suite10 = unittest.makeSuite(TestTriu, 'test')
	suite11 = unittest.makeSuite(TestTril, 'test')
	return unittest.TestSuite((suite1, suite2, suite3, suite4, suite5, suite6, suite7, suite8, suite9, suite10, suite11))

if __name__ == "__main__":
	# run all tests
	runner = unittest.TextTestRunner()
	runner.run(suite())

