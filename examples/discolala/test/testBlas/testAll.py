import unittest
from testDGEMA import *
from testDGEMM import *
from testDGEMSUM import *
from testDGEMNRM1 import *
from testDGEMSCAL import *
from testDDOT import *
from testASUM import *

def suite():
	"""
	Test all BLAS subroutines in one go.
	"""
	suite1 = unittest.makeSuite(TestDGEMA, 'test')
	suite2 = unittest.makeSuite(TestDGEMM, 'test')
	suite3 = unittest.makeSuite(TestDGEMSUM, 'test')
	suite4 = unittest.makeSuite(TestDGEMNRM1, 'test') 
	suite5 = unittest.makeSuite(TestDGEMSCAL, 'test')
	suite6 = unittest.makeSuite(TestDDOT, 'test')
	suite7 = unittest.makeSuite(TestASUM, 'test')
	return unittest.TestSuite((suite1, suite2, suite3, suite4, suite5, suite6, suite7))

if __name__ == "__main__":
	# run all tests
	runner = unittest.TextTestRunner()
	runner.run(suite())

