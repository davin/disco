import unittest

def suite():
	"""
	Test all LAPACK subroutines in one go.
	"""
	suite1 = unittest.makeSuite(TestDGEMA, 'test')
	return unittest.TestSuite((suite1))

if __name__ == "__main__":
	# run all tests
	runner = unittest.TextTestRunner()
	runner.run(suite())

