import unittest
import os, tempfile
from operations import *
from matrixWrap import *
from numpy import float64
from scipy import sparse
from scipy.sparse import coo_matrix
from common import *
from disco.util import jobname

class TestMatrixWrap(unittest.TestCase):
	def testSplitFileToFiles(self):
		testDir = tempfile.gettempdir() + "/" + "testMisc"
		if not os.path.exists(testDir):
			os.mkdir(testDir)
		for totalLines in range(0, 20):
			for totalFiles in range(1, 30):
				# clear target directory
				[os.remove(testDir+"/"+f) for f in os.listdir(testDir)]
				# create a file with random lines
				expectedArr = [i for i in range(0, totalLines)]
				srcFile = testDir + "/" + "test.tmp"
				file(srcFile, "w").write("\n".join(map(str, expectedArr)))
				# split file to many
				splitFileToFiles(srcFile, testDir, "test", totalFiles)
				os.remove(srcFile)
				# load content from split files
				actualArr = []
				for f in os.listdir(testDir):
					infile = file(testDir+"/"+f, "r")
					for line in infile:
						actualArr.append(int(line.strip()))
					infile.close()
				# compare expected vs actual
				assert expectedArr == sorted(actualArr), "actual != expected"
		# clear target directory
		[os.remove(testDir+"/"+f) for f in os.listdir(testDir)]
		os.removedirs(testDir)

	def testSplitMatrixToFiles(self):
		from numpy import random, float64
		random.seed(13)
		testDir = tempfile.gettempdir() + "/" + "testMisc"
		if not os.path.exists(testDir):
			os.mkdir(testDir)
		for m in range(1, 5):
			for n in range(1, 5):
				# create a m-by-n matrix
				rows = []
				cols = []
				vals = []
				for i in range(0, m):
					for j in range(0, n):
						rows.append(i)
						cols.append(j)
						vals.append(random.random())
				M = coo_matrix((vals, (rows,cols)), dims=(m,n))
				for totalFiles in range(1, 26):
					for elemsPerLine in range(1, 25):
						# clear target directory
						[os.remove(testDir+"/"+f) for f in os.listdir(testDir)]
						# write matrix to X files
						splitMatrixToFiles(M, testDir, "test", totalFiles, elemsPerLine, dtype=float64)
						# reconstruct matrix from split files
						rows = []
						cols = []
						vals = []
						for f in os.listdir(testDir):
							infile = file(testDir+"/"+f, "r")
							for line in infile:
								elems = line.strip().split(";")
								for elem in elems:
									i,j,val = elem.split(",")
									rows.append(int(i))
									cols.append(int(j))
									vals.append(float64(val))
							infile.close()
						Mnew = coo_matrix((vals, (rows,cols)), dims=(m,n))
						D = Mnew - M
						for i in range(0, m):
							for j in range(0, n):
								assert abs(D[i,j]) < 0.0000001, "diff between expected M[%d,%d] and actual M[%d,%d]=%.14f exceeds threshold" % (i,j,i,j,D[i,j])

if __name__ == "__main__":
	unittest.main()
