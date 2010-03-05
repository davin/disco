import os
from numpy import float64
from scipy.sparse import coo_matrix
from disco.func import chain_reader, map_line_reader
from disco.dfs.gluster import files
from scipy import sparse
from numpy import float64
from disco.core import result_iterator

def splitFileToFiles(srcFile, outDir, filenamePrefix, totalFiles):
	"""
	Split a file evenly into multiple files by lines.
	"""
	# count number of elements in matrix
	totalElems = 0.0
	infile = file(srcFile, "r")
	for line in infile:
		totalElems += 1
	infile.close()
	# split elements to files
	from math import ceil
	size = totalElems / totalFiles
	start = 0
	infile = file(srcFile, "r")
	for i in range(0, totalFiles):
		outfile = file("%s%d" % (outDir+"/"+filenamePrefix, i), "w")
		end = int(min(totalElems, ceil(start + size)))
		linesToRead = end - start
		linesRead  = 0
		buff = ""
		while linesRead < linesToRead:
			buff += infile.readline()
			linesRead += 1
			if linesRead % 1000000:
				outfile.write(buff)
				buff = ""
		outfile.write(buff)
		start = end
		outfile.close()
	infile.close()

def splitMatrixToFiles(M, outDir, filenamePrefix, totalFiles, elemsPerLine=1000, dtype=float64):
	"""
	Store the sparse matrix into multiple files in coordinate format with double precision.
	@param M Scipy sparse matrix.
	@param outDir Output directory.
	@params outFilename Full path of output file.
	"""
	from math import ceil
	Mcoo = M.tocoo()
	size = float(len(Mcoo.row)) / totalFiles
	start = 0
	for i in range(0, totalFiles):
		outfile = file("%s%d" % (outDir+"/"+filenamePrefix, i), "w")
		end = int(min(len(Mcoo.row), ceil(start + size)))
		p = start
		while p < end:
			q = min(p+elemsPerLine, end)
			if dtype == float64:
				outfile.write(";".join(["%d,%d,%.14f" % (Mcoo.row[j], Mcoo.col[j], Mcoo.data[j]) for j in range(p,q)]))
			else:
				outfile.write(";".join(["%d,%d,%f" % (Mcoo.row[j], Mcoo.col[j], Mcoo.data[j]) for j in range(p,q)]))
			p = q
			outfile.write("\n")
		start = end
		outfile.close()

class MatrixWrapper:
	RAW = "raw"
	DFS = "dfs"
	DIR = "dir"
	
	def __init__(self, urls=[], mapReader=map_line_reader):
		"""
		"""
		self.mapReader = mapReader 
		self.urls = urls 

	def purge(self, disco):
		"""
		Delete matrix from cluster.
		"""
		if len(self.urls) > 0:
			from disco.util import jobname
			disco.purge(jobname(self.urls[0]))

	def unwrapMatrix(self, m, n, dtype=float64):
		"""
		Instantiate matrix from wrapper.
		"""
		rows = []
		cols = []
		vals = []
		for url in self.urls:
			if type(url) == list:
				# dfs protocol is a nested list
				url = url[0]
			protocol,path = url.split("://")
			if protocol == MatrixWrapper.RAW:
				elems = path.split(";")
				for elem in elems:
					i,j,val = elem.split(",")
					rows.append(int(i))
					cols.append(int(j))
					vals.append(dtype(val))
			elif protocol == MatrixWrapper.DIR:
				total = 0
				for key, val in result_iterator([url]):
					elems = key.split(";")
					for elem in elems:
						i,j,val = elem.split(",")
						rows.append(int(i))
						cols.append(int(j))
						vals.append(dtype(val))
						total += 1
						assert total <= m*n, "cardinality of result set exceeds %dx%d=%d entries" % (m,n,m*n)
			elif protocol == MatrixWrapper.DFS:
				raise Exception('dfs protocol not supported yet')
			else:
				raise Exception('invalid protocol')
		return coo_matrix((vals,(rows,cols)), dtype=dtype, dims=(m,n))

	def unwrapMatrixFile(self, m, n, matrixFile, dtype=float64):
		"""
		Instantiate matrix from wrapper.
		"""
		outFile = file(matrixFile, 'w')
		for url in self.urls:
			protocol,path = url.split("://")
			if protocol == MatrixWrapper.RAW:
				outFile.write(path+"\n")
			elif protocol == MatrixWrapper.DIR:
				for key, val in result_iterator([url]):
					outFile.write(key+"\n")
			elif protocol == MatrixWrapper.DFS:
				raise Exception('dfs protocol not supported yet')
			else:
				raise Exception('invalid protocol')
		outFile.close()

	@staticmethod
	def wrapMatrix(A, protocol, dfsDir=None, maxCores=1):
		"""
		Factory method for wrapping a numpy/scipy matrix object.
		"""
		if type(A) == coo_matrix:
			Acoo = A.tocoo()
			if protocol == MatrixWrapper.RAW:
				raws = [[] for i in range(0, maxCores)]
				for i in range(0, len(Acoo.row)):
					raws[i%maxCores].append("%d,%d,%.14f" % (Acoo.row[i], Acoo.col[i], Acoo.data[i]))
				return MatrixWrapper(["raw://"+";".join(elems) for elems in raws if len(elems) > 0])
			elif protocol == MatrixWrapper.DFS:
				import os.path
				if not os.path.exists(dfsDir):
					os.mkdir(dfsDir)
				# delete any previous gluster files in the directory
				[os.remove(dfsDir+"/"+f) for f in os.listdir(dfsDir)]
				# save in coordinate format and split matrix to multiple files
				splitMatrixToFiles(A, dfsDir, "", maxCores)
				return MatrixWrapper(files(dfsDir))
			else:
				raise Exception('only dfs or raw protocol are supported')
		else:
			raise Exception('only coo_matrix is supported for now')

	@staticmethod
	def wrapMatrixFile(matrixFile, protocol, dfsDir=None, maxCores=1):
		"""
		Factory method for wrapping a matrix stored in a file in coordinate format.
		"""
		if protocol == MatrixWrapper.RAW:
			raws = [[] for i in range(0, maxCores)]
			i = 0
			infile = file(matrixFile, "r")
			for line in infile:
				for elem in line.strip().split(";"):
					raws[i%maxCores].append(elem)
					i += 1
			infile.close()
			return MatrixWrapper(["raw://"+";".join(elems) for elems in raws if len(elems) > 0])
		elif protocol == MatrixWrapper.DFS:
			if not os.path.exists(dfsDir):
				os.mkdir(dfsDir)
			# delete any previous gluster files in the directory
			[os.remove(dfsDir+"/"+f) for f in os.listdir(dfsDir)]
			splitFileToFiles(matrixFile, dfsDir, "", maxCores)
			return MatrixWrapper(files(dfsDir))
		else:
			raise Exception('only dfs or raw protocol are supported')

