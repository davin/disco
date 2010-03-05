from disco.core import Disco, result_iterator, Params
from disco.func import chain_reader
from matrixWrap import MatrixWrapper
from numpy import float64

def _partition(height, width, maxTotalBlocks):
	assert maxTotalBlocks > 0, "maxTotalBlocks must be >0"
	assert height > 0, "height must be >0"
	assert width > 0, "width must be >0"
	# number of blocks cannot be smaller than the size of the matrix
	maxTotalBlocks = min(height*width, maxTotalBlocks)
	# search for the best partition
	import sys
	bestRatioDiff = sys.maxint
	for maxTotalBlocks in range(max(1,maxTotalBlocks-3), maxTotalBlocks+1):
		# compute divisors for maxTotalBlocks
		for i in range(1, maxTotalBlocks+1):
			if maxTotalBlocks % i == 0:
				j = maxTotalBlocks / i
				ratioDiff = abs(float(i)/j - float(height)/width)
				if ratioDiff < bestRatioDiff:
					bestRatioDiff = ratioDiff
					blocksPerCol = i
					blocksPerRow = j
	return blocksPerRow, blocksPerCol

def ddot(disco, n, x, y, maxTotalBlocks=128):
	"""
	Compute general vector dot product <x,y> in double precision.
	@param n Number of elements in x and y.
	@param x MatrixWrapper object encapsulating vector x either as a row or column vector.
	@param y MatrixWrapper object encapsulating vector y either as a row or column vector.
	@param disco A Disco instance.
	@param maxTotalBlocks Suggested number of blocks to use for carrying out the computation. Ideally, this should equal to the number of cores available in the cluster. The actual number of blocks is selected based on the size of the vector 
	@return The dot product as a double precision scalar.
	"""
	def _mapDot(e, params):
		if type(e) == tuple:
			e = e[0]
		output = []
		elems = e.split(";")
		for elem in elems:
			i, j, val = elem.split(",")
			i, j = int(i), int(j)
			# check that matrix is a row or column vector
			# this assertion is fast but not bullet-proof since it ignores a matrix with only nnz in first col and row
			assert i*j == 0, "element index (%d,%d) exceeds vector dimension" % (i,j)
			if i > 0:
				assert i < params.n, "row index %d exceeds matrix dimensions" % i
				output += [(i, val)]
			elif j > 0:
				assert j < params.n, "col index %d exceeds matrix dimensions" % j
				output += [(j, val)]
			else:
				output += [(i, val)]
		return output

	def nop_map(e, params):
		return [e]

	def _reduceDot(iter, out, params):
		from numpy import float64
		s = {}
		h = {}
		for i, val in iter:
			i = int(i)
			s[i] = s.get(i, 1) * float64(val)
			h[i] = h.get(i, 0) + 1
		total = 0
		for i, val in s.iteritems():
			if h[i] > 1:
				total += val
		out.add(total, "")

	# map vectors
	totalBlocks = min(maxTotalBlocks, n)
	jobX = disco.new_job(input=x.urls, name="ddot_mapX", map_reader=x.mapReader, params=Params(n=n), map=_mapDot, nr_reduces=totalBlocks)
	resX = jobX.wait(clean=False, poll_interval=2)
	jobY = disco.new_job(input=y.urls, name="ddot_mapY", map_reader=y.mapReader, params=Params(n=n), map=_mapDot, nr_reduces=totalBlocks)
	resY = jobY.wait(clean=False, poll_interval=2)
	# dot product
	jobC = disco.new_job(input=resX+resY, name="ddot_reduce", map_reader=chain_reader, map=nop_map, reduce=_reduceDot, nr_reduces=totalBlocks)
	res = jobC.wait(clean=False, poll_interval=2)
	total = 0
	for k,v in result_iterator(res):
		total += float64(k)
	# clean up
	jobX.purge()
	jobY.purge()
	jobC.purge()
	return total 

def dnrm2(disco, n, x, maxTotalBlocks=128):
	"""
	Compute the euclidean norm of a vector in double precision.
	@param n Number of elements in x and y.
	@param x MatrixWrapper object encapsulating vector x either as a row or column vector.
	@param disco A Disco instance.
	@param maxTotalBlocks Suggested number of blocks to use for carrying out the computation. Ideally, this should equal to the number of cores available in the cluster. The actual number of blocks is selected based on the size of the vector 
	@return 2-norm of op(A) as a double precision scalar.
	"""
	from numpy import sqrt
	return sqrt(ddot(disco, n, x, x, maxTotalBlocks))

def dasum(disco, n, x, maxTotalBlocks=128):
	"""
	Compute sum of absolute values in double precision.
	@param n Number of elements in x.
	@param x MatrixWrapper object encapsulating vector x either as a row or column vector.
	@param disco A Disco instance.
	@param maxTotalBlocks Suggested number of blocks to use for carrying out the computation. Ideally, this should equal to the number of cores available in the cluster. The actual number of blocks is selected based on the size of the vector 
	@return The sum of absolute values as a double precision scalar.
	"""
	def _mapAsum(e, params):
		from numpy import float64
		if type(e) == tuple:
			e = e[0]
		total = 0 
		elems = e.split(";")
		for elem in elems:
			i, j, val = elem.split(",")
			i, j, val = int(i), int(j), abs(float64(val))
			# check that matrix is a row or column vector
			# this assertion is fast but not bullet-proof since it ignores a matrix with only nnz in first col and row
			assert i*j == 0, "element index (%d,%d) exceeds vector dimension" % (i,j)
			if i > 0:
				assert i < params.n, "row index %d exceeds matrix dimensions" % i
				total += val
			elif j > 0:
				assert j < params.n, "col index %d exceeds matrix dimensions" % j
				total += val
			else:
				total += val
		return [(total, "")]

	def _reduceAsum(iter, out, params):
		from numpy import float64
		total = 0
		for k, v in iter:
			total += float64(k)
		out.add(total, "")

	# map vectors
	totalBlocks = min(maxTotalBlocks, n)
	jobX = disco.new_job(input=x.urls, name="dasum", map_reader=x.mapReader, params=Params(n=n), map=_mapAsum, nr_reduces=totalBlocks, reduce=_reduceAsum)
	res = jobX.wait(clean=False, poll_interval=2)
	total = 0
	for k, v in result_iterator(res):
		total += float64(k)
	# clean up
	jobX.purge()
	return total 

def dgemscal(disco, transA, m, n, alpha, A, maxTotalBlocks=128):
	"""
	Compute general matrix scaling of alpha*op(A) where op(X) = X or transpose(X).
	While the same result can be achieved with dgemm or dgema, this streamlined function is faster.
	@param transA A boolean value for transposing matrix A or not.
	@param m Number of rows of matrix op(A).
	@param n Number of columns of matrix op(B).
	@param alpha Scalar multiplier for matrix A.
	@param A MatrixWrapper object encapsulating matrix A.
	@param disco A Disco instance.
	@param maxTotalBlocks Suggested number of matrix blocks to use for carrying out the addition. Ideally, this should equal to the number of cores available in the cluster. The actual number of blocks is selected based on the size of the matrix.
	@return MatrixWrapper object encapsulating the resulting matrix.
	"""
	def _mapElements(e, params):
		from numpy import float64
		if type(e) == tuple:
			e = e[0]
		output = []
		elems = e.split(";")
		for elem in elems:
			i, j, val = elem.split(",")
			if params.transA:
				i, j = j, i
			assert int(i) < params.m, "row index %d exceeds matrix dimensions" % int(i)
			assert int(j) < params.n, "col index %d exceeds matrix dimensions" % int(j)
			output += [("%s,%s,%.14f" % (i,j,params.alpha*float64(val)), "")]
		return output

	# map and scale matrix
	totalBlocks = min(m*n, maxTotalBlocks)
	res = disco.new_job(input=A.urls, name="dgemscal", map_reader=A.mapReader, map=_mapElements, params=Params(transA=transA, alpha=alpha, m=m, n=n)).wait(clean=False, poll_interval=2)
	return MatrixWrapper(res, chain_reader)

def dgemnrm1(disco, transA, m, n, A, maxTotalBlocks=128):
	"""
	Compute general matrix 1-norm (max column absolute sum) of op(A) where op(X) = X or transpose(X).
	@param transA A boolean value for transposing matrix A or not.
	@param m Number of rows of matrix op(A).
	@param n Number of columns of matrix op(B).
	@param A MatrixWrapper object encapsulating matrix A.
	@param disco A Disco instance.
	@param maxTotalBlocks Suggested number of matrix blocks to use for carrying out the addition. Ideally, this should equal to the number of cores available in the cluster. The actual number of blocks is selected based on the size of the matrix.
	@return 1-norm of op(A) as a double precision scalar.
	"""
	def _mapCols(e, params):
		if type(e) == tuple:
			e = e[0]
		output = []
		elems = e.split(";")
		for elem in elems:
			i, j, val = elem.split(",")
			if params.transA:
				i, j = j, i
			assert int(i) < params.m, "row index %d exceeds matrix dimensions" % int(i)
			assert int(j) < params.n, "col index %d exceeds matrix dimensions" % int(j)
			output += [(j, val)]
		return output

	def _reduceMaxColSums(iter, out, params):
		from numpy import float64
		s = {}
		# sum each col
		for colIdx, val in iter:
			s[colIdx] = s.get(colIdx, 0) + abs(float64(val))
		# output results
		if len(s) > 0:
			out.add("%.14f" % max(s.values()), "")

	# map matrix
	totalBlocks = min(maxTotalBlocks, n)
	jobA = disco.new_job(input=A.urls, name="dgemnrm1", map_reader=A.mapReader, map=_mapCols, params=Params(transA=transA, m=m, n=n), nr_reduces=totalBlocks, reduce=_reduceMaxColSums)
	res = jobA.wait(clean=False, poll_interval=2)
	# clean up
	jobA.purge()
	# return max col sum
	retVal = float64(0) 
	for k, v in result_iterator(res):
		retVal = max(retVal, float64(k))
	return retVal

def dgemsum(disco, transA, m, n, A, maxTotalBlocks=128):
	"""
	Compute general matrix column sum of op(A) in double precision where op(X) = X or transpose(X).
	While the same result can be achieved with dgemm, this streamlined function is faster.
	@param transA A boolean value for transposing matrix A or not.
	@param m Number of rows of matrix op(A).
	@param n Number of columns of matrix op(B).
	@param A MatrixWrapper object encapsulating matrix A.
	@param disco A Disco instance.
	@param maxTotalBlocks Suggested number of matrix blocks to use for carrying out the addition. Ideally, this should equal to the number of cores available in the cluster. The actual number of blocks is selected based on the size of the matrix.
	@return MatrixWrapper object encapsulating the resulting 1xn matrix.
	"""
	def _mapCols(e, params):
		if type(e) == tuple:
			e = e[0]
		output = []
		elems = e.split(";")
		for elem in elems:
			i, j, val = elem.split(",")
			if params.transA:
				i, j = j, i
			assert int(i) < params.m, "row index %d exceeds matrix dimensions" % int(i)
			assert int(j) < params.n, "col index %d exceeds matrix dimensions" % int(j)
			output += [(j, val)]
		return output

	def _reduceSumCols(iter, out, params):
		from numpy import float64
		s = {}
		# sum each col
		for colIdx, val in iter:
			s[colIdx] = s.get(colIdx, 0) + float64(val)
		# output results
		for colIdx, total in s.items():
			out.add("0,%s,%.14f" % (colIdx, total), "")

	# map matrix
	totalBlocks = min(maxTotalBlocks, n)
	res = disco.new_job(input=A.urls, name="dgemsum", map_reader=A.mapReader, map=_mapCols, params=Params(transA=transA, m=m, n=n), nr_reduces=totalBlocks, reduce=_reduceSumCols).wait(clean=False, poll_interval=2)
	return MatrixWrapper(res, chain_reader)

def dgema(disco, transA, transB, m, n, alpha, A, B, beta, maxTotalBlocks=128):
	"""
	Compute general matrix addition alpha*op(A) + beta*op(B) in double precision where op(X) = X or transpose(X).
	@param transA A boolean value for transposing matrix A or not.
	@param transB A boolean value for transposing matrix B or not.
	@param m Number of rows of matrix op(A).
	@param n Number of columns of matrix op(B).
	@param alpha Scalar multiplier for matrix A.
	@param beta Scalar multiplier for matrix B.
	@param A MatrixWrapper object encapsulating matrix A.
	@param B MatrixWrapper object encapsulating matrix B.
	@param disco A Disco instance.
	@param maxTotalBlocks Suggested number of matrix blocks to use for carrying out the addition. Ideally, this should equal to the number of cores available in the cluster. The actual number of blocks is selected based on the size of the matrix.
	@return MatrixWrapper object encapsulating the resulting matrix.
	"""
	def _mapBlocks(e, params):
		from math import ceil
		from numpy import float64
		if type(e) == tuple:
			e = e[0]
		output = []
		elems = e.split(";")
		for elem in elems:
			i, j, val = map(float64, elem.split(","))
			if params.transpose:
				i, j = j, i
			assert i < params.m, "row index %d exceeds matrix dimensions" % int(i)
			assert j < params.n, "col index %d exceeds matrix dimensions" % int(j)
			blockX = int(j / params.blockWidth)
			blockY = int(i / params.blockHeight)
			offsetX = ceil(params.blockWidth * blockX)
			offsetY = ceil(params.blockHeight * blockY)
			val = params.scaling * val
			if val != 0.0:
				output += [(blockY*params.blocksPerRow+blockX, "%d,%d,%.14f" % (int(i-offsetY), int(j-offsetX), val))]
		return output

	def nop_map(e, params):
		return [e]

	def _reduceAddBlocks(iter, out, params):
		from numpy import float64
		s = {}
		# add matrices
		for blockId, t in iter:
			blockId = int(blockId)
			rowIdx, colIdx, val = t.split(",")
			rowIdx = int(rowIdx)
			colIdx = int(colIdx)
			if not s.has_key(blockId):
				s[blockId] = {}
			if not s[blockId].has_key(rowIdx):
				s[blockId][rowIdx] = {}
			s[blockId][rowIdx][colIdx] = s[blockId][rowIdx].get(colIdx, 0) + float64(val)
		# output results
		from math import ceil
		from scipy.sparse import coo_matrix
		for blockId in s.keys():
			# compute the index offset in the original matrix
			offsetY = ceil(params.blockHeight * (blockId / params.blocksPerRow))
			offsetX = ceil(params.blockWidth * (blockId % params.blocksPerRow))
			# map block indices into original indices
			for rowIdx in s[blockId].keys():
				for colIdx in s[blockId][rowIdx].keys():
					out.add("%d,%d,%.14f" % (rowIdx+offsetY, colIdx+offsetX, s[blockId][rowIdx][colIdx]), "")

	# find the best way to partition matrix to blocks
	blocksPerRow, blocksPerCol = _partition(m, n, maxTotalBlocks)
	blockHeight = float(m) / blocksPerCol
	blockWidth = float(n) / blocksPerRow
	totalBlocks = blocksPerRow * blocksPerCol
	# map and scale matrices
	params = Params(blocksPerRow=blocksPerRow, blocksPerCol=blocksPerCol, blockHeight=blockHeight, blockWidth=blockWidth)
	params.transpose = transA
	params.scaling = alpha
	params.m = m
	params.n = n
	jobMapA = disco.new_job(input=A.urls, name="dgema_mapA", map_reader=A.mapReader, map=_mapBlocks, params=params, nr_reduces=totalBlocks)
	resA = jobMapA.wait(clean=False, poll_interval=2)
	params.transpose = transB
	params.scaling = beta
	jobMapB = disco.new_job(input=B.urls, name="dgema_mapB", map_reader=B.mapReader, map=_mapBlocks, params=params, nr_reduces=totalBlocks)
	resB = jobMapB.wait(clean=False, poll_interval=2)
	# add matrices
	res = disco.new_job(input=resA+resB, name="dgema_reduce", map_reader=chain_reader, map=nop_map, params=params, reduce=_reduceAddBlocks, nr_reduces=totalBlocks).wait(clean=False, poll_interval=2)
	# clean up
	jobMapA.purge()
	jobMapB.purge()
	return MatrixWrapper(res, chain_reader)

def dgemm(disco, transA, transB, m, n, k, alpha, A, B, beta, C, maxTotalBlocks=128):
	"""
	Compute general matrix multiplication alpha*op(A)*op(B) + beta*C in double precision where op(X) = X or transpose(X).
	@param transA A boolean value for transposing matrix A or not.
	@param transB A boolean value for transposing matrix B or not.
	@param m Number of rows of matrix op(A) and C.
	@param n Number of columns of matrix op(B) and C.
	@param k Number of columns of matrix op(A) and rows of matrix op(B).
	@param alpha Scalar multiplier for the matrix product A*B.
	@param beta Scalar multiplier for matrix C.
	@param A MatrixWrapper object encapsulating matrix A.
	@param B MatrixWrapper object encapsulating matrix B.
	@param C MatrixWrapper object encapsulating matrix C. If there is no C term, then pass in an empty wrapper, MatrixWrapper(), as placeholder.
	@param disco A Disco instance.
	@param maxTotalBlocks Suggested number of matrix blocks to use for carrying out the multiplication. Ideally, this should equal to the number of cores available in the cluster. The actual number of blocks is selected based on the size of the matrix.
	@return MatrixWrapper object encapsulating the resulting matrix.
	"""
	def _mapRowBlocks(e, params):
		from math import ceil
		from numpy import float64
		if type(e) == tuple:
			e = e[0]
		output = []
		elems = e.split(";")
		for elem in elems:
			i, j, val = map(float64, elem.split(","))
			if params.transA:
				i, j = j, i
			assert i < params.m, "row index %d exceeds matrix dimensions" % int(i)
			assert j < params.k, "col index %d exceeds matrix dimensions" % int(j)
			blockX = int(j / params.blockWidth)
			blockY = int(i / params.blockHeight)
			offsetY = ceil(params.blockHeight * blockY)
			val = params.alpha * val
			if val != 0.0:
				output += [(blockY*params.blocksPerRow+x, "%s,%d,%d,%.14f" % (params.matrixId, int(i-offsetY), int(j), val)) for x in range(0, params.blocksPerRow)]
		return output
		
	def _mapColBlocks(e, params):
		from math import ceil
		from numpy import float64
		if type(e) == tuple:
			e = e[0]
		output = []
		elems = e.split(";")
		for elem in elems:
			i, j, val = map(float64, elem.split(","))
			if params.transB:
				i, j = j, i
			assert i < params.k, "row index %d exceeds matrix dimensions" % int(i)
			assert j < params.n, "col index %d exceeds matrix dimensions" % int(j)
			blockX = int(j / params.blockWidth)
			blockX = int(j / params.blockWidth)
			offsetX = ceil(params.blockWidth * blockX)
			if val != 0.0:
				output += [(y*params.blocksPerRow+blockX, "%s,%d,%d,%.14f" % (params.matrixId, int(i), int(j-offsetX), val)) for y in range(0, params.blocksPerCol)]
		return output
		
	def _mapBlocks(e, params):
		from math import ceil
		from numpy import float64
		if type(e) == tuple:
			e = e[0]
		output = []
		elems = e.split(";")
		for elem in elems:
			i, j, val = map(float64, elem.split(","))
			assert i < params.m, "row index %d exceeds matrix dimensions" % int(i)
			assert j < params.n, "col index %d exceeds matrix dimensions" % int(j)
			blockX = int(j / params.blockWidth)
			blockX = int(j / params.blockWidth)
			blockY = int(i / params.blockHeight)
			offsetX = ceil(params.blockWidth * blockX)
			offsetY = ceil(params.blockHeight * blockY)
			val = params.beta*val
			if val != 0.0:
				output += [(blockY*params.blocksPerRow+blockX, "%s,%d,%d,%.14f" % (params.matrixId, int(i-offsetY), int(j-offsetX), val))]
		return output

	def nop_map(e, params):
		return [e]

	def _reduceMultiplyAndAdd(iter, out, params):
		from numpy import float64
		rows = {}
		cols = {}
		vals = {}
		maxColIdx = {}
		maxRowIdx = {}
		for blockId, s in iter:
			blockId = int(blockId)
			matrixId, rowIdx, colIdx, val = s.split(",")
			rowIdx = int(rowIdx)
			colIdx = int(colIdx)
			val = float64(val)
			if not rows.has_key(blockId):
				rows[blockId] = {}
				cols[blockId] = {}
				vals[blockId] = {}
				maxColIdx[blockId] = {}
				maxRowIdx[blockId] = {}
			if not rows[blockId].has_key(matrixId):
				rows[blockId][matrixId] = []
				cols[blockId][matrixId] = []
				vals[blockId][matrixId] = []
				maxColIdx[blockId][matrixId] = 0
				maxRowIdx[blockId][matrixId] = 0
			rows[blockId][matrixId].append(rowIdx)
			cols[blockId][matrixId].append(colIdx)
			vals[blockId][matrixId].append(val)
			maxColIdx[blockId][matrixId] = max(maxColIdx[blockId][matrixId], cols[blockId][matrixId][-1])
			maxRowIdx[blockId][matrixId] = max(maxRowIdx[blockId][matrixId], rows[blockId][matrixId][-1])
		# initialize sparse matrices
		from math import ceil
		from scipy.sparse import coo_matrix
		for blockId in rows.keys():
			# compute the index offset in the original matrix
			blockY = blockId / params.blocksPerRow
			blockX = blockId % params.blocksPerRow
			offsetY = ceil(params.blockHeight * blockY)
			offsetX = ceil(params.blockWidth * blockX)
			# compute matrix product
			if not vals[blockId].has_key('A') or not vals[blockId].has_key('B'):
				# skip multiplication since either block A or B is empty
				if vals[blockId].has_key('C'):
					# return beta*C
					P = coo_matrix((vals[blockId]['C'],(rows[blockId]['C'],cols[blockId]['C'])), dtype=float64, dims=(maxRowIdx[blockId]['C']+1, maxColIdx[blockId]['C']+1))
				else:
					P = None
			else:
				if vals[blockId].has_key('C'):
					m = max(maxRowIdx[blockId]['A'], maxRowIdx[blockId]['C']) + 1
					n = max(maxColIdx[blockId]['B'], maxColIdx[blockId]['C']) + 1
					C = coo_matrix((vals[blockId]['C'],(rows[blockId]['C'],cols[blockId]['C'])), dtype=float64, dims=(m,n))
				else:
					m = maxRowIdx[blockId]['A'] + 1
					n = maxColIdx[blockId]['B'] + 1
					C = coo_matrix(([],([],[])), dtype=float64, dims=(m,n))
				A = coo_matrix((vals[blockId]['A'],(rows[blockId]['A'],cols[blockId]['A'])), dtype=float64, dims=(m,max(maxColIdx[blockId]['A'], maxRowIdx[blockId]['B'])+1))
				B = coo_matrix((vals[blockId]['B'],(rows[blockId]['B'],cols[blockId]['B'])), dtype=float64, dims=(max(maxColIdx[blockId]['A'], maxRowIdx[blockId]['B'])+1, n))
				P = (A * B + C).tocoo()
			# map block indices into original indices
			if P != None:
				start = 0
				while start < len(P.row):
					end = min(start+params.elemsPerLine, len(P.row))
					out.add(";".join(["%d,%d,%.14f" % (P.row[i]+offsetY, P.col[i]+offsetX, P.data[i]) for i in range(start,end)]), "")
					start = end

	# find the best way to partition matrix into blocks
	blocksPerRow, blocksPerCol = _partition(m, n, maxTotalBlocks)
	blockHeight = float(m) / blocksPerCol
	blockWidth = float(n) / blocksPerRow
	totalBlocks = blocksPerRow * blocksPerCol
	#print "%dx%d blocks used with block dimension %fx%f" % (blocksPerCol, blocksPerRow, blockHeight, blockWidth)
	params = Params(blocksPerRow=blocksPerRow, blocksPerCol=blocksPerCol, blockHeight=blockHeight, blockWidth=blockWidth, alpha=alpha, beta=beta, transA=transA, transB=transB, m=m, k=k, n=n)
	params.elemsPerLine = 1000
	# map matrix A into row blocks
	params.matrixId = 'A'
	jobMapA = disco.new_job(input=A.urls, name="dgemm_mapA", map_reader=A.mapReader, map=_mapRowBlocks, params=params, nr_reduces=totalBlocks)
	resA = jobMapA.wait(clean=False, poll_interval=2)
	# map matrix B into col blocks
	params.matrixId = 'B'
	jobMapB = disco.new_job(input=B.urls, name="dgemm_mapB", map_reader=B.mapReader, map=_mapColBlocks, params=params, nr_reduces=totalBlocks)
	resB = jobMapB.wait(clean=False, poll_interval=2)
	# map matrix C into blocks
	if len(C.urls) == 0: # quick fix for disco bug
		resC = []
	else:
		params.matrixId = 'C'
		jobMapC = disco.new_job(input=C.urls, name="dgemm_mapC", map_reader=C.mapReader, map=_mapBlocks, params=params, nr_reduces=totalBlocks)
		resC = jobMapC.wait(clean=False, poll_interval=2)
	# multiply the blocks
	res = disco.new_job(input=resA+resB+resC, name="dgemm_reduce", map_reader=chain_reader, map=nop_map, nr_reduces=totalBlocks, reduce=_reduceMultiplyAndAdd, params=params).wait(clean=False, poll_interval=2)
	# clean up
	jobMapA.purge()
	jobMapB.purge()
	if len(C.urls) > 0: # quick fix for disco bug
		jobMapC.purge()
	return MatrixWrapper(res, chain_reader)
