
def triu(disco, m, n, A, k, cores=128):
	"""
	Returns the element on and above the kth diagonal of A. k = 0 is the main diagonal, k > 0 is above the main diagonal, and k < 0 is below the main diagonal.  This function uses replace.
	@param m Number of rows of matrix A.
	@param n Number of column of matrix A.
	@param A MatrixWrapper object encapsulating matrix A.
	@param The k-th diagonal where k=0 is the main diagonal, k>0 is above the main diagonal and k<0 is below the main diagonal.
	@param cores Number of partitions for the resulting matrix.
	@return MatrixWrapper object encapsulating the resulting matrix.
	"""
	f = lambda args: args['v'] if args['i'] <= args['j'] - args['k'] else 0
	return replace(disco, m, n, A, f, {"k":k}, cores)

def tril(disco, m, n, A, k, cores=128):
	"""
	Returns the element on and below the kth diagonal of A. k = 0 is the main diagonal, k > 0 is above the main diagonal, and k < 0 is below the main diagonal.  This function uses replace.
	@param m Number of rows of matrix A.
	@param n Number of column of matrix A.
	@param A MatrixWrapper object encapsulating matrix A.
	@param The k-th diagonal where k=0 is the main diagonal, k>0 is above the main diagonal and k<0 is below the main diagonal.
	@param cores Number of partitions for the resulting matrix.
	@return MatrixWrapper object encapsulating the resulting matrix.
	"""
	f = lambda args: args['v'] if args['i'] >= args['j'] - args['k'] else 0
	return replace(disco, m, n, A, f, {"k":k}, cores)

def replace(disco, m, n, A, f, args={}, cores=128):
	"""
	Replace elements in matrix A with the values returned by evaluating the given function. 
	@param m Number of rows of matrix op(A).
	@param n Number of columns of matrix op(A).
	@param A MatrixWrapper object encapsulating matrix A.
	@param f A function that accepts a dictionary with predefined values for int i, int j and float64 v, where A[i,j]=v, and returns a double precision floating number. Entries from parameter args are also added to the dictionary.
	@param args Dictionary containing entries to be passed to function f.
	@param cores Suggested number of partitions for the resulting matrix.
	@return A copy of matrix A with elements replaced by the return values of evaluatng the given function.
	"""
	def _mapReplace(e, params):
		from numpy import float64
		output = []
		if type(e) == tuple:
			e = e[0]
		elems = e.split(";")
		for elem in elems:
			i, j, val = elem.split(",")
			i, j = int(i), int(j)
			assert i < params.m, "row index %d exceeds matrix dimensions" % i
			assert j < params.n, "col index %d exceeds matrix dimensions" % j
			params.args["i"] = i
			params.args["j"] = j
			params.args["v"] = float64(val)
			val = params.f(params.args)
			if val != 0:
				output.append("%d,%d,%.14f" % (i,j,val))
		if len(output) > 0:
			return [(";".join(output), "")]
		return output 

	from disco.core import result_iterator, Params
	from disco.func import chain_reader
	from matrixWrap import MatrixWrapper
	totalBlocks = min(max(m, n), cores)
	res = disco.new_job(input=A.urls, name="replace", params=Params(m=m, n=n, f=f, args=args), map_reader=A.mapReader, map=_mapReplace, nr_reduces=totalBlocks).wait(clean=False)
	return MatrixWrapper(res, chain_reader)

def minDim(disco, A):
	"""
	Return the minimum dimension of matrix A based on its maximum row and column index.
	The dimension returned is essentially the maximum row and column index plus one unless the matrix is empty, in which case, (0,0) is returned.
	@param A MatrixWrapper object encapsulating matrix A.
	@return A dimension tuple.
	"""
	def _mapMinDim(e, params):
		maxRowIdx = 0
		maxColIdx = 0
		if type(e) == tuple:
			e = e[0]
		elems = e.split(";")
		for elem in elems:
			i, j, val = elem.split(",")
			i, j = int(i), int(j)
			maxRowIdx = max(maxRowIdx, i)
			maxColIdx = max(maxColIdx, j)
		return [("m", maxRowIdx+1), ("n", maxColIdx+1)]

	def _reduceMinDim(iter, out, params):
		s = {}
		for k, v in iter:
			s[k] = max(s.get(k, 0), int(v))
		for k, v in s.items():
			out.add(k, v)

	from disco.core import result_iterator
	from disco.func import chain_reader
	from matrixWrap import MatrixWrapper
	jobA = disco.new_job(input=A.urls, name="minDim", map_reader=A.mapReader, map=_mapMinDim, nr_reduces=2, reduce=_reduceMinDim)
	res = jobA.wait(clean=False)
	m, n = 0, 0
	for k, v in result_iterator(res):
		if k == "m":
			m = int(v)
		elif k == "n":
			n = int(v)
	# clean up
	jobA.purge()
	return (m, n)

def nnz(disco, m, n, A):
	"""
	Return the number of nonzero matrix elements.
	@param m Number of rows of matrix op(A).
	@param n Number of columns of matrix op(A).
	@param A MatrixWrapper object encapsulating matrix A.
	@return An integer.
	"""
	def _mapNnz(e, params):
		nnz = 0
		if type(e) == tuple:
			e = e[0]
		elems = e.split(";")
		for elem in elems:
			i, j, val = elem.split(",")
			assert int(i) < params.m, "row index %d exceeds matrix dimensions" % int(i)
			assert int(j) < params.n, "col index %d exceeds matrix dimensions" % int(j)
			nnz += 1
		return [("nnz", nnz)]

	from disco.core import Params, result_iterator
	from disco.func import chain_reader
	from matrixWrap import MatrixWrapper
	jobA = disco.new_job(input=A.urls, name="nnz", map_reader=A.mapReader, params=Params(m=m, n=n), map=_mapNnz)
	res = jobA.wait(clean=False)
	nnz = 0
	for k, v in result_iterator(res):
		nnz += int(v)
	# clean up
	jobA.purge()
	return nnz 

def crop(disco, transA, m, n, A, minRowId, minColId, maxRowId, maxColId, cores=128):
	"""
	Crop a portion of the matrix op(A) leaving the remaining portion set to zeros where op(X) = X or transpose(X).
	@param transA A boolean value for transposing matrix A or not.
	@param m Number of rows of matrix op(A).
	@param n Number of columns of matrix op(A).
	@param A MatrixWrapper object encapsulating matrix A.
	@param minRowId The minimum row index of matrix op(A), inclusive.
	@param minColId The minimum column index of matrix op(A), inclusive.
	@param maxRowId The maximum row index of matrix op(A), inclusive.
	@param maxColId The maximum col index of matrix op(A), inclusive.
	@param cores Suggested number of partitions for the resulting matrix.
	@return MatrixWrapper object encapsulating the resulting matrix.
	"""
	def _mapCrop(e, params):
		output = []
		if type(e) == tuple:
			e = e[0]
		elems = e.split(";")
		for elem in elems:
			i, j, val = elem.split(",")
			if params.transA:
				i, j = j, i
			i, j = int(i), int(j)
			assert i < params.m, "row index %d exceeds matrix dimensions" % i
			assert j < params.n, "col index %d exceeds matrix dimensions" % j
			if i >= params.minRowId and i <= params.maxRowId and j >= params.minColId and j <= params.maxColId:
				output.append("%d,%d,%s" % (i,j,val))
		if len(output) > 0:
			return [(";".join(output), "")]
		return output 

	from disco.core import Params
	from disco.func import chain_reader
	from matrixWrap import MatrixWrapper
	totalBlocks = min(max(m, n), cores)
	params = Params(transA=transA, minRowId=minRowId, minColId=minColId, maxRowId=maxRowId, maxColId=maxColId, m=m, n=n)
	res = disco.new_job(input=A.urls, name="crop", map_reader=A.mapReader, map=_mapCrop, params=params, nr_reduces=totalBlocks).wait(clean=False)
	return MatrixWrapper(res, chain_reader)

def eye(disco, m, n, cores=128):
	"""
	Generate an m-by-n matrix with 1's on the diagonal and zeros elsewhere.
	@param m Number of rows of the generated matrix.
	@param n Number of columns of the generated matrix.
	@param cores Number of cores used for generating the matrix and the number of partitions for the resulting matrix.
	@return MatrixWrapper object encapsulating the resulting matrix.
	"""
	def _mapEye(e, params):
		retVal = []
		if params.n > 0 and params.m > 0:
			elems = e.split(",")
			for elem in elems:
				i = int(elem)
				retVal.append("%d,%d,1" % (i,i))
			retVal = [(";".join(retVal), "")]
		return retVal

	from disco.core import Params
	from disco.func import chain_reader
	from matrixWrap import MatrixWrapper
	totalBlocks = min(m, n, cores)
	# generate input seeds
	indices = range(0, min(m, n))
	input = []
	size = int(min(m, n) / totalBlocks)
	k = 0
	while k < min(m, n):
		input.append("raw://" + ",".join([str(indices[i]) for i in range(k, min(k+size, min(m, n)))]))
		k += size
	res = disco.new_job(input=input, name="eye", map=_mapEye, params=Params(m=m, n=n), nr_reduces=totalBlocks).wait(clean=False)
	return MatrixWrapper(res, chain_reader)

def zeros(disco, m, n):
	"""
	Generate an m-by-n matrix with all zeros. 
	@param m Number of rows of the generated matrix.
	@param n Number of columns of the generated matrix.
	@return MatrixWrapper object encapsulating the resulting matrix.
	"""
	from matrixWrap import MatrixWrapper
	from disco.func import map_line_reader
	return MatrixWrapper([], map_line_reader)

def ones(disco, m, n, cores=128):
	"""
	Generate an m-by-n matrix with all ones.
	@param m Number of rows of the generated matrix.
	@param n Number of columns of the generated matrix.
	@param cores Suggested number of cores used for generating the matrix and the number of partitions for the resulting matrix.
	@return MatrixWrapper object encapsulating the resulting matrix.
	"""
	def _mapRows(e, params):
		from disco.core import Params
		m, n = params.m, params.n
		output = []
		if m > 0:
			elems = e.split(",")
			l = range(0, n)
			for elem in elems:
				retVal = []
				i = int(elem)
				for j in range(0, n):
					retVal.append("%d,%d,1" % (i,j))
					# break output into tuples so reduce can distribute the load
					if len(retVal) > 1000:
						output += [(";".join(retVal), "")]
						retVal = []
				if len(retVal) > 0:
					output += [(";".join(retVal), "")]
		return output 

	def _mapCols(e, params):
		from disco.core import Params
		m, n = params.m, params.n
		output = []
		if n > 0:
			elems = e.split(",")
			l = range(0, m)
			for elem in elems:
				retVal = []
				j = int(elem)
				for i in range(0, m):
					retVal.append("%d,%d,1" % (i,j))
					# break output into tuples so reduce can distribute the load
					if len(retVal) > 1000:
						output += [(";".join(retVal), "")]
						retVal = []
				if len(retVal) > 0:
					output += [(";".join(retVal), "")]
		return output

	from matrixWrap import MatrixWrapper
	from disco.func import chain_reader
	if m>0 and n>0:
		from disco.core import Params
		from disco.func import nop_reduce
		totalBlocks = min(max(m, n), cores)
		# generate input seeds
		indices = range(0, min(m, n))
		input = []
		size = int(max(1, len(indices) / totalBlocks))
		k = 0
		while k<len(indices):
			input.append("raw://" + ",".join([str(indices[i]) for i in range(k, min(k+size, len(indices)))]))
			k += size
		# reduce to distribute the map keys evenly across cluster.
		# this is needed when min(m,n) << totalBlocks like in the case of creating a random vector.
		if m < n:
			res = disco.new_job(input=input, name="ones", map=_mapRows, params=Params(m=m, n=n), reduce=nop_reduce, nr_reduces=totalBlocks).wait(clean=False)
		else:
			res = disco.new_job(input=input, name="ones", map=_mapCols, params=Params(m=m, n=n), reduce=nop_reduce, nr_reduces=totalBlocks).wait(clean=False)
		return MatrixWrapper(res, chain_reader)
	else:
		return MatrixWrapper([], chain_reader)

def diag(disco, m, n, A, k=0, cores=128):
	"""
	If A is a vector, then construct a matrix with the k-th diagonal being the elements of A.
	Otherwise, construct a column vector (stored as a matrix with column size 1) from the k-th diagonal of A.
	@param m Number of rows of matrix A.
	@param n Number of column of matrix A.
	@param A MatrixWrapper object encapsulating matrix A.
	@param The k-th diagonal where k=0 is the main diagonal, k>0 is above the main diagonal and k<0 is below the main diagonal.
	@param cores Number of partitions for the resulting matrix.
	@return MatrixWrapper object encapsulating the resulting matrix.
	"""
	def _mapDeflatDiag(e, params):
		if type(e) == tuple:
			e = e[0]
		k = int(params.k)
		output = []
		elems = e.split(";")
		for elem in elems:
			i, j, val = elem.split(",")
			i, j = int(i), int(j)
			assert i < params.m, "row index %d exceeds matrix dimensions" % i
			assert j < params.n, "col index %d exceeds matrix dimensions" % j
			if j-k == i:
				if k > 0:
					output.append(("%d,%d,%s" % (j-k,0,val), ""))
				else:
					output.append(("%d,%d,%s" % (i+k,0,val), ""))
		return output

	def _mapInflatDiag(e, params):
		if type(e) == tuple:
			e = e[0]
		k = int(params.k)
		output = []
		elems = e.split(";")
		for elem in elems:
			i, j, val = elem.split(",")
			i, j = int(i), int(j)
			assert i < params.m, "row index %d exceeds matrix dimensions" % i
			assert j < params.n, "col index %d exceeds matrix dimensions" % j
			if params.m > params.n:
				if k > 0:
					output.append(("%d,%d,%s" % (i,i+k,val), ""))
				else:
					output.append(("%d,%d,%s" % (i+abs(k),i,val), ""))
			else:
				if k > 0:
					output.append(("%d,%d,%s" % (j,j+k,val), ""))
				else:
					output.append(("%d,%d,%s" % (j+abs(k),j,val), ""))
		return output

	from disco.func import chain_reader
	from matrixWrap import MatrixWrapper
	from disco.core import Params
	params = Params(m=m, n=n, k=k)
	if m == 1 or n == 1:
		res = disco.new_job(input=A.urls, name="diag", map_reader=A.mapReader, map=_mapInflatDiag, params=params, nr_reduces=cores).wait(clean=False)
	else:
		res = disco.new_job(input=A.urls, name="diag", map_reader=A.mapReader, map=_mapDeflatDiag, params=params, nr_reduces=cores).wait(clean=False)
	return MatrixWrapper(res, chain_reader)

def randSym(disco, n, sparsity, lb=0.0, ub=1.0, cores=128):
	"""
	Generate a n-by-n symmetric matrix containing pseudo-random values drawn from an uniform distribution in double precision.
	The number of non-zero elements is approximately n*n*sparsity.
	@param n Number of rows and columns in matrix
	@param sparsity Density of the matrix in the range of [0-1] with 0=dense matrix.
	@param lb The lower bound of the interval to draw from, inclusive. Default is 0.0.
	@param ub The upper bound of the interval to draw from, exclusive. Default is 1.0.
	@param disco Disco instance
	@param cores Number of cores used for generating the matrix and the number of partitions for the resulting matrix.
	@return MatrixWrapper object
	"""
	def _mapRows(e, params):
		from numpy import random
		from disco.core import Params
		n = params.n
		retVal = []
		if n > 0:
			elems = e.split(",")
			l = range(0, n)
			random.shuffle(l)
			for elem in elems:
				i = int(elem)
				nnz = n * (1.0 - params.sparsity)
				stepSize = int(n / nnz)
				k = int(random.random() * (n % nnz))
				while k<n:
					j = l[k]
					k += stepSize
					if j >= i:
						val = params.lb + (params.ub-params.lb) * random.random()
						retVal.append("%d,%d,%.14f" % (i,j,val))
						if j != i:
							retVal.append("%d,%d,%.14f" % (j,i,val))
			retVal = [(";".join(retVal), "")]
		return retVal

	from matrixWrap import MatrixWrapper
	from disco.func import chain_reader, nop_reduce
	if n > 0:
		from disco.core import Params
		from numpy import random
		totalBlocks = min(n, cores)
		# generate input seeds
		rowIndices = range(0, n)
		random.shuffle(rowIndices)
		input = []
		size = int(n / totalBlocks)
		k = 0
		while k<n:
			input.append("raw://" + ",".join([str(rowIndices[i]) for i in range(k, min(k+size, n))]))
			k += size
		res = disco.new_job(input=input, name="randSym", map=_mapRows, params=Params(n=n, sparsity=sparsity, lb=lb, ub=ub), nr_reduces=totalBlocks, reduce=nop_reduce).wait(clean=False)
		return MatrixWrapper(res, chain_reader)
	else:
		return MatrixWrapper([], chain_reader)

def rand(disco, m, n, sparsity, lb=0.0, ub=1.0, cores=128):
	"""
	Generate a m-by-n symmetric matrix containing pseudo-random values drawn from an uniform distribution in double precision.
	The number of non-zero elements is approximately m*n*sparsity.
	@param m number of rows in matrix
	@param n number of columns in matrix
	@param sparsity Density of the matrix in the range of [0-1] with 0=dense matrix.
	@param lb The lower bound of the interval to draw from, inclusive. Default is 0.0.
	@param ub The upper bound of the interval to draw from, exclusive. Default is 1.0.
	@param disco Disco instance
	@param cores Number of cores used for generating the matrix and the number of partitions for the resulting matrix.
	@return MatrixWrapper object
	"""
	def _mapRows(e, params):
		from numpy import random
		from disco.core import Params
		m, n = params.m, params.n
		output = []
		if m > 0:
			elems = e.split(",")
			l = range(0, n)
			random.shuffle(l)
			for elem in elems:
				retVal = []
				i = int(elem)
				nnz = n * (1.0 - params.sparsity)
				stepSize = int(n / nnz)
				k = int(random.random() * (n % nnz))
				while k<n:
					j = l[k]
					k += stepSize
					val = params.lb + (params.ub-params.lb) * random.random()
					retVal.append("%d,%d,%.14f" % (i,j,val))
					# break output into tuples so reduce can distribute the load
					if len(retVal) > 1000:
						output += [(";".join(retVal), "")]
						retVal = []
				if len(retVal) > 0:
					output += [(";".join(retVal), "")]
		return output 

	def _mapCols(e, params):
		from numpy import random
		from disco.core import Params
		m, n = params.m, params.n
		output = []
		if n > 0:
			elems = e.split(",")
			l = range(0, m)
			random.shuffle(l)
			for elem in elems:
				retVal = []
				j = int(elem)
				nnz = m * (1.0 - params.sparsity)
				stepSize = int(m / nnz)
				k = int(random.random() * (m % nnz))
				while k<m:
					i = l[k]
					k += stepSize
					val = params.lb + (params.ub-params.lb) * random.random()
					retVal.append("%d,%d,%.14f" % (i,j,val))
					# break output into tuples so reduce can distribute the load
					if len(retVal) > 1000:
						output += [(";".join(retVal), "")]
						retVal = []
				if len(retVal) > 0:
					output += [(";".join(retVal), "")]
		return output

	from matrixWrap import MatrixWrapper
	from disco.func import chain_reader
	if m>0 and n>0:
		from disco.core import Params
		from disco.func import nop_reduce
		from numpy import random
		totalBlocks = min(max(m, n), cores)
		# generate input seeds
		indices = range(0, min(m, n))
		random.shuffle(indices)
		input = []
		size = int(max(1, len(indices) / totalBlocks))
		k = 0
		while k<len(indices):
			input.append("raw://" + ",".join([str(indices[i]) for i in range(k, min(k+size, len(indices)))]))
			k += size
		# reduce to distribute the map keys evenly across cluster.
		# this is needed when min(m,n) << totalBlocks like in the case of creating a random vector.
		if m < n:
			res = disco.new_job(input=input, name="rand", map=_mapRows, params=Params(m=m, n=n, sparsity=sparsity, lb=lb, ub=ub), reduce=nop_reduce, nr_reduces=totalBlocks).wait(clean=False)
		else:
			res = disco.new_job(input=input, name="rand", map=_mapCols, params=Params(m=m, n=n, sparsity=sparsity, lb=lb, ub=ub), reduce=nop_reduce, nr_reduces=totalBlocks).wait(clean=False)
		return MatrixWrapper(res, chain_reader)
	else:
		return MatrixWrapper([], chain_reader)

