//https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/discuss/85201/c-on-time-on-space-solution-with-detail-intuitive-explanation

Preparations
------------------
//     When n==1 (i.e. the matrix is 1x1. n is the number of row), the problem is trival. 
//     Hencefore we only consider the case n>=2.
      
//     Rather than finding one k-th element from the matrix, we will select TWO elements 
//     (say, k0-th element and k1-th element) simultaneously, such that 0<=k0<=k1<n*n and k1-k0<=4n. 
//     Obviously, if we can complete the aforementioned selection in O(n), we can find the k-th 
//     element in O(n) by simply letting k=k0=k1.
      
//     Let x0 denote the k0-th element; let x1 denote the k1-th element. Obviously we have x0<=x1.

// Now we will introduce how to select x0 and x1 in O(n).

General idea:
----------------
// For an nxn matrix, where n is large, we try to select x0 and x1 in a recursive way.

//     (Determine submatrix) This step constructs one submatrix, whose number of elements will be 
//     approximately a quarter of the original matrix. The submatrix is defined as every other 
//     row and every other column of the original matrix. The last row and the last column are 
//     included too (the reason will be stated in the sequel.) Then the dimension of the matrix is 
//     approximately (n/2) x (n/2). The submatrix is recorded by the indices in the original matrix.
  
//     Example 1: the original matrix has indices {0, 1, 2, 3, 4}, then the submatrix has indices {0, 2, 4}.
//     Example 2: the original matrix has indices {0,1, 2, 3, 4, 5}, then the submatrix has indices {0, 2,4, 5}.

//     (Determine new k's) This step determines two new k's (denoted as k0_ and k1_) such that (i) k0_ is the 
//     largest possible integer to ensure k0_-th element in the new submatrix (denoted as x0_) is not 
//      greater than x0; (ii) k1_ is the smallest possible integer to ensure k1_-th element in the 
//      new submatrix (denoted as x1_) is not less than x1. This step is the most tricky step.

//      k0_ = floor(k0 / 4)
//      k1_ = floor(k1 / 4) + n + 1 (when n is even)
//           floor((k1 + 2 * n + 1) / 4) (when n is odd)



// Recall that we mentioned the last row and column shall always be included in the matrix. 
//  That is to ensure we can always found the x1_ such that x1_ >= x1.
// 3. (Call recursively) Obtainx0_ and x1_ by recursion.
// 4. (Partition) Partition all elements in the original nxn elements into three parts: 
// P1={e: e < x0_}, P2={e: x0_ <= e < x1_ }, P3={e: x1_ < e}. We only need to record the 
// cardinality of P1 and P2 (denoted as |P1| and |P2| respectively), and the elements in P2. 
// Obviously, the cardinality of P2 is O(n).
// 5. (Get x0 and x1) From the definition of k0_ and k1_, we have |P1| < k0 <= |P1|+|P2|. 
// When |P1| < k0 < |P1|+|P2|, x0 is the k0-|P1|-th element of P2; otherwise x0=x1_. 
// x1 can be determined in a similar way. This action is also O(n).

// Complexities:

//     Time: O(n) ----- Apply T(n) = T(n/2) + O(n) in the Master's Theorem.
//     Space: O(n)



class Solution {
public:
	int kthSmallest(const std::vector<std::vector<int>> & matrix, int k)
	{
		if (k == 1) // guard for 1x1 matrix
		{
			return matrix.front().front();
		}

		size_t n = matrix.size();
		std::vector<size_t> indices(n);
		std::iota(indices.begin(), indices.end(), 0);
		std::array<size_t, 2> ks = { k - 1, k - 1 }; // use zero-based indices
		std::array<int, 2> results = biSelect(matrix, indices, ks);
		return results[0];
	}

private:
	// select two elements from four elements, recursively
	std::array<int, 2> biSelect(
		const std::vector<std::vector<int>> & matrix,
		const std::vector<size_t> & indices,
		const std::array<size_t, 2> & ks)
	// Select both ks[0]-th element and ks[1]-th element in the matrix,
	// where k0 = ks[0] and k1 = ks[1] and n = indices.size() satisfie
	// 0 <= k0 <= k1 < n*n  and  k1 - k0 <= 4n-4 = O(n)   and  n>=2
	{
		size_t n = indices.size();		
		if (n == 2u) // base case of resursion
		{			
			return biSelectNative(matrix, indices, ks);
		}
		
		// update indices
		std::vector<size_t> indices_;
		for (size_t idx = 0; idx < n; idx += 2)
		{
			indices_.push_back(indices[idx]);
		}
		if (n % 2 == 0) // ensure the last indice is included
		{
			indices_.push_back(indices.back());
		}

		// update ks
		// the new interval [xs_[0], xs_[1]] should contain [xs[0], xs[1]]
		// but the length of the new interval should be as small as possible
		// therefore, ks_[0] is the largest possible index to ensure xs_[0] <= xs[0]
		// ks_[1] is the smallest possible index to ensure xs_[1] >= xs[1]
		std::array<size_t, 2> ks_ = { ks[0] / 4, 0 };
		if (n % 2 == 0) // even
		{
			ks_[1] = ks[1] / 4 + n + 1;
		}
		else // odd
		{
			ks_[1] = (ks[1] + 2 * n + 1) / 4;
		}

		// call recursively
		std::array<int, 2> xs_ = biSelect(matrix, indices_, ks_);

		// Now we partipate all elements into three parts:
		// Part 1: {e : e < xs_[0]}.  For this part, we only record its cardinality
		// Part 2: {e : xs_[0] <= e < xs_[1]}. We store the set elementsBetween
		// Part 3: {e : x >= xs_[1]}. No use. Discard.
		std::array<int, 2> numbersOfElementsLessThanX = { 0, 0 };
		std::vector<int> elementsBetween; // [xs_[0], xs_[1])

		std::array<size_t, 2> cols = { n, n }; // column index such that elem >= x
		 // the first column where matrix(r, c) > b
		 // the first column where matrix(r, c) >= a
		for (size_t row = 0; row < n; ++row)
		{
			size_t row_indice = indices[row];
			for (size_t idx : {0, 1})
			{
				while ((cols[idx] > 0)
					&& (matrix[row_indice][indices[cols[idx] - 1]] >= xs_[idx]))
				{
					--cols[idx];
				}
				numbersOfElementsLessThanX[idx] += cols[idx];
			}
			for (size_t col = cols[0]; col < cols[1]; ++col)
			{
				elementsBetween.push_back(matrix[row_indice][indices[col]]);
			}
		}

		std::array<int, 2> xs; // the return value
		for (size_t idx : {0, 1})
		{
			size_t k = ks[idx];
			if (k < numbersOfElementsLessThanX[0]) // in the Part 1
			{
				xs[idx] = xs_[0];
			}
			else if (k < numbersOfElementsLessThanX[1]) // in the Part 2
			{
				size_t offset = k - numbersOfElementsLessThanX[0];
				std::vector<int>::iterator nth = std::next(elementsBetween.begin(), offset);
				std::nth_element(elementsBetween.begin(), nth, elementsBetween.end());
				xs[idx] = (*nth);
			}
			else // in the Part 3
			{
				xs[idx] = xs_[1];
			}
		}
		return xs;
	}

	// select two elements from four elements, using native way
	std::array<int, 2> biSelectNative(
		const std::vector<std::vector<int>> & matrix,
		const std::vector<size_t> & indices,
		const std::array<size_t, 2> & ks)
	{
		std::vector<int> allElements;
		for (size_t r : indices)
		{
			for (size_t c : indices)
			{
				allElements.push_back(matrix[r][c]);
			}
		}
		std::sort(allElements.begin(), allElements.end());
		std::array<int, 2> results;
		for (size_t idx : {0, 1})
		{
			results[idx] = allElements[ks[idx]];
		}
		return results;
	}
};
