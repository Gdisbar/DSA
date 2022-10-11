https://leetcode.com/problems/k-th-smallest-prime-fraction/discuss/115819/Summary-of-solutions-for-problems-%22reducible%22-to-LeetCode-378
===========================================================================================================================================================
786. K-th Smallest Prime Fraction
===================================

// This post is a quick summary of all common solutions applicable to problems similar 
// to 378. Kth Smallest Element in a Sorted Matrix, where we are given an n x n matrix 
// with each of the rows and columns sorted in ascending order, and need to find the 
// kth smallest element in the matrix.

// For your reference, the following is a list of LeetCode problems that can be 
// transformed into problems equivalent to finding the "Kth Smallest Element in a 
// Sorted Matrix" and thus can be solved effectively using the algorithms developed 
// here:

//     373. Find K Pairs with Smallest Sums
//     378. Kth Smallest Element in a Sorted Matrix
//     668. Kth Smallest Number in Multiplication Table
//     719. Find K-th Smallest Pair Distance
//     786. K-th Smallest Prime Fraction

// This list may not be complete and you''re welcome to add more examples. 
// Also if you''re only interested in solutions to this problem 
// (i.e., 786. K-th Smallest Prime Fraction), you may go directly to the last 
// section (section VIII) where I have listed a couple of solutions as
// demonstrations to algorithms that will be derived shortly.

I -- Sorting-based solution
--------------------------------
// This is the most straightforward solution, where we put all the elements in the matrix into an array and sort 
// it in ascending order, then the kth smallest element in the matrix will be the one at index k-1 of the array. 



public int kthSmallest(int[][] matrix, int k) {
    int n = matrix.length, i = 0;
    
    int[] nums = new int[n * n];
    
    for (int[] row : matrix) {
        for (int ele : row) {
            nums[i++] = ele;
        }
    }
    
    Arrays.sort(nums);
    
    return nums[k - 1];
}


// Time complexity: O(n^2 * log(n^2))
// Space complexity: O(n^2)

// As expected, this naive solution is not very performant in terms of time (due to sorting) 
// and space (due to the extra array) complexities, 

II -- PriorityQueue-based solution
-------------------------------------
// One observation here is that we don't need to keep track of all the n^2 elements since only 
// the kth smallest element is required. We may maintain a PriorityQueue with size k to track 
// only the k smallest elements in the matrix. At the end, the kth smallest element in the matrix 
// will be the largest one in the PriorityQueue. 


public int kthSmallest(int[][] matrix, int k) {
    PriorityQueue<Integer> pq = new PriorityQueue<>(Collections.reverseOrder());
    
    for (int[] row : matrix) {
        for (int ele : row) {
            pq.offer(ele);
            
            if (pq.size() > k) {
                pq.poll();
            }
        }
    }
    
    return pq.peek();
}


// Time complexity: O(n^2 * logk)
// Space complexity: O(k)

// Though in the worst case (k = n^2), the time and space complexities of this solution are the same as 
// those of the naive solution above, we are still able to achieve slight performance gains for average cases where k 
// is generally smaller compared to n^2.

III -- PriorityQueue-based solution with optimization
---------------------------------------------------------
// Up to this point, you may notice that the above two solutions actually apply to arbitrary matrices -- they 
// will find the kth smallest element in the matrix whether or not its rows or columns are sorted. What happens
// if we take advantage of the sorted properties of the matrix?

// Well, I have yet to mention another straightforward way for finding the kth smallest element in the matrix: 
// if we keep removing the smallest element from the matrix for k-1 times, then after removing, the smallest 
// element now in the matrix will be the kth smallest element we are looking for.

// If the elements in the matrix are in random order, we have no better ways for finding and removing the smallest 
// elements in the matrix other than adding all elements to a PriorityQueue. This will yield a time (and space) 
// complexity no better, if not worse, than that of the naive solution. However, with the rows (or columns) of the 
// matrix sorted, we don''t have to add all elements to the PriorityQueue at once. Instead, we can create a pool of 
// candidate elements as long as we can ensure that it contains the smallest element of the matrix (even after removing).

// Assume the rows are sorted in ascending order, which implies initially the smallest element of the matrix must be 
// within the first column, therefore the pool can be initialized with elements from the first column. Now as we are 
// extracting and removing smallest elements from the pool, we need to supplement more candiate elements. 
// The key observation here is that for each extracted element, it suffices to add to the pool only the element 
// immediately following it in the same row, without violating the property that the pool always contains the 
// smallest element of the matrix (even after removing). 

public int kthSmallest(int[][] matrix, int k) {
    PriorityQueue<int[]> pq = new PriorityQueue<>(new Comparator<int[]>() {
        public int compare(int[] a, int[] b) { // (a[0], a[1]) and (b[0], b[1]) are positions in the matrix
            return Integer.compare(matrix[a[0]][a[1]], matrix[b[0]][b[1]]);
        }
    });
    
    int n = matrix.length;
    
    for (int i = 0; i < n; i++) {
        pq.offer(new int[] {i, 0});  // initialize the pool with elements from the first column
    }
    
    while (--k > 0) {                // remove the smallest elements from the matrix (k-1) times
        int[] p = pq.poll();
        
        if (++p[1] < n) {
            pq.offer(p);             // add the next element in the same row if it exists
        }
    }
    
    return matrix[pq.peek()[0]][pq.peek()[1]];
}

// C++
vector<int> kthSmallestPrimeFraction(vector<int>& A, int K) {
        priority_queue<pair<double, pair<int,int>>>pq;
        for(int i = 0; i < A.size(); i++)
            pq.push({-1.0*A[i]/A.back(), {i,A.size()-1}});
        while(--K > 0)
        {
            pair<int,int> cur = pq.top().second;
            pq.pop();
            cur.second--;
            pq.push({-1.0*A[cur.first]/A[cur.second], {cur.first, cur.second}});
        }
        return {A[pq.top().second.first], A[pq.top().second.second]};
    }

// Time complexity: O(max(n, k) * logn)
// Space complexity: O(n)

// Note that by making use of the sorted properties of the matrix, we were able to cut down the space complexity 
// to O(n) and achieve a slightly better time complexity compared to that of the naive solution 
// (though the worst case is the same).

IV -- BinarySearch-based solution
----------------------------------
// The binary search solution is essentially a special case of the more general "trial and error" 
// algorithm described in my other post, where I have summarized the algorithm''s key ideas and demonstrated 
// them for LeetCode 719. I would recommend you read at least the beginning part to get familiar with the 
// terminologies that will be used here.


// The candidate solution: in this case, the cadidate solution is simply an integer.

// The search space: in this case, the search space is given by [MIN, MAX], where MIN and MAX are the minimum 
// and maximum elements in the matrix, respectively.

// The traverse method: in this case, we can do a binary search since the search space is sorted naturally 
// in ascending order (this also accounts for the name "BinarySearch-based solution").

// The verification algorithm: in this case, the verification algorithm is implemented by comparing the count 
// of elements in the matrix less than or equal to the candidate solution, denoted as cnt, 
// with the rank k: if cnt < k, we throw away the left half of the search space; otherwise we discard the right half.


// Note that the verification algorithm is based on the following two observations:

//     There will be at least k elements in the matrix that are less than or equal to the kth smallest element.

//     If there are at least k elements in the matrix that are less than or equal to a candidate solution, 
// then the actual kth smallest element in the matrix cannot be greater than this candiate solution.

// Also the counting of elements no greater than a candidate solution can be done in linear time by employing the 
// classic two-pointer techinique which takes advantage of the sorted properties of the matrix. 



public int kthSmallest(int[][] matrix, int k) {
    int n = matrix.length;
    
    int l = matrix[0][0];               // minimum element in the matrix
    int r = matrix[n - 1][n - 1];       // maximum element in the matrix
    
    for (int cnt = 0; l < r; cnt = 0) { // this is the binary search loop
        int m = l + (r - l) / 2;
        
        for(int i = 0, j = n - 1; i < n; i++)  {
            while (j >= 0 && matrix[i][j] > m) j--;  // the pointer j will only go in one direction
            cnt += (j + 1);
        }
        
        if (cnt < k) {
            l = m + 1;   // cnt less than k, so throw away left half
        } else {
            r = m;       // otherwise discard right half
        }
    }
    
    return l;
}

//C++

vector<int> kthSmallestPrimeFraction(vector<int>& arr, int k) {
        int n = arr.size();
        double l = 0, r = 1.0;
        while(l<r){
            double m = (l+r)/2;
			//max_f is used to store the maximum fraction less than mid
            double max_f = 0.0;
			//p and q are used for storing the indices of max fraction
            int total=0,p=0,q=0;
            int j=1;
            for(int i=0;i<n-1;i++){
			//if this fraction is greater than mid , move denominator rightwards to find a smaller mid
                while(j<n && arr[i] > m*arr[j])
                    j++;
					//j elements are greater than mid in this row , n-j are smaller , add them to result
                total += (n-j);
                if(j==n)
                    break;
				//cast to double speedily
                double f = static_cast<double>(arr[i]) / arr[j];
				//update max fraction for this mid
                if (f > max_f) {
                  p = i;
                  q = j;
                  max_f = f;
                }
            }
            if (total == k)
                return {arr[p], arr[q]};       
			//there are too many fractions less than mid=> mid is too big => make mid smaller and try
            else if (total > k)
                r = m;
            else
                l = m;
        }
        return {};
    }

// Time complexity: O(n * log(MAX - MIN))
// Space complexity: O(1)

// The binary search solution is much more efficient compared to those derived in previous sections. 
// We use only constant space and the time complexity is almost linear for any practical integer 
// matrices (for which MIN and MAX are within the 32-bit integer range).

V -- ZigzagSearch-based solution
----------------------------------------
// The zigzag search solution is another special version of the more general "trial and error" algorithm, 
// where now the search space is formed only by integers contained in the input matrix itself, rather than 
// continous integers in the range [MIN, MAX]. We can summarize the following properties of this solution:


// The candidate solution: in this case, the cadidate solution is also an integer, but it must be an element of the input matrix.

// The search space: in this case, the search space is given by the input matrix itself.

// The traverse method: in this case, we cannot do a binary search since there is no total ordering of 
// the candidate solutions. However, we do have partial orderings where each row and column of the matrix 
// are sorted in ascending order. This enables us to do a zigzag search where we start from the upper-right 
// corner of the matrix and proceed to either the next row or previous column depending on the result of the 
// verification algorithm (we can also start from the bottom-left corner and proceed to either the previous row or next column).

// The verification algorithm: in this case, the verification algorithm is implemented by comparing two counts, 
// denoted as cnt_lt and cnt_le respectively, with the rank k: if cnt_le < k, we proceed to the 
// next row; else if cnt_lt >= k, we proceed to the previous column; otherwise 
// we've found the kth smallest element in the matrix so return it. Here cnt_lt denotes the number of elements 
// in the matrix less than the candidate solution, while cnt_le denotes the number of elements in the matrix 
// less than or equal to the candidate solution (the reason we need two counts is that there 
// may be duplicates in the input matrix).


// Note that the verification algorithm is based on the following two observations:

//     There will be at most k - 1 elements in the matrix that are less than the kth smallest element.

//     There will be at least k elements in the matrix that are less than or equal to the kth smallest element.

// And again the counting can be done in linear time using the classic two-pointer techinique. 



public int kthSmallest(int[][] matrix, int k) {
    int n = matrix.length;
    
    int row = 0;          // we start from the upper-right corner
    int col = n - 1;      
    
    for (int cnt_le = 0, cnt_lt = 0; true; cnt_le = 0, cnt_lt = 0) {
        for (int i = 0, j = n - 1, p = n - 1; i < n; i++) {
            while (j >= 0 && matrix[i][j] > matrix[row][col]) j--;    // pointer j for counting cnt_le
            cnt_le += (j + 1);
            
            while (p >= 0 && matrix[i][p] >= matrix[row][col]) p--;   // pointer p for counting cnt_lt
            cnt_lt += (p + 1);
        }
        
        if (cnt_le < k) {         // candidate solution too small so increase it
            row++; 
        } else if (cnt_lt >= k) { // candidate solution too large so decrease it
            col--;
        } else {                  // candidate solution equal to the kth smallest element so return
            return matrix[row][col];
        }
    }
}


// Time complexity: O(n^2)
// Space complexity: O(1)

// The zigzag search solution is less efficient compared to the binary search solution, 
// but still achieves significant performance gains compared to the other three solutions.

VI -- BiSelect solution
----------------------------------

//https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/discuss/85201/c-on-time-on-space-solution-with-detail-intuitive-explanation


// Time complexity: O(n)
// Space complexity: O(n)

VII -- Transform other problems into "Kth Smallest Element in a Sorted Matrix"
--------------------------------------------------------------------------------------------


// Now the key to the transformation process is to build the sorted matrix in terms of the problem input. On the one 
// hand, we need to relate each matrix element matrix[i][j] to the problem input. On the other hand, we need to arrange 
// these elements so that each row and column of the matrix are sorted in ascending order. Again, these details will 
// depend on the nature of the problems. But as far as this post is concerned, I will assume the input are two arrays 
// sorted in ascending order, denoted as a and b respectively, and the matrix elements can be computed using arithmetic 
// operations of the two array elements.

//     Addition: matrix[i][j] = a[i'] + b[j']
//     Subtraction: matrix[i][j] = a[i'] - b[j']
//     Multiplication: matrix[i][j] = a[i'] * b[j']
//     Division: matrix[i][j] = a[i'] / b[j']

// Note that the index mappings i --> i' and j --> j' are not necessarily identity mappings, but rather they 
// will be chosen to ensure that the rows and columns in the matrix are sorted in ascending order. 
// For simplicity, we name two type of mappings here which are identity mapping (where i' = i and j' = j) 
// and "negative" mapping (where i' = na - 1 - i and j' = nb - 1 - j, with na, nb being the lengths of the two arrays).


// Next I will reformulate LeetCode problems 378, 668, 719 and 786 as examples to show how the transformation is done:

//     373. Find K Pairs with Smallest Sums: In this case, the two input arrays are nums1 and nums2, so a will 
//     refer to nums1 and b to nums2. And we have matrix[i][j] = nums1[i] + nums2[j], where identity mappings 
//     are chosen for both the row and column indices. Note that for this problem we are required to find all the 
//     K pairs with the smallest sums, so the optimized PriorityQueue solution would be the best to try.


//     668. Kth Smallest Number in Multiplication Table: In this case, only the lengths of the two arrays are 
//     given as input, but it's trival to get the array elements (a[i] = i + 1, b[j] = j + 1). And we 
//     have matrix[i][j] = a[i] * b[j], where again identity mappings are chosen for both the row and column indices. 
//     The Kth smallest element in the multiplication table will be the same as the Kth smallest element in the matrix.


//     719. Find K-th Smallest Pair Distance: In this case there is only one array nums as input, so both a and b will 
//     refer to nums. After sorting nums in ascending order, we have matrix[i][j] = nums[i] - nums[n - 1 - j], 
//     where n = nums.length. Note that the mapping for column index is chosen to be "negative" in order to make sure 
//     the columns are sorted in ascending order. Also note that our matrix will contain all pair distances 
//     (including negative ones) while the original problem asks for the Kth smallest pair distance out of absolute 
//     pair distances (there are n(n-1)/2 such pairs). So we need to shift the rank up to K' = K + n(n+1)/2. 
//     The Kth smallest pair distance then will be the K'th smallest element in the matrix.


//     786. K-th Smallest Prime Fraction: In this case again there is only one array A as input, so both a 
//     and b will refer to A. We have matrix[i][j] = nums[i] / nums[n - 1 - j], where n = A.length and / is floating 
//     point division (not integer division). Again the mapping for column index is chosen to be "negative" to 
//     ensure the columns are sorted in ascending order. The Kth smallest prime fraction then will be the Kth 
//     smallest element in the matrix.


// Note that there may be other properties of the matrix that are specific to a particular problem, such as 
// whether the matrix contains duplicates or not, whether the matrix elements are integers or not, etc. 
// These extra properties may help optimize the algorithms developed earlier.

   
