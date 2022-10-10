378. Kth Smallest Element in a Sorted Matrix
==============================================
// Given an n x n matrix where each of the rows and columns is sorted in 
// ascending order, return the kth smallest element in the matrix.

// Note that it is the kth smallest element in the sorted order, not the 
// kth distinct element.

// You must find a solution with a memory complexity better than O(n2).

 

// Example 1:

// Input: matrix = [[1,5,9],[10,11,13],[12,13,15]], k = 8
// Output: 13
// Explanation: The elements in the matrix are [1,5,9,10,11,12,13,13,15], 
// and the 8th smallest number is 13

// Example 2:

// Input: matrix = [[-5]], k = 1
// Output: -5


//     Time: O(K * logK)
//     Space: O(K)


int kthSmallest(vector<vector<int>> &matrix, int k) {
        int m = matrix.size(), n = matrix[0].size(), ans; 
        // For general, the matrix need not be a square
        // pq = {val,r,c}
        priority_queue<vector<int>, vector<vector<int>>, greater<>> minHeap;
        for (int r = 0; r < min(m, k); ++r)
            minHeap.push({matrix[r][0], r, 0});

        for (int i = 1; i <= k; ++i) {
            auto top = minHeap.top(); minHeap.pop();
            int r = top[1], c = top[2];
            ans = top[0];
            if (c + 1 < n) minHeap.push({matrix[r][c + 1], r, c + 1});
        }
        return ans;
    }


class Solution:  # 204 ms, faster than 54.32%
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        m, n = len(matrix), len(matrix[0])  # For general, the matrix need not be a square
        minHeap = []  # val, r, c
        for r in range(min(k, m)):
            heappush(minHeap, (matrix[r][0], r, 0))

        ans = -1  # any dummy value
        for i in range(k):
            ans, r, c = heappop(minHeap)
            if c+1 < n: heappush(minHeap, (matrix[r][c + 1], r, c + 1))
        return ans

// Time: O((M+N) * logD), where M <= 300 is the number of rows, N <= 300 is the 
// number of columns, D <= 2*10^9 is the 
// difference between the maximum element and the minimum element in the matrix.

// How to count number of elements less or equal to x efficiently?

// Since our matrix is sorted in ascending order by rows and columns.
// We use two pointers, one points to the rightmost column c = n-1, and 
// one points to the lowest row r = 0.
// 1. If matrix[r][c] <= x then the number of elements in row r less or equal 
//     to x is (c+1) (Because row[r] is sorted in ascending order, so if 
//     matrix[r][c] <= x then matrix[r][c-1] is also <= x). 
//     Then we go to next row to continue counting.
// 2. Else if matrix[r][c] > x, we decrease column c until matrix[r][c] <= x 
//     (Because column is sorted in ascending order, so if matrix[r][c] > x 
//     then matrix[r+1][c] is also > x).

// Time complexity for counting: O(M+N).



class Solution { // 20 ms, faster than 98.92%
public:
    int m, n;
    int kthSmallest(vector<vector<int>>& matrix, int k) {
        m = matrix.size(), n = matrix[0].size(); // For general, the matrix need not be a square
        int left = matrix[0][0], right = matrix[m-1][n-1], ans = -1;
        while (left <= right) {
            int mid = (left + right) >> 1;
	//find smaller value by searching in the left side. Otherwise, we search 
    //in the right side.
            if (countLessOrEqual(matrix, mid) >= k) {
                ans = mid;
                right = mid - 1; // try to looking for a smaller value in the left side
            } else left = mid + 1; // try to looking for a bigger value in the right side
        }
        return ans;
    }
    int countLessOrEqual(vector<vector<int>>& matrix, int x) {
        int cnt = 0, c = n - 1; // start with the rightmost column
        for (int r = 0; r < m; ++r) {
            while (c >= 0 && matrix[r][c] > x) 
            	--c;  // decrease column until matrix[r][c] <= x
            cnt += (c + 1);
        }
        return cnt;
    }
};