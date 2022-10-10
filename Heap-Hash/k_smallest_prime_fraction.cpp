786. K-th Smallest Prime Fraction
======================================
// You are given a sorted integer array arr containing 1 and prime numbers, 
// where all the integers of arr are unique. You are also given an integer k.

// For every i and j where 0 <= i < j < arr.length, we consider the fraction 
// arr[i] / arr[j].

// Return the kth smallest fraction considered. Return your answer as an array 
// of integers of size 2, where answer[0] == arr[i] and answer[1] == arr[j].

 

// Example 1:

// Input: arr = [1,2,3,5], k = 3
// Output: [2,5]
// Explanation: The fractions to be considered in sorted order are:
// 1/5, 1/3, 2/5, 1/2, 3/5, and 2/3.
// The third fraction is 2/5.

// Example 2:

// Input: arr = [1,7], k = 1
// Output: [1,7]

// it is like find k-th smallest element in n sorted array, which has a 
// classic solution using priority_queue

// consider an input of [n1, n2, n3, n4, n5], the possible factors are:
// [n1/n5, n1/n4, n1/n3, n1/n2, n1/n1]
// [n2/n5, n2/n4, n2/n3, n2/n2]
// [n3/n5, n3/n4, n3/n3]
// [n4/n5, n4/n4]
// [n5/n5]

class Solution {
public:
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
};





//Pattern reducible to -  "Kth Smallest Element in a Sorted Matrix" 
https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/discuss/1322101/C%2B%2BJavaPython-MaxHeap-MinHeap-Binary-Search-Picture-Explain-Clean-and-Concise

//     373. Find K Pairs with Smallest Sums
//     378. Kth Smallest Element in a Sorted Matrix
//     668. Kth Smallest Number in Multiplication Table
//     719. Find K-th Smallest Pair Distance
//     786. K-th Smallest Prime Fraction

// This list may not be complete and you're welcome to add more examples. Also if 
// you're only interested in solutions to this problem (i.e., 786. K-th Smallest 
// Prime Fraction), you may go directly to the last section (section VIII) 
// where I have listed a couple of solutions as demonstrations to algorithms 
// that will be derived shortly.

// I -- Sorting-based solution

int kthSmallest(vector<vector < int>> &matrix, int k)
        {
            int count = 1;
            vector<int> v;
            for (auto &arr: matrix)
            {
                for (auto &i: arr)
                {
                    v.push_back(i);
                }
            }
            sort(v.begin(), v.end());

            return v[k - 1];
        }


// Time complexity: O(n^2 * log(n^2))
// Space complexity: O(n^2)

// II -- PriorityQueue-based solution or using multiset (insert all + pop 1st k)

int kthSmallest(vector<vector<int>>& matrix, int k) 
    {
        priority_queue<int> max_heap;

        for(auto row : matrix)
        {
            for(auto col : row)
            {
                max_heap.push(col);
                if(max_heap.size()>k)
                    max_heap.pop();
            }
        }

        return max_heap.top();
    }

// Time complexity: O(n^2 * logk)
// Space complexity: O(k)

//III -- BinarySearch-based solution (assuming matrix sorted)

 int kthSmallest(vector<vector<int>>& matrix, int k) {
        // Store value of matrix size
        int n = matrix.size();
        
        int low = matrix[0][0]; // first element
        int high = matrix[n-1][n-1]; // Last element
        
        int mid, temp, count;
        
        while(low < high){
            mid = low + (high-low)/2;
            temp = n - 1;
            count = 0;
            
            // For each row count the elements that are smaller than mid
            for(int i = 0; i < n; i++){
                
                while(temp >= 0 && matrix[i][temp] > mid){
                    temp--;
                }
                count+= (temp+1);
            }
            
            if(count < k){
                low = mid + 1;
            }else{
                high = mid;
            }
        }
        return low;
    }

//V -- ZigzagSearch-based solution

// The search space: in this case, the search space is given by the input 
// matrix itself.

// start from the upper-right corner of the matrix and proceed to either the next 
// row or previous column depending on the result of the verification 
// algorithm (we can also start from the bottom-left corner and proceed to 
// 	either the previous row or next column).

// cnt_lt = no. of elements in the matrix less than the candidate solution
// cnt_le = no. of elements in the matrix less than or 
// equal to the candidate solution (the reason we need two counts is that 
// there may be duplicates in the input matrix)

// The verification algorithm: in this case, the verification algorithm 
// is implemented by comparing two counts, denoted as cnt_lt and cnt_le 
// respectively, with the rank k: if cnt_le < k, we proceed to the next row; 
// else if cnt_lt >= k, we proceed to the previous column; otherwise we''ve 
// found the kth smallest element in the matrix so return it.

// Note that the verification algorithm is based on the following two observations:

//     There will be at most k - 1 elements in the matrix that are less than the 
//     kth smallest element.

//     There will be at least k elements in the matrix that are less than or 
//     equal to the kth smallest element.


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

