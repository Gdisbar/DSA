378. Kth Smallest Element in a Sorted Matrix
================================================
Given an n x n matrix where each of the rows and columns is sorted in ascending 
order, return the kth smallest element in the matrix.

Note that it is the kth smallest element in the sorted order, not the 
kth distinct element.

You must find a solution with a memory complexity better than O(n^2).

 

Example 1:

Input: matrix = [[1,5,9],[10,11,13],[12,13,15]], k = 8
Output: 13
Explanation: The elements in the matrix are [1,5,9,10,11,12,13,13,15], 
and the 8th smallest number is 13

Example 2:

Input: matrix = [[-5]], k = 1
Output: -5


//M-1 convert to 1D array (n*n) + sort(nlog(n))+find k-th again k

//M-2, using maxheap

// TC: r*c*log(k) , r<=300,c<=300 
// SC: k

int kthSmallest(vector<vector<int>> &matrix, int k) {
        int row = matrix.size(), col = matrix[0].size(); // For general, the matrix need not be a square
        priority_queue<int> maxHeap;
        for (int r = 0; r < row; ++r) {
            for (int c = 0; c < col; ++c) {
                maxHeap.push(matrix[r][c]);
                if (maxHeap.size() > k) maxHeap.pop();
            }
        }
        return maxHeap.top();
    }

//M-3, minheap
//TC : K * logK
//SC : K
We start the pointers to point to the beginning of each rows, then we iterate k 
times, for each time ith, the top of the minHeap is the ith smallest element in the 
matrix. We pop the top from the minHeap then add the next element which has the same 
row with that top to the minHeap.

[ [1,3,7],
  [5,10,12],  
  [6,10,15]
 ]
K=4
i=1,minHeap.top=1,add=3  
i=2,minHeap.top=3,add=7
i=3,minHeap.top=5,add=10
i=4,minHeap.top=6,add=10,ans=6

int kthSmallest(vector<vector<int>> &matrix, int k) {
        int m = matrix.size(), n = matrix[0].size(), ans; 
        //priority_queue <Type, vector<Type>, ComparisonType > min_heap;
        //priority_queue <int, vector<int>, greater<int> > pq; --> store single value
        priority_queue<vector<int>, vector<vector<int>>, greater<>> minHeap;
        for (int r = 0; r < min(m, k); ++r) // as we need k elements
            minHeap.push({matrix[r][0], r, 0});

        for (int i = 1; i <= k; ++i) {
            auto top = minHeap.top(); minHeap.pop();
            int r = top[1], c = top[2];
            ans = top[0];
            if (c + 1 < n) minHeap.push({matrix[r][c + 1], r, c + 1});
        }
        return ans;
    }

//M-4 , Binary Search


//TC: (M+N) * logD,  D <= 2*10^9 is the difference between the maximum element and the minimum element in the matrix.
//SC: 1

We binary search to find the smallest ans in range [minOfMatrix...maxOfMatrix] as 
long as countLessOrEqual(ans) >= k, where countLessOrEqual(x) is the number of 
elements less than or equal to x.
//This is similar to upper_bound() in finding median in row-wise sorted matrix

--> Why ans must be as smallest as possible?

[ [1,3,7],
  [5,10,12], ----> k=5,o/p = 7 ,when m=8 countLessOrEqual(m)==5(also equal to k)
  [6,10,15]       if we return m=8 as ans,it''s wrong as we''ve 7 as better ans as         
 ]                it''s smallest m for which countLessOrEqual(m=7)>=k 

--> Why countLessOrEqual(ans) >= k but not countLessOrEqual(ans) == k?
    
[ [1,3,7],   ----> k=6,o/p = 10 ,when m=10 countLessOrEqual(m)==7(greater than k)
  [5,10,12],      it''s wrong,if we don''t save the ans before looking left. 
  [6,10,15]       so we save the m=10 to ans as it''s smallest m for which countLessOrEqual(m=10)>=k 
 ]


Algorithm
------------
Start with left = minOfMatrix = matrix[0][0] and 
    right = maxOfMatrix = matrix[n-1][n-1].
Find the mid of the left and the right. This middle number is NOT necessarily an 
     element in the matrix.
If countLessOrEqual(mid) >= k, we keep current ans = mid and try to find smaller 
     value by searching in the left side. Otherwise, we search in the right side.
Since ans is the smallest value which countLessOrEqual(ans) >= k, so it''s the k th 
      smallest element in the matrix.

How to count number of elements less or equal to x efficiently?

Since our matrix is sorted in ascending order by rows and columns.
We use two pointers, one points to the rightmost column c = n-1, and one points 
to the lowest row r = 0.
    If matrix[r][c] <= x then the number of elements in row r less or equal to 
    x is (c+1) (Because row[r] is sorted in ascending order, so if matrix[r][c] <= x 
    then matrix[r][c-1] is also <= x). Then we go to next row to continue counting.
    Else if matrix[r][c] > x, we decrease column c until matrix[r][c] <= x 
    (Because column is sorted in ascending order, so if matrix[r][c] > x then 
    matrix[r+1][c] is also > x).

[ [1,3,7],
  [5,10,12], 
  [6,10,15]               
 ]  

l=1,h=15,m=8 , How to count number of elements less or equal to 8 ?

r=0,c=n-1=2,cnt=0 
//matrix[r][c]<=8 will be applicable for cnt=col+1 # col is an index value 
r=0,c=2,matrix[r][c]=7<=8 -> cnt+=3=3
r=1,c=2,matrix[r][c]=10>8  -> decrease c until matrix[r][c]<=8
r=1,c=0,matrix[r][c]=5<=8 -> cnt+=1=4
r=2,c=0,matrix[r][c]<=8 -> cnt+=1=5
cnt=5,so there are 5 elements less than 8

//Time complexity for counting: O(M+N).
//It''s exactly the same idea with this problem: 240. Search a 2D Matrix II


class Solution { // 20 ms, faster than 98.92%
public:
    int m, n;
    int kthSmallest(vector<vector<int>>& matrix, int k) {
        m = matrix.size(), n = matrix[0].size(); // For general, the matrix need not be a square
        int left = matrix[0][0], right = matrix[m-1][n-1], ans = -1;
        while (left <= right) {
            int mid = (left + right) >> 1;
            if (countLessOrEqual(matrix, mid) >= k) {
        //we save current value as countLessOrEqual(ans) >= k, not countLessOrEqual(ans) == k
                ans = mid; 
                right = mid - 1; // try to looking for a smaller value in the left side
            } else left = mid + 1; // try to looking for a bigger value in the right side
        }
        return ans;
    }
    int countLessOrEqual(vector<vector<int>>& matrix, int x) {
        int cnt = 0, c = n - 1; // start with the rightmost column
        for (int r = 0; r < m; ++r) {
            //we've used this technique in finding common element in all rows
            while (c >= 0 && matrix[r][c] > x) --c;  // decrease column until matrix[r][c] <= x
            cnt += (c + 1);
        }
        return cnt;
    }
};