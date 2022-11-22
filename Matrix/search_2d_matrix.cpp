74. Search a 2D Matrix
=========================
Write an efficient algorithm that searches for a value target in an m x n 
integer matrix matrix. This matrix has the following properties:

    Integers in each row are sorted from left to right.
    The first integer of each row is greater than the last integer of the 
    previous row.

 

Example 1:

Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
Output: true



Example 2:

Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 13
Output: false

// r*c matrix convert to an array => matrix[x][y] => a[x*c+y]
// an array convert to r*c matrix => a[x] =>matrix[x/c][x%c];

bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int r = matrix.size(),c=matrix[0].size();
        int l = 0, h = r*c-1;
        while(l!=h){
            int m = (h+l-1)>>1;
            if(matrix[m/c][m%c]<target)
                l=m+1;
            else
                h=m;
        }
        return matrix[h/c][h%c]==target;
    }



// BS + recursion --> faster

class Solution {
private:
    int findRow(int start, int end, int col, vector<vector<int>>& matrix, int target){
        if(start > end) return -1;
        int mid = (start+end)/2;
        if(matrix[mid][0] <= target && matrix[mid][col] >= target){
            return mid;
        } else if(matrix[mid][col] > target){
            return findRow(0, mid -1, col, matrix, target);
        }else{
            return findRow(mid +1, end, col, matrix, target);
        }
    }
    int findCol(int start, int end, int row,vector<vector<int>>& matrix, int target){
        if(start > end) return -1; 
        int mid = (start+end)/2;
        if(matrix[row][mid] == target){ 
            return mid;
        } else if(matrix[row][mid] > target){
            return findCol(0, mid -1, row, matrix, target);
        }else{
            return findCol(mid +1, end, row, matrix, target);
        }
    }
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int r = matrix.size()-1,c = matrix[0].size()-1;
        if(r <= 0 & c <= 0){
          return matrix[r][c] == target;
        } 

        int row = findRow(0, r, c, matrix, target);
        // If row == -1 then we know that the value can either present in 0th row or nowhere.
        int column = findCol(0, c, row == -1 ? 0 : row, matrix, target);
        return row != -1 && column != -1;

    }
};


240. Search a 2D Matrix II
============================
Write an efficient algorithm that searches for a value target in an m x n integer 
matrix matrix. This matrix has the following properties:

    Integers in each row are sorted in ascending from left to right.
    Integers in each column are sorted in ascending from top to bottom.

 

Example 1:

Input: matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],
    [18,21,23,26,30]], target = 5
Output: true


// If we use the above approach it'll fail fr this case
// Input: [[1,4],[2,5]]
// 2
// Output: false
// Expected: true

//M-1, m*log(n)
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
    int m = matrix.size();
    if(m<1) return false;
    int n = matrix[0].size();
    if(n<1) return false;
    
    int i=0;
    while(i<m && target >= matrix[i][0])
    {
        int low = 0;
        int high = n;
        while(low < high)
        {
            int mid = (low+high)/2;
            if(matrix[i][mid] > target)
                high = mid;
            else if(matrix[i][mid] < target)
                low = mid+1;
            else
                return true;
        }
        ++i;
    }
    return false;
}
//M-2
Search from the top-right element and reduce the search space by one row or 
column at each time.

[[ 1,  4,  7, 11, 15],
 [ 2,  5,  8, 12, 19], 
 [ 3,  6,  9, 16, 22],
 [10, 13, 14, 17, 24],
 [18, 21, 23, 26, 30]]

Suppose we want to search for 12 in the above matrix. compare 12 with the 
top-right element nums[0][4] = 15. Since 12 < 15, 12 cannot appear in the column 
of 15 since all elements in that column are greater than or equal to 15. Now we 
reduce the search space by one column (the last column).

We further compare 12 with the top-right element of the remaining matrix, 
which is nums[0][3] = 11. Since 12 > 11, 12 cannot appear in the row of 11 since 
all elements in this row are less than or equal to 11 (the last column has been 
discarded). Now we reduce the search space by one row (the first row).

We move on to compare 12 with the top-right element of the remaining matrix, 
which is nums[1][3] = 12. Since it is equal to 12, we return true.


bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int row=matrix.size() , col=matrix[0].size();
        if(row==0)return false;
        int r = 0,c=col-1;
        while(r<row&&c>=0){
            if(matrix[r][c]==target){
                return true;
            }
            else if(matrix[r][c]>target){
                c--;
            }
            else{
                r++;
            }
        }
        return false;
    }

There are 3 recursive calls, and each call reduce the problem size to 1/4 of the 
original problem.

So T(N) <= 3T(N/4) + O(1). Using master theorem, the time complexity is 
O(N^(log(4,3)), which is approximately O(N^0.79), where N is the total number 
of elements in the matrix, which is m*n