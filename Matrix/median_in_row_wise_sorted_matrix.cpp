Median in a row-wise sorted Matrix 
=====================================
Given a row wise sorted matrix of size RxC where R and C are always odd, 
find the median of the matrix.

Example 1:

Input:
R = 3, C = 3
M = [[1, 3, 5], 
     [2, 6, 9], 
     [3, 6, 9]]

Output: 5

Explanation:
Sorting matrix elements gives us {1,2,3,3,5,6,6,9,9}. Hence, 5 is median. 

// Time Complexity: O(N*M*log(N*M)

// Space Complexity: O(N*M)

int median(vector<vector<int>> &matrix, int r, int c){
       vector<int> tmp(r*c);
       for(int i = 0;i<r;i++)
         for(int j=0;j<c;j++)
            tmp[i*c+j]=matrix[i][j];
       
       sort(tmp.begin(),tmp.end());
       return tmp[(r*c+1)/2-1];
    }




// The idea is that for a number to be median there should be exactly (n/2) 
// numbers that are less than this number. So, we try to find the count of 
// numbers less than all the numbers. 

// For a number to be median, there should be (r*c)/2 numbers smaller than that 
// number. So for every number, we get the count of numbers less than that by 
// using upper_bound() in each row of the matrix, if it is less than the required 
// count, the median must be greater than the selected number, else the median 
// must be less than or equal to the selected number.

M = [[1, 3, 5], 
     [2, 6, 9], 
     [3, 6, 9]]

M={1,2,3,3,5,6,6,9,9}
// count of numbers less than or equal to our mid
l=1,h=9,m=5,cnt=5
l=1,h=5=m,m=3,cnt=4
l=4=m+1,h=5,m=4,cnt=4

int median(vector<vector<int>> &matrix, int r, int c){
       int l=INT_MAX,h=INT_MIN,idx=(1+r*c)/2; //r*c is odd
       for(int i = 0;i<r;i++){
           l=min(l,matrix[i][0]); // 1st column,lowest value
           h=max(h,matrix[i][c-1]); // last column,highest value
       }
       
       while(l<h){
           int m = l+(h-l)/2;
           int cnt = 0;
           for(int i = 0;i<r;i++){
           	// count of numbers less than or equal to our mid,as adjacent columns are sorted
               cnt += upper_bound(matrix[i].begin(),matrix[i].begin()+c,m)-matrix[i].begin();
           }
           if(cnt<idx) l=m+1;
           else h=m;
       }
       return l;
    }

// Time Complexity: O(N*log(M)) 

// Space Complexity: O(1)
