// Pattern --> non-leaf=max(left)*max(right) , find min(Σ non-leaf)
//Approach --> we need to find all possible tree --> DP
//Brute force --> DP (n^3) ---> greedy (n^2) ---> monotonic stack (n)

//Brute force 

// 1. (max value of subtree,sum of non-leaf) // structure
// 2. l>r /* impossible case */(0,INT_MAX) 
//    l==r /* base case*/ (a[l],0)
//    ans= /*initialize with impossible case*/

// 3. find (max value,sum of non-leaf) for a[l...i],a[i+1...r] //left,right
//    find rootVal=max value of left * max value of right
//    total=rootVal+left sum of non-leaf+right sum of non-leaf
//    total<ans//update ans
//        ans.max= max(max value of left , max value of right)
//        ans.sum=total



// Top-Down DP on Tree --->  n element * every combination of (l,r)[~ n* n]

// l >= r , return 0 //Base-case/Start calculation of recursion 
// dp[l][r]!=-1,return dp[l][r] //Return case/end of recursion 
// res=INT_MAX
// for (i=l to r-1){
//     rootVal = fmax(a, l, i) * fmax(a, i+1, r); // max(a[l...i]),max(a[i+1...r])
//     nonLeafNodeFromLeftSubtree = helper(a, l, i,dp);
//     nonLeafNodeFromRightSubtree = helper(a, i+1, r,dp);
//     res = min(res, rootVal+nonLeafNodeFromLeftSubtree
//                     +nonLeafNodeFromRightSubtree);
// }
// dp[l][r] = res; //Memorization 
// return res //One subtree done look for next subtree 


// //Bottom-Up DP --->
// dp[i][i] = 0 //we're not moving al all
// //p=2 to n
// //i=0 to n-p i.e max upto n-2
// j=i-p+1
// // k=i to i-p/j-1
// int rootVal = fmax(a, i, k) * fmax(a, k+1, j);
// dp[i][j] = min(dp[i][j], rootVal + dp[i][k] + dp[k + 1][j]);

// return dp[0][n-1]

// //Greedy ---> (n^2) 
// we combine(left+right+parent) & store result in current node(res)

// //until 1 element left
// a[i]=parent,left=a[i-1],right=a[i+1] //i=index of min(a)
// res+=parent+left*right
// i>0&&i<n-1 //parent+left+right exist
// i==0 //parent+right exist
// i==n-1 //parent+left exist
// erase(a[i]) //delete the current minimum value

// //monotonic stack ---> (n)
// s.push(INT_MAX)
// //for i=0 to n-1
// s.top()<=a[i] // continue pushing into stack even this condition is not met
//     mid=s.top(),s.pop()
//     res += mid * min(stack.top(), num); //left max & right max exist

// //have combined all leaf to non-leaf
// s.size()>2
//     val=s.top(),s.pop()
//     res+=val*s.top(); //propagate upward

1130. Minimum Cost Tree From Leaf Values
============================================
// Given an array arr of positive integers, consider all binary trees such that:

// Each node has either 0 or 2 children;
// The values of arr correspond to the values of each leaf in an in-order 
// traversal of the tree.
// The value of each non-leaf node is equal to the product of the largest leaf 
// value in its left and right subtree, respectively.

// Among all possible binary trees considered, return the smallest possible 
// sum of the values of each non-leaf node. It is guaranteed this sum 
// fits into a 32-bit integer.

// A node is a leaf if and only if it has zero children.



// Example 1:
//                             24            24
//                            /  \          /  \
//                           12   4        6    8
//                          /  \               / \
//                         6    2             2   4

// Input: arr = [6,2,4]
// Output: 32
// Explanation: There are two possible trees shown.
// The first has a non-leaf node sum 36=24+12, and the second has non-leaf 
// node sum 32=24+8.

// Example 2:
//                             44
//                            /  \
//                           4   11

// Input: arr = [4,11]
// Output: 44

//  BRUTE FORCE
--------------------
// class Solution {
//     public int mctFromLeafValues(int[] a) {
//         int n=a.length;
//         return fun(a,0,n-1).sum;
//     }
//     public pair fun(int a[],int l,int r){
//         if(l>r)
//             return new pair(0,1000000);
//         if(l==r)
//             return new pair(a[l],0);
//         pair ans=new pair(0,1000000);
//         for(int i=l;i<r;i++){
//             pair p_left=fun(a,l,i);
//             pair p_right=fun(a,i+1,r);
//             int total=p_left.max*p_right.max+p_left.sum+p_right.sum;
//             if(total<ans.sum)
//             {
//                 ans.max=Math.max(p_left.max,p_right.max);
//                 ans.sum=total;
//             }
//         }
//         return ans;
//     }
//     class pair{
//         int max;
//         int sum;
//         pair(int max,int sum){
//             this.max=max;
//             this.sum=sum;
//         }
//     }
// }


//1. Dynamic programming approach //O(n ^ 3)
----------------------------------------------

// Target- which node belongs to which subtree(left/right) [given list of all 
//     the leaf nodes values (inorder)]

// Approach -
// 1. inorder given , so it has a pivot (left subtree |pivot |right subtree)
// 2. if we know sum for each subtree --> we add them up to get parent sum


// res(i, j) ---> min(all Σ non-leaf) from a[i] to a[j]

// non-leaf from left & right subtree =max(a[i...k]),max(a[k+1...j])

// res(i, j) = min(res(i,k)+res(k+1,j)+max(a[i...k])*max(a[k+1...j])) 
//           = left-subtree + right-subtree + rootVal



// class Solution:
//     def mctFromLeafValues(self, arr: List[int]) -> int:
//         return self.helper(arr, 0, len(arr) - 1, {})
        
//     def helper(self, arr, l, r, cache):
//         if (l, r) in cache:
//             return cache[(l, r)]
//         if l >= r:
//             return 0
        
//         res = float('inf')
//         for i in range(l, r):
//             rootVal = max(arr[l:i+1]) * max(arr[i+1:r+1])
//             res = min(res, rootVal + self.helper(arr, l, i, cache) + self.helper(arr, i + 1, r, cache))
        
//         cache[(l, r)] = res
//         return res

 |  0 |1  | 2 | 3
-|----|---|---|----
0|  0 |12 |32 |inf 
-------------------
1|inf | 0 |8  |inf 
-------------------
2|inf |inf|0  |inf 
----------------------
3|inf |inf|inf| 0 


//Top Down DP  

class Solution {
private:
   int fmax(vector<int>& arr,int l,int h){
       int mx=INT_MIN;
       for(int i=l;i<=h;++i) mx=max(mx,arr[i]);
       return mx;
   }
    int helper(vector<int>& arr, int l, int r,vector<vector<int>> &dp)
    {
        if(l >= r)
            return 0;
        if(dp[l][r] != -1)
            return dp[l][r];

        int res = INT_MAX;
        for(int i=l;i<r;i++){
            int rootVal = fmax(arr, l, i) * fmax(arr, i+1, r);
            int nonLeafNodeFromLeftSubtree = helper(arr, l, i,dp);
            int nonLeafNodeFromRightSubtree = helper(arr, i+1, r,dp);
            res = min(res, rootVal+nonLeafNodeFromLeftSubtree
                            +nonLeafNodeFromRightSubtree);
        }
        
        dp[l][r] = res;
        return res;
    }
public:
    int mctFromLeafValues(vector<int>& arr) {
         int n = arr.size();
         vector<vector<int>> dp(n+1,vector<int> (n+1,-1));
         return helper(arr,0,n-1,dp);
    }
};

//Complexity Calculation --> n element * every combination of (l,r)[~ n* n]


//Bottom Up dp

int mctFromLeafValues(vector<int>& arr) {
     int n = arr.size();
     vector<vector<int>> dp(n+1,vector<int> (n+1,INT_MAX));
     for(int i=0;i<=n;i++) dp[i][i] = 0;

     for(int l=2;l<=n;l++){
         for(int i=0;i<=n-l;i++){
             int j = i + l - 1;
             for(int k=i;k<j;k++){
                 int rootVal = fmax(arr, i, k) * fmax(arr, k+1, j);
                 dp[i][j] = min(dp[i][j], rootVal + dp[i][k] + dp[k + 1][j]);
             }
         }
     }
     return dp[0][n - 1];
}



//2. Greedy approach // O(n ^ 2)
------------------------------------
// Approach -
// put big leaf nodes close to the root. So find smallest value in array ,
// use it & its smaller neighbour to build non-leaf node then delete them 
// (since it has smaller value than its neightbor,so it won't be used again)
// Repeat until a single nose left

// a[i]=parent,left=a[i-1],right=a[i+1] //i=index of min(a)
// res+=parent+left*right
// i>0&&i<n-1 //parent+left+right exist
// i==0 //parent+right exist
// i==n-1 //parent+left exist
// erase(a[i])


int mctFromLeafValues(vector<int>& arr) {
    int n = arr.size();
    vector<vector<int>> dp(n+1,vector<int> (n+1,-1));
    vector<int> A=arr;
    int res = 0;
    while(A.size() != 1){
        int mn=*min_element(A.begin(),A.end()); // n
        int minIndex = find(A.begin(),A.end(),mn)-A.begin(); //nlogn
        
        if(minIndex > 0 && minIndex < A.size()-1)
            res += A[minIndex] * min(A[minIndex-1], A[minIndex+1]);   
        
        else if(minIndex == 0)
            res += A[minIndex] * A[minIndex+1];
        
        else        //minIndex=n-1
            res += A[minIndex] * A[minIndex-1];
        
        A.erase(A.begin()+minIndex); //delete the current minimum value
    }
    return res;
}



//3. Monotonic stack approach // O(n)
--------------------------------------
// Can we optimize min finding ?

// for each leaf node in the array, when it becomes the minimum value in the 
// remaining array, its left and right neighbors will be the first bigger value 
// in the original array to its left and right. ---> monotonic stack



int mctFromLeafValues(vector<int>& arr) {
     int res=0;
     stack<int> stack;
     stack.push(INT_MAX);
     for (int num : arr){
         while (stack.top() <= num) {
              int mid = stack.top(); //left max
              stack.pop();
              res += mid * min(stack.top(), num); //left max & right max exist
         }
         stack.push(num);
      }
      //combined all leaf into non-leaf nodes
      while (stack.size() > 2){
          int val=stack.top(); //rootVal 
          stack.pop();
          res += val * stack.top(); //propagate upward
      }
          
      
      return res;
}