Maximum Sum Non Adjacent Elements or 
======================================
198. House Robber
=====================
// You are a professional robber planning to rob houses along a street. 
// Each house has a certain amount of money stashed, the only constraint 
// stopping you from robbing each of them is that adjacent houses have security 
// systems connected and it will automatically contact the police if two adjacent 
// houses were broken into on the same night.

// Given an integer array nums representing the amount of money of each house, r
// eturn the maximum amount of money you can rob tonight without alerting the police.

 

// Example 1:

// Input: nums = [1,2,3,1]
// Output: 4
// Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
// Total amount you can rob = 1 + 3 = 4.

// Example 2:

// Input: nums = [2,7,9,3,1]
// Output: 12
// Explanation: Rob house 1 (money = 2), rob house 3 (money = 9) and rob house 5 (money = 1).
// Total amount you can rob = 2 + 9 + 1 = 12.


class Solution {
public:
    int rob(vector<int>& A, int i = 0) {
        return i < size(A) ? max(rob(A, i+1), A[i] + rob(A, i+2)) : 0;
    }
};
 
// MaxSum[i] = max(MaxSum[i-1] , MaxSum[i-2] + array[i]) //Basecase - 0,1,2 from 3 onwards use the formula

// Time and Space = O(n) | O(n).

// op = MaxSum = [7,10,19,19,28,33]


// Now let's see what we can optimize from the above solution.

//     We have to visit every element in the array so that will take O(n)
//     But space instead of storing entires array's maximum sum, We can store the 
//     max sum of the current element thus reducing space complexity from O(n) to O(1).

// We can have 2 variables - nonNeighbor and neighbor sum.
// We will still follow the same equation
// Equation =
// MaxSum[i] = max(MaxSum[i-1] , MaxSum[i-2] + array[I])
// or
// curSum = max(neighborSum , nonNeighborSum + array[i])

def maxSubsetSumNoAdjacent(array):
    if len(array) == 0:
		return 0
	elif len(array) == 1:
		return array[0]
	else:
		nonNeighborSum = array[0]
		neighborSum = max(array[0], array[1])
		
		for i in range(2,len(array)):
			curSum = max(neighborSum , nonNeighborSum + array[i])
			nonNeighborSum = neighborSum
			neighborSum = curSum
		return neighborSum


//DP
int rob(vector<int>& A) {
        vector<int> dp(size(A),-1);
        return rob(A, dp, 0);
    }
    int rob(vector<int>& A, vector<int>& dp, int i) {
        if(i >= size(A)) return 0;
        if(dp[i] != -1) return dp[i];
        return dp[i] = max(rob(A, dp, i+1), A[i] + rob(A, dp, i+2));
    }
// DP - tabulation

int rob(vector<int>& A) {
        if(size(A) == 1) return A[0];
        vector<int> dp(A);
        dp[1] = max(A[0], A[1]);
        for(int i = 2; i < size(A); i++)
            dp[i] = max(dp[i-1], A[i] + dp[i-2]);
        return dp.back();
    }

//  the value of current states (for ith element) depends upon only two states of 
//  the previous element. So instead of creating a 2D array, we can use only two 
//  variables to store the two states of the previous element.

// Say excl stores the value of the maximum subsequence sum till i-1 when arr[i-1] is 
// excluded and 
// incl stores the value of the maximum subsequence sum till i-1 when arr[i-1] is included.
// The value of excl for the current state( say excl_new) will be max(excl ,incl).
//  And the value of incl will be updated to excl + arr[i].
// Consider arr[] = {5,  5, 10, 100, 10, 5}

//     Initially at i = 0:  incl = 5, excl = 0

//     For i = 1: arr[i] = 5
//             => excl_new = 5
//             => incl = (excl + arr[i]) = 5
//             => excl = excl_new = 5

//     For i = 2: arr[i] = 10
//             => excl_new =  max(excl, incl) = 5
//             => incl =  (excl + arr[i]) = 15
//             => excl = excl_new = 5

//     For i = 3: arr[i] = 100
//             => excl_new =  max(excl, incl) = 15
//             => incl =  (excl + arr[i]) = 105
//             => excl = excl_new = 15

//     For i = 4: arr[i] = 10
//             => excl_new =  max(excl, incl) = 105
//             => incl =  (excl + arr[i]) = 25
//             => excl = excl_new = 105

//     For i = 5: arr[i] = 5
//             => excl_new =  max(excl, incl) = 105
//             => incl =  (excl + arr[i]) = 110
//             => excl = excl_new = 105

//     So, answer is max(incl, excl) =  110

// Follow the steps mentioned below to implement the above approach:

//     Initialize incl and excl with arr[0] and 0 respectively.
//     Iterate from i = 1 to N-1:
//         Update the values of incl and excl as mentioned above.
//     Return the maximum of incl and excl after the iteration is over as the answer.

// Function to return max sum such that 
// no two elements are adjacent 
int FindMaxSum(vector<int> arr, int n)
{
    int incl = arr[0]; //max subsequence sum till i-1 when arr[i-1] is included
    int excl = 0; //max subsequence sum till till i-1 when a[i-1] is excluded
    int excl_new;
  
    for (int i = 1; i < n; i++) {
        // Current max excluding i
        excl_new = max(incl, excl);
  
        // Current max including i
        incl = excl + arr[i];
        excl = excl_new;
    }
  
    // Return max of incl and excl
    return max(incl, excl);
}
  

515 · Paint House
===================
// There are a row of n houses, each house can be painted with one of the three 
// colors: red, blue or green. The cost of painting each house with a certain color is 
// different. You have to paint all the houses such that no two adjacent houses have 
// the same color, and you need to cost the least. Return the minimum cost.

// The cost of painting each house with a certain color is represented by a n x 3 
// cost matrix. For example, costs[0][0] is the cost of painting house 0 with color 
// red; costs[1][2] is the cost of painting house 1 with color green, and so on... 
// Find the minimum cost to paint all houses.

// Input: [[14,2,11],[11,14,5],[14,3,10]]

// Output: 10

// Explanation: Paint house 0 into blue, paint house 1 into green, paint house 2 
// into blue. Minimum cost: 2 + 5 + 3 = 10.


// Input: [[1,2,3],[1,4,6]]

// Output: 3

//Brute force  : find all possible ways for coloring --> no of color ^ no of element , here 3^N

// The idea is to find the minimum cost of painting the current house by any color 
// on the basis of the minimum cost of the other two colors of previously colored 
// houses. 

// dp[i]=dp[i-1]+min(cost[i][0],min(cost[i][1]),cost[i][2]) , won't work

// Function to find the minimum cost of
// coloring the houses such that no two
// adjacent houses has the same color

// TC : n , SC : n
int minCost(vector<vector<int> >& costs,
            int N)
{
    // Corner Case
    if (N == 0)
        return 0;
 
    // Auxiliary 2D dp array
    vector<vector<int> > dp(
        N, vector<int>(3, 0));
 
    // Base Case
    dp[0][0] = costs[0][0];
    dp[0][1] = costs[0][1];
    dp[0][2] = costs[0][2];
 
    for (int i = 1; i < N; i++) {
 
        // If current house is colored
        // with red the take min cost of
        // previous houses colored with
        // (blue and green)
        dp[i][0] = min(dp[i - 1][1],
                       dp[i - 1][2])
                   + costs[i][0];
 
        // If current house is colored
        // with blue the take min cost of
        // previous houses colored with
        // (red and green)
        dp[i][1] = min(dp[i - 1][0],
                       dp[i - 1][2])
                   + costs[i][1];
 
        // If current house is colored
        // with green the take min cost of
        // previous houses colored with
        // (red and blue)
        dp[i][2] = min(dp[i - 1][0],
                       dp[i - 1][1])
                   + costs[i][2];
    }
 
    // Print the min cost of the
    // last painted house
    cout << min(dp[N - 1][0],
                min(dp[N - 1][1],
                    dp[N - 1][2]));
}

//concise
int Solution::solve(vector<vector > &A)
{
    int n=A.size();
    vector<vector>dp(n+1,vector(3,0));
    for(int i=1;i<=n;i++)
    for(int j=0;j<3;j++) 
        dp[i][j]=min(dp[i-1][(j+1)%3],dp[i-1][(j+2)%3])+A[i-1][j];

    return min(dp[n][0],min(dp[n][1],dp[n][2]));

}

// recursive
 int mincost(vector<vector<int> > &A, int i, int j, int m, int n)
{
    if(i == m) return 1;
    if(m == 1) return min(A[i][1],min(A[i][0],A[i][2]));
    if(j == - 1)
    {
        return min(A[i][1],min(A[i][0],A[i][2])) + min(mincost(A,i+1,1,m,n),min(mincost(A,i+1,0,m,n),mincost(A,i+1,2,m,n)));
    }
    else
    {
        if(j == 2)
        {
            return min(A[i][1],A[i][0]) + min(mincost(A,i+1,1,m,n),mincost(A,i+1,0,m,n));
        }
        if(j == 1)
        {
            return min(A[i][2],A[i][0]) + min(mincost(A,i+1,2,m,n),mincost(A,i+1,0,m,n));
        }
        if(j == 0)
        {
            return min(A[i][1],A[i][2]) + min(mincost(A,i+1,1,m,n),mincost(A,i+1,2,m,n));
        }
    }
}

int Solution::solve(vector<vector<int> > &A)
{
    int m = A.size();
    int n = A[0].size();
    return mincost(A,0,-1,m,n);
}
516 · Paint House II
=======================
// There are a row of n houses, each house can be painted with one of the k colors. 
// The cost of painting each house with a certain color is different. You have to 
// paint all the houses such that no two adjacent houses have the same color.

// The cost of painting each house with a certain color is represented by a n x k 
// cost matrix. For example, costs[0][0] is the cost of painting house 0 with color 0; 
// costs[1][2] is the cost of painting house 1 with color 2, and so on... 
// Find the minimum cost to paint all houses.

// Input:

// costs = [[14,2,11],[11,14,5],[14,3,10]]

// Output: 10

// Explanation:

// The three house use color [1,2,1] for each house. The total cost is 10.


// Input:

// costs = [[5]]

// Output: 5

// Explanation:

// There is only one color and one house.

// TC : n*k*k , SC : n*k

public int minCostII(int[][] costs) {
        if (costs == null || costs.length == 0) {
            return 0;
        }
         
        int n = costs.length;
        int k = costs[0].length;
         
        // dp[i][j] means the min cost painting for house i, with color j
        int[][] dp = new int[n][k];
         
        // Initialization
        for (int i = 0; i < k; i++) {
            dp[0][i] = costs[0][i];
        }
         
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < k; j++) {
                dp[i][j] = Integer.MAX_VALUE;
                for (int m = 0; m < k; m++) {
                    if (m != j) {
                        dp[i][j] = Math.min(dp[i - 1][m] + costs[i][j], dp[i][j]);
                    }
                }
            }
        }
         
        // Final state
        int minCost = Integer.MAX_VALUE;
        for (int i = 0; i < k; i++) {
            minCost = Math.min(minCost, dp[n - 1][i]);
        }
         
        return minCost;
    }


// TC : n*k

public int minCostII(int[][] costs) {
        if (costs == null || costs.length == 0) {
            return 0;
        }
         
        int n = costs.length;
        int k = costs[0].length;
         
        // dp[j] means the min cost for color j
        int[] dp1 = new int[k];
        int[] dp2 = new int[k];
         
        // Initialization
        for (int i = 0; i < k; i++) {
            dp1[i] = costs[0][i];
        }
         
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < k; j++) {
                dp2[j] = Integer.MAX_VALUE;
                for (int m = 0; m < k; m++) {
                    if (m != j) {
                        dp2[j] = Math.min(dp1[m] + costs[i][j], dp2[j]);
                    }
                }
            }
             
            for (int j = 0; j < k; j++) {
                dp1[j] = dp2[j];
            }
        }
         
        // Final state
        int minCost = Integer.MAX_VALUE;
        for (int i = 0; i < k; i++) {
            minCost = Math.min(minCost, dp1[i]);
        }
         
        return minCost;
    }