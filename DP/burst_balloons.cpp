312. Burst Balloons
=====================
// You are given n balloons, indexed from 0 to n - 1. Each balloon is 
// painted with a number on it represented by an array nums. You are asked to 
// burst all the balloons.

// If you burst the ith balloon, you will get nums[i - 1] * nums[i] * nums[i + 1] 
// coins. If i - 1 or i + 1 goes out of bounds of the array, then treat it as if 
// there is a balloon with a 1 painted on it.

// Return the maximum coins you can collect by bursting the balloons wisely.

 

// Example 1:

// Input: nums = [3,1,5,8]
// Output: 167
// Explanation:
// nums = [3,1,5,8] --> [3,5,8] --> [3,8] --> [8] --> []
// coins =  3*1*5   +   3*5*8   +  1*3*8  + 1*8*1 = 167
// coins = 15 + 120 + 24 + 8 


// Example 2:

// Input: nums = [1,5]
// Output: 10



// Basic - We have n balloons to burst, which mean we have n steps in the game. 
// In the i-th step we have n-i balloons to burst, i = 0~n-1. Therefore we are 
// looking at an algorithm of O(n!). Well, it is slow, probably works 
// for n < 12 only.

// Greedy - when we burst a balloon for cost we look for left & right, what if
// they're adjacent i.e even no of balloons 

// DP- The coins you get for a balloon does not depend on the balloons already  
// bursted.Therefore instead of divide the problem by the first balloon to burst, 
// we divide the problem by the last balloon to burst. So we know the adjacent 
// element beforehand + for 1st & last balloon cost we can use 1 

// we don't care about 0's as they don't give any coin

// TC : n^3
// SC : n^2


// recursion
int burst(int[][] memo, int[] nums, int left, int right) {
    if (left + 1 == right) return 0;
    if (memo[left][right] > 0) return memo[left][right];
    int ans = 0;
    for (int i = left + 1; i < right; ++i)
        ans = Math.max(ans, nums[left] * nums[i] * nums[right] 
        + burst(memo, nums, left, i) + burst(memo, nums, i, right));
    memo[left][right] = ans;
    return ans;
}
public int maxCoins(int[] iNums) {
    int[] nums = new int[iNums.length + 2];
    int n = 1;
    for (int x : iNums) 
        if (x > 0) 
            nums[n++] = x;
    nums[0] = nums[n++] = 1;


    int[][] memo = new int[n][n];
    return burst(memo, nums, 0, n - 1);
}


// 12 ms
nums = [3,1,5,8]
i =     0 1 2 3
nums = [3,1,5,8] --> [3,5,8] --> [3,8] --> [8] --> []
coins =  3*1*5   +   3*5*8   +  1*3*8  + 1*8*1 = 167

dp[0][2]=1*3*1+0+0
dp[1][3]=3*1*5+0+0
dp[2][4]=5*8*1+0+0
dp[0][3]=1*3*1+0+dp[1][3]
dp[0][3]=5+dp[0][2]+0 //won't be accepted as dp[0][3]=30 better

//dp[l][r] = nums[l]*nums[i]*nums[r]+dp[l][i]+dp[i][r]
0 0 3 30 159 167 
0 0 0 15 135 159 
0 0 0 0 40 48 
0 0 0 0 0 40 
0 0 0 0 0 0 

int maxCoinsDP(vector<int> &iNums) {
    int nums[iNums.size() + 2];
    int n = 1;
    for (int x : iNums) if (x > 0) nums[n++] = x;
    nums[0] = nums[n++] = 1;

    //dp[l][r] = sum of all coins (for bursting) from l to r
    vector<vector<int>> dp(n,vector<int>(n));
    // k = gap between l & r atleast 2 - according to problem
    for (int k = 2; k < n; ++k) {
        for (int l = 0; l < n - k; ++l){
            int r = l + k;
            for (int i = l + 1; i < r; ++i){
                int x = nums[l]*nums[i]*nums[r]+dp[l][i]+dp[i][r];
                dp[l][r]=max(dp[l][r],x);
            }
        }      
    }
    // for(int i=0;i<n;++i){
    //     for(int j=0;j<n;++j)
    //         cout<<dp[i][j]<<" ";
    //     cout<<endl;
    // }
    return dp[0][n - 1];
}
// 16 ms