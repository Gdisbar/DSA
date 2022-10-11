322. Coin Change
=================
You are given an integer array coins representing coins of different 
denominations and an integer amount representing a total amount of money.

Return the fewest number of coins that you need to make up that amount. 
If that amount of money cannot be made up by any combination of the coins, 
return -1.

You may assume that you have an infinite number of each kind of coin.

 

Example 1:

Input: coins = [1,2,5], amount = 11
Output: 3
Explanation: 11 = 5 + 5 + 1

Example 2:

Input: coins = [2], amount = 3
Output: -1

Example 3:

Input: coins = [1], amount = 0
Output: 0

class Solution {
    int f(vector<int>& coins,int idx,int amount,vector<vector<int>> &dp){
        if(idx==0){
            if(amount%coins[idx]==0) return amount/coins[idx];
            else return INT_MAX;
        }
        if(dp[idx][amount]!=-1)return dp[idx][amount];
        int not_take=0+f(coins,idx-1,amount,dp);
        int take=INT_MAX;
        if(coins[idx]<=amount) 
            take=1+f(coins,idx,amount-coins[idx],dp);
        return dp[idx][amount]=min(take,not_take);
    }
public:
    int coinChange(vector<int>& coins, int amount) {
        int n = coins.size();
        vector<vector<int>> dp(n,vector<int>(amount+1,-1));
        int res = f(coins,n-1,amount,dp);
        if(res==INT_MAX) return -1;
        else return res;
        
    }
};

int coinChange(vector<int>& coins, int amount) {
        int n = coins.size();
        vector<vector<int>> dp(n,vector<int>(amount+1,0));
        for(int i=0;i<=amount;++i){
            if(i%coins[0]==0) dp[0][i]=i/coins[0];
            else dp[0][i]=1e9;
        }
        for(int i=1;i<n;++i){
            for(int j=0;j<=amount;++j){
                int not_take=0+dp[i-1][j];
                int take=INT_MAX;
                if(coins[i]<=j) 
                    take=1+dp[i][j-coins[i]];
                dp[i][j]=min(take,not_take);
            }
             
        }
        int res=dp[n-1][amount];
        if(res==1e9) return -1;
        else return res;
    }


int coinChange(vector<int>& coins, int amount) {
        int n = coins.size();
        vector<int> cur(amount+1,0),prev(amount+1,0);
        for(int i=0;i<=amount;++i){
            if(i%coins[0]==0) prev[i]=i/coins[0];
            else prev[i]=1e9;
        }
        for(int i=1;i<n;++i){
            for(int j=0;j<=amount;++j){
                int not_take=0+prev[j];
                int take=INT_MAX;
                if(coins[i]<=j) 
                    take=1+cur[j-coins[i]];
                cur[j]=min(take,not_take);
            }
             prev=cur;
        }
        int res=prev[amount];
        if(res==1e9) return -1;
        else return res;
    }

518. Coin Change 2
=====================
// You are given an integer array coins representing coins of different 
// denominations and an integer amount representing a total amount of money.

// Return the number of combinations that make up that amount. If that 
// amount of money cannot be made up by any combination of the coins, return 0.

// You may assume that you have an infinite number of each kind of coin.

// The answer is guaranteed to fit into a signed 32-bit integer.

 

// Example 1:

// Input: amount = 5, coins = [1,2,5]
// Output: 4
// Explanation: there are four ways to make up the amount:
// 5=5
// 5=2+2+1
// 5=2+1+1+1
// 5=1+1+1+1+1

// Example 2:

// Input: amount = 3, coins = [2]
// Output: 0
// Explanation: the amount of 3 cannot be made up just with coins of 2.

// Example 3:

// Input: amount = 10, coins = [10]
// Output: 1


    int change(int amount, vector<int>& coins) {
        int n = coins.size();
        vector<vector<int>> dp(n,vector<int>(amount+1,0));
        for(int i=0;i<=amount;++i){
            dp[0][i]=(i%coins[0]==0);
        }
        for(int i=1;i<n;++i){
            for(int j=0;j<=amount;++j){
                int not_take=dp[i-1][j];
                int take=0;
                if(coins[i]<=j) 
                    take=dp[i][j-coins[i]];
                dp[i][j]=take+not_take;
            }
             
        }
        return dp[n-1][amount];
    }


    int change(int amount, vector<int>& coins) {
        int n = coins.size();
        vector<int> prev(amount+1,0),cur(amount+1,0);
        //vector<vector<int>> dp(n,vector<int>(amount+1,0));
        for(int i=0;i<=amount;++i){
            prev[i]=(i%coins[0]==0);
        }
        for(int i=1;i<n;++i){
            for(int j=0;j<=amount;++j){
                int not_take=prev[j];
                int take=0;
                if(coins[i]<=j) 
                    take=cur[j-coins[i]];
                cur[j]=take+not_take;
            }
           prev=cur;  
        }
        return prev[amount];
    }

int change(int amount, vector<int>& coins) {
        vector<int> dp(amount+1,0);
        dp[0] = 1;
        for (int coin : coins) {
            for (int i = coin; i <= amount; i++) {
                dp[i] += dp[i-coin];
            }
        }
        return dp[amount];
    }   