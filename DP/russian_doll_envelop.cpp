354. Russian Doll Envelopes
=============================
// You are given a 2D array of integers envelopes where envelopes[i] = [wi, hi] 
// represents the width and the height of an envelope.

// One envelope can fit into another if and only if both the width and height of one 
// envelope are greater than the other envelope's width and height.

// Return the maximum number of envelopes you can Russian doll (i.e., put one inside 
// the other).

// Note: You cannot rotate an envelope.

 

// Example 1:

// Input: envelopes = [[5,4],[6,4],[6,7],[2,3]]
// Output: 3
// Explanation: The maximum number of envelopes you can Russian doll is 3 
// ([2,3] => [5,4] => [6,7]).

// Example 2:

// Input: envelopes = [[1,1],[1,1],[1,1]]
// Output: 1


//DP - n*n

int maxEnvelopes(vector<vector<int>>& env) {
        if(env.empty()) return 0;
        vector<int> dp(env.size(),1);
        sort(env.begin(),env.end());
        for(int i=0;i<env.size();++i){
            for(int j=0;j<i;++j){
                if(env[i][0]>env[j][0]&&env[i][1]>env[j][1])                                    
                	dp[i]=max(dp[i],dp[j]+1);
            }
        }
        return *max_element(dp.begin(),dp.end());
    }

For example, for input [[1,1],[2,2],[2,3],[2,4],[3,4]] we will get 
[[1,1],[2,2],[3,4]] instead of [[1,1],[2,4]]. by using (a[0]==b[0] && (a[1]>b[1]))

// nlogn
int maxEnvelopes(vector<vector<int>>& env) {
        if(env.empty()) return 0;
        sort(env.begin(),env.end(),[](vector<int> &a,vector<int> &b){ 
                return (a[0]<b[0] || (a[0]==b[0] && (a[1]>b[1])));
         });
        vector<int> dp;
        for(auto e:env){
        auto itr=lower_bound(dp.begin(),dp.end(),e[1]);
        //we got new higher element,so increase LIS length
        if(itr==dp.end()){
            dp.push_back(e[1]);
        }
        else{        //e[1]<*itr ,lower_bound returns the next element which doesn't compare less than value, since its either equal or greater, so replacing would be always correct.
            *itr=e[1]; //start new LIS
        }
    }
    return dp.size();
    
}