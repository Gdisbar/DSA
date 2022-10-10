875. Koko Eating Bananas
===========================
Koko loves to eat bananas. There are n piles of bananas, the i-th pile has piles[i] 
bananas. The guards have gone and will come back in h hours.

Koko can decide her bananas-per-hour eating speed of k. Each hour, she chooses 
some pile of bananas and eats k bananas from that pile. If the pile has less than 
k bananas, she eats all of them instead and will not eat any more bananas during 
this hour.

Koko likes to eat slowly but still wants to finish eating all the bananas before 
the guards return.

Return the minimum integer k such that she can eat all the bananas within h hours.

 

Example 1:

Input: piles = [3,6,7,11], h = 8
Output: 4

Example 2:

Input: piles = [30,11,23,4,20], h = 5
Output: 30

Example 3:

Input: piles = [30,11,23,4,20], h = 6
Output: 23

 // for unknown reason giving TLE --> can't do the function call + ceil 

class Solution {
private:
    bool ispossible(vector<int>& piles, int m,int h){
            int cnt=0;
            for(int i=0;i<piles.size();++i){
                cnt+=ceil(piles[i]/m);
                if(piles[i]%m) cnt++;
            }
            return cnt==h;
        }
public:
    int minEatingSpeed(vector<int>& piles, int h) {    
       int l =1;
       int r=*max_element(piles.begin(),piles.end());
       int ans=0;
       if(h==piles.size()){return h;}
       while(l<=r){
           int m=l+(h-l)/2;
           if(ispossible(piles,m,h)){
               ans=m;
               h=m-1;
           }
           else l=m+1;
       }
       return ans; 
    }
};

Input : [312884470] 312884469 // if we use l = min(piles)
Output : 312884470
Expected : 2

// 47% faster, 77% less memory ---> 97% faster ,25% less memory for changing m=(l+r)/2 to m=l+(r-l)/2 
// TC : N*log(max(P))
int minEatingSpeed(vector<int>& piles, int h) {
        int l = 1, r = *max_element(piles.begin(),piles.end());
        while (l < r) {
            int m = l+(r-l)/2, total = 0;
            for (int p : piles)
                total += (p + m - 1) / m; //same as ceil(p/m),but here this one works
            if (total > h)
                l = m + 1;
            else           //total<=h ,we've found less value now look for lesser
                r = m;
        }
        return l;
    }