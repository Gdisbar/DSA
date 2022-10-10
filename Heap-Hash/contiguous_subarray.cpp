525. Contiguous Array
=======================
Given a binary array nums, return the maximum length of a contiguous 
subarray with an equal number of 0 and 1.

 

Example 1:

Input: nums = [0,1]
Output: 2
Explanation: [0, 1] is the longest contiguous subarray with an equal number of 
0 and 1.

Example 2:

Input: nums = [0,1,0]
Output: 2
Explanation: [0, 1] (or [1, 0]) is a longest contiguous subarray with equal 
number of 0 and 1.


//Brute : count # of zeros & ones using 2-loops --> mx=max(mx,j-i+1) if(zeros==ones)
//TC : n^2 , SC: 1

we replace 0 with -1 & now when we get a sum = 0 we got a new subarray , we calculate 
it''s lengths

i= 0 1 2 3 4 5 6 7 8 
a= 1 1 0 0 1 1 0 1 1 
s= 1 2 1 0 1 2 1 2 3
-------------------------
i= 0 1 2       3       4       5       6        7       8
mx=0 0 2=(2-0) 4=(3+1) 4=(4-0) 4=(5-1) 6=(6-0)  6=(7-1) 6
mp=<1,0>,<2,1>,<3,8>

//TC : n , SC : n

int findMaxLength(vector<int>& nums) {
        unordered_map<int, int> mp;
        int sum=0,mx = 0;
        for (int i=0;i<nums.size();++i) {
            if (nums[i]==0) nums[i]=-1;
            sum+=nums[i];
            if(sum==0){ // when we get 0 we store i-0+1 i.e from starting index upto i
                mx=max(mx,i+1);
            }else{
                if(mp.find(sum)!=mp.end()){
                    int idx=mp[sum];
                    mx=max(mx,i-idx);
                   // mp[sum]=idx;
                }
                else mp[sum]=i;
            }
        }
        return mx;
    }


696. Count Binary Substrings
==============================
Given a binary string s, return the number of non-empty substrings that have 
the same number of 0's and 1''s, and all the 0's and all the 1''s in these 
substrings are grouped consecutively.

Substrings that occur multiple times are counted the number of times they 
occur.

 

Example 1:

Input: s = "00110011"
Output: 6
Explanation: There are 6 substrings that have equal number of consecutive 1''s 
and 0's: "0011", "01", "1100", "10", "0011", and "01".
Notice that some of these substrings repeat and are counted the number of 
times they occur.
Also, "00110011" is not a valid substring because all the 0's (and 1''s) 
are not grouped together.

Example 2:

Input: s = "10101"
Output: 4
Explanation: There are 4 substrings: "10", "01", "10", "01" that have 
equal number of consecutive 1's and 0's.



Explanation

First, I count the number of 1 or 0 grouped consecutively.
For example "0110001111" will be [1, 2, 3, 4].

Second, for any possible substrings with 1 and 0 grouped consecutively, 
the number of valid substring will be the minimum number of 0 and 1.
For example "0001111", will be min(3, 4) = 3, ("01", "0011", "000111")

Complexity

Time O(N)
Space O(1)

int countBinarySubstrings(string s) {
        int cur = 1, pre = 0, res = 0;
        for (int i = 1; i < s.size(); i++) {
            if (s[i] == s[i - 1]) cur++;
            else {
                res += min(cur, pre);
                pre = cur;
                cur = 1;
            }
        }
        return res + min(cur, pre);
    }