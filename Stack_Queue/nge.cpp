//pattern
vector<int> s; //  stores largest index
vector<pair<int,int>> p
// greater element on right ---> i=0 to n-1
//on left how many are smaller than current   ---> i=n-1 to 0
for(int i=0 to n-1 or i=n-1 to 0){ 
    if(!s.empty()&&a[s.back()]<a[i]){
        //when we need element
        ans[s.back()]=a[i] 
        //when we need to find span/width between current & next greater
        ans[s.back()]=i-s.bacK() //for i=0 to n-1
        ans[s.back()]=s.back()-i //for i=n-1 to 0
        //calculation of any other parameter
        s.pop_back()
    }
    s.push_back(i)//storing index 
    p.push_back(pair<int,int>)
}


503. Next Greater Element II
==============================
// Given a circular integer array nums (i.e., the next element of 
// nums[nums.length - 1] is nums[0]), return the next greater number
// for every element in nums.

// The next greater number of a number x is the first greater number to its 
// traversing-order next in the array, which means you could search circularly to 
// find its next greater number. If it doesn''t exist, return -1 for this number.

 

// Example 1:

// Input: nums = [1,2,1]
// Output: [2,-1,2]
// Explanation: The first 1''s next greater number is 2; 
// The number 2 can''t find next greater number. 
// The second 1''s next greater number needs to search circularly, 
// which is also 2.

// Example 2:

// Input: nums = [1,2,3,4,3]
// Output: [2,3,4,-1,4]

// TC : n*n , SC : 1 , 5% faster , 96% less memory

vector<int> nextGreaterElements(vector<int>& nums) {
        int n=nums.size();
        vector<int> ans(n,-1);
        for(int i=0;i<n;++i){
            for(int j=i+1;j<n;++j){
                if(nums[j]>nums[i]){
                    ans[i]=nums[j];
                    break;
                }
            }
            for(int j=0;j<i;++j){
                if(nums[j]>nums[i]){
                    ans[i]=nums[j];
                    break;
                }
            }
        }
        return ans;
    }

// or it can be done using a single loop , though it takes much more 
// than previous

for(int i=0;i<2*n;++i){
    for(int j=(i+1)%n;j%n!=i%n;++j){
        if(nums[j%n]>nums[i%n]){
            ans[i%n]=nums[j%n];
            break;
        }
    }
}

// TC : n , SC : n , 27% faster, 89% less memory

// Loop once, we can get the Next Greater Number of a normal array.
// Loop twice, we can get the Next Greater Number of a circular array

vector<int> nextGreaterElements(vector<int>& nums) {
        int n=nums.size();
        vector<int> s,ans(n,-1);
        for(int i=0;i<2*n;++i){
           while(!s.empty()&&nums[s.back()]<nums[i%n]){
               ans[s.back()]=nums[i%n];
               s.pop_back();
           }
           s.push_back(i%n);
        }
        return ans;
    }

739. Daily Temperatures
===========================
// Given an array of integers temperatures represents the daily temperatures, 
// return an array answer such that answer[i] is the number of days you have to 
// wait after the i-th day to get a warmer temperature. If there is no future day 
// for which this is possible, keep answer[i] == 0 instead.

 

// Example 1:

// Input: temperatures = [73,74,75,71,69,72,76,73]
// Output: [1,1,4,2,1,1,0,0]

// Example 2:

// Input: temperatures = [30,40,50,60]
// Output: [1,1,1,0]

// Example 3:

// Input: temperatures = [30,60,90]
// Output: [1,1,0]


//TC : n, SC :n, 87% faster,58% less memory

vector<int> dailyTemperatures(vector<int>& nums) {
        int n=nums.size();
        vector<int> s,ans(n,0);
        for(int i=0;i<n;++i){
            //int cnt=0;
            while(!s.empty()&&nums[s.back()]<nums[i]){
            //    cnt++; 
                ans[s.back()]=i-s.back();
                s.pop_back();
            }
            s.push_back(i);
        }
        return ans;
    }


901. Online Stock Span
=========================
// Design an algorithm that collects daily price quotes for some stock and 
// returns the span of that stock''s price for the current day.

// The span of the stock''s price today is defined as the maximum number of 
// consecutive days (starting from today and going backward) for which the stock 
// price was less than or equal to today''s price.

//     For example, if the price of a stock over the next 7 days were 
//     [100,80,60,70,60,75,85], then the stock spans would be [1,1,1,2,1,4,6].

// Implement the StockSpanner class:

// StockSpanner() Initializes the object of the class.
// int next(int price) Returns the span of the stock''s price given that 
// today''s  price is price.

 

// Example 1:

// Input
// ["StockSpanner", "next", "next", "next", "next", "next", "next", "next"]
// [[], [100], [80], [60], [70], [60], [75], [85]]
// Output
// [null, 1, 1, 1, 2, 1, 4, 6]

// Explanation
// StockSpanner stockSpanner = new StockSpanner();
// stockSpanner.next(100); // return 1
// stockSpanner.next(80);  // return 1
// stockSpanner.next(60);  // return 1
// stockSpanner.next(70);  // return 2
// stockSpanner.next(60);  // return 1
// stockSpanner.next(75);  // return 4, because the last 4 prices (including today's price of 75) were less than or equal to today's price.
// stockSpanner.next(85);  // return 6




// 71% faster, 62% less memory

class StockSpanner {
public:
//     StockSpanner() {
        
//     }
    vector<pair<int,int>> s; //price,span
    int next(int price) {
           int span=1;
            while(!s.empty()&&s.back().first<=price){
                span+=s.back().second;
                s.pop_back();
            }
            s.push_back(make_pair(price,span));
            return span;
    }
};

/**
 * Your StockSpanner object will be instantiated and called as such:
 * StockSpanner* obj = new StockSpanner();
 * int param_1 = obj->next(price);
 */
// vector form
vector<int> fnext(vector<int> price) {
        int n = price.size();
        vector<int> s,ans(n,1);
        for(int i=n-1;i>=0;--i){
            while(!s.empty()&&price[s.back()]<price[i]){
                ans[s.back()]=s.back()-i;
                s.pop_back();
            }
            s.push_back(i);
        }
        return ans;
    }