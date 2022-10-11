238. Product of Array Except Self
===================================
Given an integer array nums, return an array answer such that answer[i] is equal 
to the product of all the elements of nums except nums[i].

The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.

You must write an algorithm that runs in O(n) time and without using the division 
operation.

 

Example 1:

Input: nums = [1,2,3,4]
Output: [24,12,8,6]

left=  [1 2 6 24] 
right= [24 24 12 4] 

Example 2:

Input: nums = [-1,1,0,-3,3]
Output: [0,0,9,0,0]

left=  [-1 -1 0 0 0] 
right= [0 0 0 -9 3]


vector<int> productExceptSelf(vector<int>& nums) {
        int n=nums.size();
        vector<int> left(n),right(n),ans(n,1);
    
        for(int i = 0;i<n;i++){
            if(i==0) left[i]=nums[i];
            else left[i]=left[i-1]*nums[i];
           
        }
      
        for(int i = n-1;i>=0;i--){
            if(i==n-1) right[i]=nums[i];
            else right[i]=right[i+1]*nums[i];
            
        }

        ans[0]=right[1],ans[n-1]=left[n-2];
        for(int i = 1;i<n-1;i++){
            // if(i==0) ans[i]=right[i+1];
            // if(i==n-1) ans[i]=left[i-1];
            ans[i]=left[i-1]*right[i+1];
        }
        return ans;
    }


//Faster & concise

vector<int> productExceptSelf(vector<int>& nums) {
    int n = nums.size();
    int prefix = 1, suffix = 1;
    vector<int> prod(n,1);
    for (int i = 0;i<n;i++) {
        prod[i]*=prefix;
        prefix*=nums[i];
        prod[n-i-1]*=suffix;
        suffix*=nums[n-i-1];
    }
    return prod;
}