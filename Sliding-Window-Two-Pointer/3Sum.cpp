// Given an integer array nums, return all the triplets 
//[nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, 
//and nums[i] + nums[j] + nums[k] == 0.

// Notice that the solution set must not contain duplicate triplets.

 

// Example 1:

// Input: nums = [-1,0,1,2,-1,-4]
// Output: [[-1,-1,2],[-1,0,1]]

// Example 2:

// Input: nums = []
// Output: []

//===========================================================================================
// The key idea is the same as the TwoSum problem. When we fix the 1st number, 
//the 2nd and 3rd number can be found following the same reasoning as TwoSum.

// The only difference is that, the TwoSum problem of LEETCODE has a unique 
// solution. However, in ThreeSum, we have multiple duplicate solutions that can 
// be found. Most of the OLE errors happened here because you could''ve ended up 
// with a solution with so many duplicates.

// The naive solution for the duplicates will be using the STL methods like 
//below :

// std::sort(res.begin(), res.end());
// res.erase(unique(res.begin(), res.end()), res.end());

// But according to my submissions, this way will cause you double your time 
//consuming almostly.

// A better approach is that, to jump over the number which has been 
//scanned, no matter it is part of some solution or not.

// If the three numbers formed a solution, we can safely ignore all 
//the duplicates of them.

// We can do this to all the three numbers such that we can remove the 
//duplicates.


//===========================================================================================

vector<vector<int>> threeSum(vector<int>& nums) {
        sort(nums.begin(),nums.end());
        int n=nums.size();
        vector<vector<int>> res;
        for(int i=0;i<n-2;i++){
        	   //duplicate exist for 2 consecutive
               if(i>0 && (nums[i]==nums[i-1]) )continue;
               int l=i+1, r= n-1;
               while(l<r){
                   int sum =nums[i]+nums[l]+nums[r];              
                   if(sum<0) l++;
                   else if(sum>0)r--;
                   else {
                       res.push_back(vector<int>{nums[i],nums[l],nums[r]});
            //duplicate of number 2,roll front poninter to next unique number
                       while(l+1<r && nums[l]==nums[l+1])l++;
            //duplicate of number 3,roll front poninter to next unique number
                       while(l+1<r && nums[r]==nums[r-1]) r--;
                       l++; r--;
                   }
               }
        }
       
        return res;
    }


//===========================================================================================
// Without handeling duplicates
//===========================================================================================
// function to print triplets with 0 sum
void findTriplets(int arr[], int n)
{
    bool found = false;
 
    for (int i=0; i<n-1; i++)
    {
        // Find all pairs with sum equals to
        // "-arr[i]"
        unordered_set<int> s;
        for (int j=i+1; j<n; j++)
        {
            int x = -(arr[i] + arr[j]);
            if (s.find(x) != s.end())
            {
                printf("%d %d %d\n", x, arr[i], arr[j]);
                found = true;
            }
            else
                s.insert(arr[j]);
        }
    }
 
    if (found == false)
        cout << " No Triplet Found" << endl;
}
//===========================================================================================
