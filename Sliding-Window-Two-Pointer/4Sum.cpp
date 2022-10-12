// Given an array nums of n integers, return an array of all the 
//unique quadruplets [nums[a], nums[b], nums[c], nums[d]] such that:

//     0 <= a, b, c, d < n
//     a, b, c, and d are distinct.
//     nums[a] + nums[b] + nums[c] + nums[d] == target

// You may return the answer in any order.

 

// Example 1:

// Input: nums = [1,0,-1,0,-2,2], target = 0
// Output: [[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]

// Example 2:

// Input: nums = [2,2,2,2,2], target = 8
// Output: [[2,2,2,2]]


//===========================================================================================

//Using Extra space

// The function finds four
// elements with given sum X
void findFourElements(int arr[], int n, int X)
{
    // Store sums of all pairs
    // in a hash table
    unordered_map<int, pair<int, int> > mp;
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            mp[arr[i] + arr[j]] = { i, j };
 
    // Traverse through all pairs and search
    // for X - (current pair sum).
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            int sum = arr[i] + arr[j];
 
            // If X - sum is present in hash table,
            if (mp.find(X - sum) != mp.end()) {
 
                // Making sure that all elements are
                // distinct array elements and an element
                // is not considered more than once.
                pair<int, int> p = mp[X - sum];
                if (p.first != i && p.first != j
                    && p.second != i && p.second != j) {
                    cout << arr[i] << ", " << arr[j] << ", "
                         << arr[p.first] << ", "
                         << arr[p.second];
                    return;
                }
            }
        }
    }
}

//===========================================================================================
 vector<vector<int>> fourSum(vector<int>& nums, int target) {
        vector<vector<int>> ans;
        int n = nums.size();
        sort(nums.begin(),nums.end());
        for(int i = 0;i<n;i++){
            for(int j = i+1;j<n;j++){
                int remain = target - nums[i] - nums[j];
                int l = j+1;
                int h = n - 1;
                while(l<h){
                    if(remain>nums[l]+nums[h])
                        l++;
                    else if(remain<nums[l]+nums[h])
                        h--;
                    else{
                        vector<int> res = {nums[i],nums[j],nums[l],nums[h]};
                        ans.push_back(res);
                        while(l<h&&nums[l]==res[2])
                            l++;
                        while(l<h&&nums[h]==res[3])
                            h--;
                    }
                    
                }
                //outside j loop moves 1 step at a time but if we encounter j at 2
                //2 2 2 3 then next j should be at 3, not the next 2
                while(j+1<n&&nums[j]==nums[j+1])
                        j++;
            }
            //same like j to handle duplicates
            while(i+1<n&&nums[i]==nums[i+1])
                i++;
        }
        return ans;
    }