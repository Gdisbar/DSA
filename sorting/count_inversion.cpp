315. Count of Smaller Numbers After Self 
==============================================

// You are given an integer array nums and you have to return a new counts array. 
// The counts array has the property where counts[i] is the number of smaller 
// elements to the right of nums[i].



// Example 1:

// Input: nums = [5,2,6,1]
// Output: [2,1,1,0]
// Explanation:
// To the right of 5 there are 2 smaller elements (2 and 1).
// To the right of 2 there is only 1 smaller element (1).
// To the right of 6 there is 1 smaller element (1).
// To the right of 1 there is 0 smaller element.

// Example 2:

// Input: nums = [-1]
// Output: [0]

// Example 3:

// Input: nums = [-1,-1]
// Output: [0,0]


// //Brute Force
// //using 2 loops find & store lesser elements on right

// i=   [0, 1, 2, 3,4, 5,6,7]
// nums=[5, 2, 6, 3,2, 1,2,1]
// tmp=[-1,-1,-1,-1,1,-1,4,5]
// res=[ 6, 2, 5, 4,2, 0,1,0]
// cnt 0 res[tmp[7]] 0
// cnt 1 res[tmp[6]] 0
// cnt 2 res[tmp[4]] 0

//TLE 
    vector<int> countSmaller(vector<int>& nums) {
        vector<int> tmp(nums.size(),-1);
        unordered_map<int,int> mp;
        for(int i=0;i<nums.size();++i){
            if(mp.find(nums[i])!=mp.end()){
                tmp[i]=mp[nums[i]];
            }
            mp[nums[i]]=i;
        }

        vector<int> res(nums.size());
        for(int i=nums.size()-1;i>=0;--i){
            int cnt=0;
            if(tmp[i]==-1){
                for(int j=i;j<nums.size();++j)
                    if(nums[i]>nums[j]) cnt++;
                res[i]=cnt;
            }
            else{
                for(int j=i;j<nums.size();++j)
                    if(nums[i]>nums[j]) cnt++;
                res[i]=cnt+res[tmp[i]]; //storing previous result (memorization)
            
            }
        }
        return res;
    }


//TLE , BST solution suppoesed to be n*log(n)

// class Solution {
// private:
//         struct Node {
//         int val;
//         int c; // / Number of current value.
//         Node * left, * right;
//         int leftCount; // Number of left child Node for current node.
//         Node(int x):val(x), left(nullptr), right(nullptr), leftCount(0), c(0) {};
//         int addNode(int x) {return addNode(this,x,0);}
//         int addNode(Node* node, int v, int c){
//             if (v<=node->val) {
//                 node->leftCount++;
//                 if(node->left==nullptr) {
//                     node->left = new Node(v);
//                     return c;
//                 } else return addNode(node->left, v, c);
//             } else {
//                 if(node->right==nullptr){
//                     node->right = new Node(v);
//                     return c+(1+node->leftCount); //right subtree > left subtree + root
//                 } else return addNode(node->right, v, c+1+node->leftCount);
//             }
//         }
//     };
// public:
//     vector<int> countSmaller(vector<int>& nums) {
//         int n = nums.size();
//         vector<int> res(n,0);
//         Node *root;
//         if(n>0){
//             root = new Node(nums[n-1]);
//             for(int i=n-2;i>=0;i--)
//                 res[i] = root->addNode(nums[i]);
//         }
//         return res;
//     }
// };

// using ordered_set will give similar result like Segment-tree
// By default order_set can't handle duplicates,use pair for that

#include <ext/pb_ds/assoc_container.hpp> // Common file 
#include <ext/pb_ds/tree_policy.hpp> 
using namespace __gnu_pbds; 

typedef tree<pair<int,int>, null_type, less<pair<int,int>>, rb_tree_tag, 
             tree_order_statistics_node_update> pbds;
class Solution {
public:
    vector<int> countSmaller(vector<int>& nums) {
        pbds st;
        vector<int> res(nums.size());
        for(int i=nums.size()-1;i>=0;i--){
            res[i]=st.order_of_key({nums[i],-1});
            st.insert({nums[i],i});
        }
        return res;
    }
};

//best solution , Segment-tree , 84% faster , 86% less memory

    // BIT Update
    void update(int ind,vector<int>& BIT){
        while(ind<=BIT.size()){
            BIT[ind]++;
            ind += ind & -ind;
        }
    }
 
    // BIT answer
    int answer(int ind,vector<int>& BIT){
        int ans = 0;
        while(ind>0){
            ans+=BIT[ind];
            ind -= ind & -ind;
        }
        return ans;
    }
    vector<int> countSmaller(vector<int>& nums) {
        vector<int> ans(nums.size(),0);
        vector<int> BIT(20005,0); // Binary Indexed Tree Array
        for(int i=0;i<nums.size();i++){
            nums[i]+=10001; // Make all Numbers positive
        }
        for(int i=nums.size()-1;i>=0;i--){
            ans[i] = answer(nums[i]-1,BIT); // get answer
            update(nums[i],BIT); // update entry in BIT Array
        }
        return ans;
    }

// merge-sort approach

class Solution {
private:
    void merge_sort(int start, int end, vector<pair<int, int>>& nums, 
        vector<int>& indices, vector<pair<int, int>>& temp) {
        
        if(start >= end) return;
        
        int mid = start + (end - start) / 2;
        
        merge_sort(start, mid, nums, indices, temp);
        merge_sort(mid + 1, end, nums, indices, temp);
        
        int left = start, right = mid + 1;
        int idx = start;
        int nRightLessThanLeft = 0;
        while(left <= mid and right <= end) {
            if(nums[left] < nums[right]) {
                indices[nums[left].second] += nRightLessThanLeft;
                temp[idx++] = nums[left++];
            } else if(nums[left] > nums[right]) {
                temp[idx++] = nums[right++];
                nRightLessThanLeft++;
            } else
                left++, right++;
        }
        
        while(left <= mid) {
            indices[nums[left].second] += nRightLessThanLeft;
            temp[idx++] = nums[left++];
        }
        
        while(right <= end)
            temp[idx++] = nums[right++];
        
        for(int i=start; i<=end; i++)
            nums[i] = temp[i];
    }
    
public:
    vector<int> countSmaller(vector<int>& nums) {
        int n = nums.size();
        vector<pair<int, int>> new_nums; // {num, original_idx}
        vector<int> indices(n, 0);
        vector<pair<int, int>> temp;
        
        for(int i=0; i<n; i++) {
            new_nums.push_back({nums[i], i});
            temp.push_back({nums[i], i});
        }
        
        merge_sort(0, n-1, new_nums, indices, temp);
        
        return indices;
    }
};

315. Count of Smaller Numbers After Self
============================================================================

// Return the number of j''s such that i < j and nums[j] < nums[i].

    #define iterator vector<vector<int>>::iterator

    void sort_count(iterator l, iterator r, vector<int>& count) {
        if (r - l <= 1) return;
        iterator m = l + (r - l) / 2;
        sort_count(l, m, count);
        sort_count(m, r, count);
        for (iterator i = l, j = m; i < m; i++) {
            while (j < r && (*i)[0] > (*j)[0]) j++;
            count[(*i)[1]] += j - m; // add the number of valid "j"s to the indices of *i
        }
        inplace_merge(l, m, r);
    }
    vector<int> countSmaller(vector<int>& nums) {
        vector<vector<int>> hold;
        int n = nums.size();
        for (int i = 0; i < n; ++i) 
            hold.push_back(vector<int>({nums[i], i})); // "zip" the nums with their indices
        vector<int> count(n, 0);
        sort_count(hold.begin(), hold.end(), count);
        return count;
    }

493. Reverse Pairs 
================================
// Return the number of reverse pairs s.t. i < j and nums[i] > 2*nums[j].

    int sort_count(vector<int>::iterator begin, vector<int>::iterator end) {
        if (end - begin <= 1) return 0;
        vector<int>::iterator middle = begin + (end  - begin) / 2;
        int count = 0;
        if (begin < middle) count += sort_count(begin, middle);
        if (middle < end) count += sort_count(middle, end);
        vector<int>::iterator i, j;
        for (i = begin, j = middle; i < middle; ++i) { // double pointers trick
            while (j < end && *i > 2L * *j) {
                j++;
            }
            count += j - middle;
        }
        inplace_merge(begin, middle, end);
        return count;
    }
    int reversePairs(vector<int>& nums) {
        return sort_count(nums.begin(), nums.end());
    }

327. Count of Range Sum
====================================
 // Return the number of range sums that lie in [lower, upper] inclusive-inclusive. 
 // Let prefix-array sum be sums[0...n+1], the problem is to find pairs of i and j 
 // such that lower <= sums[j] - sums[i] <= upper.

    int countRangeSum(vector<int>& nums, int lower, int upper) {
        int n = nums.size();
        vector<long> sums(n + 1, 0);
        for (int i = 0; i < n; ++i) sums[i + 1] = sums[i] + nums[i];
        return sort_count(sums, 0, n + 1, lower, upper);
    }
    
    int sort_count(vector<long>& sums, int l, int r, int lower, int upper) {
        if (r - l <= 1) return 0;
        int m = (l + r) / 2, i, j1, j2;
        int count = sort_count(sums, l, m, lower, upper) + sort_count(sums, m, r, lower, upper);
        for (i = l, j1 = j2 = m; i < m; ++i) { 
            // we have two j pointers now and one i pointer, but still linear time
            while (j1 < r && sums[j1] - sums[i] < lower) j1++; 
            while (j2 < r && sums[j2] - sums[i] <= upper) j2++;
            count += j2 - j1;
        }
        inplace_merge(sums.begin() + l, sums.begin() + m, sums.begin() + r);
        return count;
    }
