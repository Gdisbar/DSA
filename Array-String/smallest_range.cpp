632. Smallest Range Covering Elements from K Lists
====================================================
// You have k lists of sorted integers in non-decreasing order. Find the smallest 
// range that includes at least one number from each of the k lists.

// We define the range [a, b] is smaller than range [c, d] if b - a < d - c or 
// a < c if b - a == d - c.

 

// Example 1:

// Input: nums = [[4,10,15,24,26],[0,9,12,20],[5,18,22,30]]
// Output: [20,24]
// Explanation: 
// List 1: [4, 10, 15, 24,26], 24 is in range [20,24].
// List 2: [0, 9, 12, 20], 20 is in range [20,24].
// List 3: [5, 18, 22, 30], 22 is in range [20,24].

// Example 2:

// Input: nums = [[1,2,3],[1,2,3],[1,2,3]]
// Output: [1,1]

// nums = [[4,10,15,24,26],[0,9,12,20],[5,18,22,30]]
// after merge we get : 
// [(0, 1), (4, 0), (5, 2), (9, 1), (10, 0), (12, 1), (15, 0), (18, 2), 
// 	(20, 1), (22, 2), (24, 0), (26, 0), (30, 2)]

// and see only group, it''s
// [1, 0, 2, 1, 0, 1, 0, 2, 1, 2, 0, 0, 2]

// we can slide window by group when current groups satifies condition and recoard 
// min range.

// [1 0 2] 2 1 0 1 0 2 1 2 0 0 2    [0, 5]
// 1 [0 2 1] 1 0 1 0 2 1 2 0 0 2    [0, 5]
// 1 0 [2 1 0] 0 1 0 2 1 2 0 0 2    [0, 5]
// 1 0 [2 1 0 1] 1 0 2 1 2 0 0 2    [0, 5]
// 1 0 [2 1 0 1 0] 0 2 1 2 0 0 2    [0, 5]
// 1 0 2 1 0 [1 0 2] 2 1 2 0 0 2    [0, 5]
// 1 0 2 1 0 1 [0 2 1] 1 2 0 0 2    [0, 5]
// 1 0 2 1 0 1 [0 2 1 2] 2 0 0 2    [0, 5]
// 1 0 2 1 0 1 0 2 [1 2 0] 0 0 2    [20, 24]
// 1 0 2 1 0 1 0 2 [1 2 0 0] 0 2    [20, 24]
// 1 0 2 1 0 1 0 2 [1 2 0 0 2] 2    [20, 24]

// nlogn+n

vector<int> smallestRange(vector<vector<int>>& nums){
	vector<pair<int,int>> a; //([],k)
	for(int k =0;k<nums.size();++k){
		for(auto x : nums[k])
			a.push_back({x,k});
	}
	sort(a.begin(),a.end());
	vector<int> ans;
	unordered_map<int,int> mp;
    int i=0,k=0;
	for(int j=0;j<a.size();++j){
		if(!mp[a[j].second]++) 
			++k;
		if(k==nums.size()){
			while(mp[a[i].second]>1)
				--mp[a[i++].second];
			if(ans.empty()||ans[1]-ans[0]>a[j].first-a[i].first){
				ans=vector<int>{a[i].first,a[j].first};
			}
		}
	}
	return ans;
}

// We keep two pointer left and right. We update left pointer by left++ only when 
// the current window contains K kinds of list-id (In this case we need to make the 
// window shorter since it already contains K kinds of list-id). And let right 
// pointer be the current number index we are visiting.

class Solution(object):
    def smallestRange(self, nums: List[List[int]]) -> List[int]:
        """
        :type nums: List[List[int]]
        :rtype: List[int]
        """
        d = []
        K = len(nums)
        count = collections.defaultdict(int)
        for i, num in enumerate(nums):
            for n in num:
                d.append([n, i])
        d.sort(key=lambda x: x[0])
        res = []
        left = 0
        for right, n in enumerate(d):
            count[n[1]] += 1
            while len(count)==K:
                if not res or d[right][0]-d[left][0]<res[1]-res[0]:
                    res = [d[left][0], d[right][0]]
                count[d[left][1]] -= 1
                if count[d[left][1]]==0:
                    del count[d[left][1]]
                left += 1
        return res

// using priority queue n*logK

struct Item {
    int val; // 
    int r; // index of list
    int c; // index of element in a particular list
    
    Item(int val, int r, int c): val(val), r(r), c(c) {
    }
};

struct Comp { //sort according to decreasing order
    bool operator() (const Item& it1, const Item& it2) {
        return it2.val < it1.val;
    }
};

class Solution {
public:
    vector<int> smallestRange(vector<vector<int>>& nums) {
        priority_queue<Item, vector<Item>, Comp> pq; //min heap
        
        int high = numeric_limits<int>::min();
        int n = nums.size();
        // store all the 1st elements from all the lists
        for (int i = 0; i < n; ++i) {
            pq.push(Item(nums[i][0], i, 0));
            high = max(high , nums[i][0]);
        }
        int low = pq.top().val;
        
        vector<int> res{low, high};
        //loop until any of the sub-list is exhausted ,  
        //if we've traversed any one of the sub-list then pq.size()<n
        while (pq.size() == (size_t)n) {
            auto it = pq.top();
            pq.pop();
            //go to next element of current sub-list until this list is exhaused
            if ((size_t)it.c + 1 < nums[it.r].size()) {
            	//next element of current sub-list : nums[it.r][it.c + 1]
                pq.push(Item(nums[it.r][it.c + 1], it.r, it.c + 1));
                high = max(high, nums[it.r][it.c + 1]);
                low = pq.top().val;
                if (high - low < res[1] - res[0]) {
                    res[0] = low;
                    res[1] = high;
                }
            }
        }
        
        return res;
    }
};


def smallestRange(self, A):
    pq = [(row[0], i, 0) for i, row in enumerate(A)]
    heapq.heapify(pq)
    
    ans = -1e9, 1e9
    right = max(row[0] for row in A)
    while pq:
        left, i, j = heapq.heappop(pq)
        if right - left < ans[1] - ans[0]:
            ans = left, right
        if j + 1 == len(A[i]):
            return ans
        v = A[i][j+1]
        right = max(right, v)
        heapq.heappush(pq, (v, i, j+1))