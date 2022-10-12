Given an integer array nums, return the number of reverse pairs in the array.

A reverse pair is a pair (i, j) where 0 <= i < j < nums.length and 
nums[i] > 2 * nums[j].

 

Example 1:

Input: nums = [1,3,2,3,1]
Output: 2

Example 2:

Input: nums = [2,4,3,5,1]
Output: 3

//===========================================================================================

// 315. Count of Smaller Numbers After Self
// 327. Count of Range Sum

// we can reverse the direction for insert and search in BIT
// so what we get is always the number greater than query values by using 
  single search method.

 int sort_and_count(vector<int>::iterator begin, vector<int>::iterator end) {
        if (end - begin <= 1)
            return 0;
        auto mid = begin + (end - begin) / 2;
        int count = sort_and_count(begin, mid) + sort_and_count(mid, end);
        for (auto i = begin, j = mid; i != mid; ++i) {
            while (j != end and *i > 2L * *j)
                ++j;
            count += j - mid;
        }
        inplace_merge(begin, mid, end);
        return count;
    }

    int reversePairs(vector<int>& nums) {
        return sort_and_count(nums.begin(), nums.end());
    }


//===========================================================================================
//TC -> nlogn + n + n
//SC -> n

int Merge(vector<int>&nums,int l,int m,int h){
	int cnt = 0;
	int j = m + 1;
	for(int i = l;i<=m;i++){
		while(j<=h&&nums[i]>2LL * nums[j]){
			j++;
		}
		cnt += j - m -1;
	}
	vector<int> tmp;
	int left = l,right = m + 1;
	while(left<=m&&right<=h){
		if(nums[left]<=nums[right])
			tmp.push_back(nums[left++]);
		else
			tmp.push_back(nums[right++]);
	}
	while(left<=m)
		tmp.push_back(nums[left++]);
	while(right<=h){
		tmp.push_back(nums[right++]);
		//cnt += right - m - 1;
	}
	for(int i = l;i<=h;i++)
		nums[i]=tmp[i-l];
	return cnt;
}
int MergeSort(vector<int>&nums,int l,int h){
	if(l>=h)return 0;
	int m = (h+l)/2;
	int inv = MergeSort(nums,l,m);
	inv += MergeSort(nums,m+1,h);
	inv += Merge(nums,l,m,h);
	return inv;
}
int reversePairs(vector<int>& nums) {
    return MergeSort(nums,0,nums.size()-1);
}