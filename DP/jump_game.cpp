45. Jump Game II
====================
// Given an array of non-negative integers nums, you are initially positioned 
// at the first index of the array.
// Each element in the array represents your maximum jump length at that position.
// Your goal is to reach the last index in the minimum number of jumps.
// You can assume that you can always reach the last index.


// Example 1:

// Input: nums = [2,3,1,1,4]
// Output: 2
// Explanation: The minimum number of jumps to reach the last index is 2. 
// Jump 1 step from index 0 to 1, then 3 steps to the last index.

// Example 2:

// Input: nums = [2,3,0,1,4]
// Output: 2


int jump(vector<int>& nums) {
        int n = nums.size(),jump=0,end=0,far=0;
        for(int i=0;i<n-1;++i){
            far=max(far,i+nums[i]);
            if(i==end){
                jump++;
                end=far;
                if(end>=n-1) break;
            }
        }
        return jump;
    }


// using bfs

// I try to change this problem to a BFS problem, where nodes in level i are 
// all the nodes that can be reached in (i-1)-th jump. 

//2 3 1 1 4 

// 2||
// 3 1||
// 1 4 ||

// clearly, the minimum jump of 4 is 2 since 4 is in level 3. my ac code.

 int jump(int A[], int n) {
	 if(n<2)return 0;
	 int level=0,currentMax=0,i=0,nextMax=0;

	 while(currentMax-i+1>0){		//nodes count of current level>0
		 level++;
		 for(;i<=currentMax;i++){	//traverse current level 
		 	//and update the max reach of next level
			nextMax=max(nextMax,A[i]+i);
			// if last element is in level+1,  then the min jump=level 
			if(nextMax>=n-1)return level;   
		 }
		 currentMax=nextMax;
	 }
	 return 0;
 }