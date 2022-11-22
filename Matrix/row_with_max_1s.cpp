Row with max 1s
================
Given a boolean 2D array of r x c dimensions where each row is sorted. 
Find the 0-based index of the first row that has the maximum number of 1''s.

Example 1:

Input: 
N = 4 , M = 4
Arr[][] = {{0, 1, 1, 1},
           {0, 0, 1, 1},
           {1, 1, 1, 1},
           {0, 0, 0, 0}}
Output: 2
Explanation: Row 2 contains 4 1''s (0-based indexing).

//TC : r*log(c)


int rowWithMax1s(vector<vector<int> > arr, int r, int c) {
	    int mx=0,idx=-1;
	    for(int i = 0;i<r;i++){
	    	//iterator to first element in the range [first,last) which has a value greater than x
	        auto ub = upper_bound(arr[i].begin(),arr[i].end(),1); //end of 1's
	        
	        //iterator to first element in the range [first,last) which has a value not less than x
	        auto lb = lower_bound(arr[i].begin(),arr[i].end(),1); //starting of 1's
	        if(int(ub-lb)>mx){
	            mx=int(ub-lb);
	            idx=i;
	        }
	    }
	    return idx;
	}