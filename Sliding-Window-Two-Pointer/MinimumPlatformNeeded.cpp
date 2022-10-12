// Given the arrival and departure times of all trains that reach a 
//railway station, the task is to find the minimum number of platforms required 
//for the railway station so that no train waits. 
// We are given two arrays that represent the arrival and departure times of 
//trains that stop.

// Examples: 

//     Input: arr[] = {9:00, 9:40, 9:50, 11:00, 15:00, 18:00} 
//     dep[] = {9:10, 12:00, 11:20, 11:30, 19:00, 20:00} 
//     Output: 3 
//     Explanation: There are at-most three trains at a time 
//          (time between 11:00 to 11:20)

//     Input: arr[] = {9:00, 9:40} 
//     dep[] = {9:10, 12:00} 
//     Output: 1 
//     Explanation: Only one platform is needed. 

//===========================================================================================
// Returns minimum number of platforms required
int findPlatform(int arr[], int dep[], int n)
{
 
    // plat_needed indicates number of platforms
    // needed at a time
    int plat_needed = 1, result = 1;
    int i = 1, j = 0;
 
    // run a nested  loop to find overlap
    for (int i = 0; i < n; i++) {
        // minimum platform
        plat_needed = 1;
 
        for (int j = i + 1; j < n; j++) {
            // check for overlap
            if ((arr[i] >= arr[j] && arr[i] <= dep[j]) ||
           (arr[j] >= arr[i] && arr[j] <= dep[i]))
                plat_needed++;
        }
 
        // update result
        result = max(result, plat_needed);
    }
 
    return result;
}

//===========================================================================================

// Time Complexity: O(N * log N). 
// One traversal O(n) of both the array is needed after sorting O(N * log N), 
//so the time complexity is O(N * log N).
//===========================================================================================

int findPlatform(int arr[], int dep[], int n)
    {
    	
    	sort(arr,arr+n);
    	sort(dep,dep+n);
    	int ans = 0,platformNeeded = 1;
    	int i = 1,j = 0;
    	while(i<n &&j<n){
    	    if(arr[i]<=dep[j]){
    	        platformNeeded++;
    	        i++;
    	        if(platformNeeded>ans){
    	            ans=platformNeeded;
    	        }
    	    }
    	    if(arr[i]>dep[j]){
    	        platformNeeded--;
    	        j++;
    	    }
    	    
    	}
    	return ans;
    }