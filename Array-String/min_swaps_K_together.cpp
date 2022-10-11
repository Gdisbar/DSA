Minimum swaps and K together
=============================
// Given an array arr of n positive integers and a number k. One can apply a 
// swap operation on the array any number of times, i.e choose any two index i 
// and j (i < j) and swap arr[i] , arr[j] . Find the minimum number of swaps 
// required to bring all the numbers less than or equal to k together, i.e. make them a 
// contiguous subarray.

// Example 1:

// Input : 
// arr[ ] = {2, 1, 5, 6, 3} 
// K = 3
// Output : 
// 1
// Explanation:
// To bring elements 2, 1, 3 together,
// swap index 2 with 4 (0-based indexing),
// i.e. element arr[2] = 5 with arr[4] = 3
// such that final array will be- 
// arr[] = {2, 1, 3, 6, 5}


// Example 2:

// Input : 
// arr[ ] = {2, 7, 9, 5, 8, 7, 4} 
// K = 6 
// Output :  
// 2 
// Explanation: 
// To bring elements 2, 5, 4 together, 
// swap index 0 with 2 (0-based indexing)
// and index 4 with 6 (0-based indexing)
// such that final array will be- 
// arr[] = {9, 7, 2, 5, 4, 7, 8}

// A simple solution is to first count all elements less than or equal to 
// k(say ‘good’). Now traverse for every sub-array and swap those elements whose 
// value is greater than k. The time complexity of this approach is O(n^2)
// An efficient approach is to use the two-pointer technique and a sliding window. 
// The time complexity of this approach is O(n)


//     1. Find the count of all elements which are less than or equal to ‘k’. 
//     Let’s say the count is 'cnt'
//     2. Using the two-pointer technique for a window of length ‘cnt’, each time keep 
//     track of how many elements in this range are greater than ‘k’. Let’s say the 
//     total count is 'c'.
//     3. Repeat step 2, for every window of length 'cnt' and take a minimum of count 
//     'c' among them into 'res'. This 'res' will be the final answer.

int minSwap(int a[], int n, int k) {
        int cnt=0,c=0;
    	for(int i=0;i<n;++i)
    		if(a[i]<=k) cnt++;
    	// for 1st window calculate numbers that are less than k
    	for(int i=0;i<cnt;++i)
    		if(a[i]>k) c++;
    	// sliding window on rest of the array
    	int i=0,j=cnt,res=c;
    	while(j<n){
    		if(a[i]>k) c--; //out of window & it was > k
    		if(a[j]>k) c++; //last element of new window & it is > k
    		res=min(res,c);
    		i++,j++;
    	}
	    return res;
    }