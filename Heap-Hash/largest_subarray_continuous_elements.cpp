Length of the largest subarray with contiguous elements 
=========================================================
// Given an array of distinct integers, find length of the longest subarray 
// which contains numbers that can be arranged in a continuous sequence. 

// Examples: 

// Input:  arr[] = {10, 12, 11};
// Output: Length of the longest contiguous subarray is 3

// Input:  arr[] = {14, 12, 11, 20};
// Output: Length of the longest contiguous subarray is 2

// Input:  arr[] = {1, 56, 58, 57, 90, 92, 94, 93, 91, 45};
// Output: Length of the longest contiguous subarray is 5


// Returns length of the longest contiguous subarray
int findLength(int arr[], int n)
{
    int max_len = 1;  // Initialize result
    for (int i=0; i<n-1; i++)
    {
        // Initialize min and max for all subarrays starting with i
        int mn = arr[i], mx = arr[i];
  
        // Consider all subarrays starting with i and ending with j
        for (int j=i+1; j<n; j++)
        {
            // Update min and max in this subarray if needed
            mn = min(mn, arr[j]);
            mx = max(mx, arr[j]);
  
            // If current subarray has all contiguous elements
            if ((mx - mn) == j-i)
                max_len = max(max_len, mx-mn+1);
        }
    }
    return max_len;  // Return result
}


Length of the largest subarray with contiguous elements (With duplicates)
==========================================================================
// Input:  arr[] = {10, 12, 11};
// Output: Length of the longest contiguous subarray is 3

// Input:  arr[] = {10, 12, 12, 10, 10, 11, 10};
// Output: Length of the longest contiguous subarray is 2 

// This function prints all distinct elements
int findLength(int arr[], int n)
{
    int max_len = 1; // Initialize result
 
    // One by one fix the starting points
    for (int i=0; i<n-1; i++)
    {
        // Create an empty hash set and
        // add i'th element to it.
        set<int> myset;
        myset.insert(arr[i]);
 
        // Initialize max and min in
        // current subarray
        int mn = arr[i], mx = arr[i];
 
        // One by one fix ending points
        for (int j=i+1; j<n; j++)
        {
            // If current element is already
            // in hash set, then this subarray
            // cannot contain contiguous elements
            if (myset.find(arr[j]) != myset.end())
                break;
 
            // Else add current element to hash
            // set and update min, max if required.
            myset.insert(arr[j]);
            mn = min(mn, arr[j]);
            mx = max(mx, arr[j]);
 
            // We have already checked for
            // duplicates, now check for other
            // property and update max_len
            // if needed
            if (mx - mn == j - i)
                max_len = max(max_len, mx - mn + 1);
        }
    }
    return max_len; // Return result
}
 