First negative integer in every window of size k
==================================================
// Given an array and a positive integer k, find the first negative integer for 
// each window(contiguous subarray) of size k. If a window does not contain a 
// negative integer, then print 0 for that window.

// Examples:  

// Input : arr[] = {-8, 2, 3, -6, 10}, k = 2
// Output : -8 0 -6 -6

// First negative integer for each window of size k
// {-8, 2} = -8
// {2, 3} = 0 (does not contain a negative integer)
// {3, -6} = -6
// {-6, 10} = -6

// Input : arr[] = {12, -1, -7, 8, -15, 30, 16, 28} , k = 3
// Output : -1 -1 -7 -15 -15 0

void printFirstNegativeInteger(int arr[], int n, int k)
{
    // flag to check whether window contains
    // a negative integer or not
    bool flag;
     
    // Loop for each subarray(window) of size k
    for (int i = 0; i<(n-k+1); i++)          
    {
        flag = false;
 
        // traverse through the current window
        for (int j = 0; j<k; j++)
        {
            // if a negative integer is found, then
            // it is the first negative integer for
            // current window. Print it, set the flag
            // and break
            if (arr[i+j] < 0)
            {
                cout << arr[i+j] << " ";
                flag = true;
                break;
            }
        }
         
        // if the current window does not
        // contain a negative integer
        if (!flag)
            cout << "0" << " ";
    }   
}

//better

void printFirstNegativeInteger(int arr[], int n, int k)
{
    // A Double Ended Queue, Di that will store indexes of
    // useful array elements for the current window of size k.
    // The useful elements are all negative integers.
    deque<int>  dq;
  
    /* Process first k (or first window) elements of array */
    int i;
    for (i = 0; i < k; i++)
        // Add current element at the rear of dq if it is a negative integer
        if (arr[i] < 0)
            dq.push_back(i);
     
    // Process rest of the elements, i.e., from arr[k] to arr[n-1]
    for ( ; i < n; i++){
        // if dq is not empty then the element at the
        // front of the queue is the first negative integer
        // of the previous window
        if (!dq.empty())
            cout << arr[dq.front()] << " ";
         
        // else the window does not have a
        // negative integer
        else
            cout << "0" << " ";
  
        // Remove the elements which are out of this window
        while ( (!dq.empty()) && dq.front() < (i - k + 1))
            dq.pop_front();  // Remove from front of queue
  
        // Add current element at the rear of dq
        // if it is a negative integer
        if (arr[i] < 0)
            dq.push_back(i);
    }
  
    // Print the first negative
    // integer of last window
    if (!dq.empty())
           cout << arr[dq.front()] << " ";
    else
        cout << "0" << " ";      
     
}
 
239. Sliding Window Maximum
==============================
// You are given an array of integers nums, there is a sliding window of 
// size k which is moving from the very left of the array to the very right. 
// You can only see the k numbers in the window. Each time the sliding window 
// moves right by one position.

// Return the max sliding window.

 

// Example 1:

// Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
// Output: [3,3,5,5,6,7]
// Explanation: 
// Window position                Max
// ---------------               -----
// [1  3  -1] -3  5  3  6  7       3
//  1 [3  -1  -3] 5  3  6  7       3
//  1  3 [-1  -3  5] 3  6  7       5
//  1  3  -1 [-3  5  3] 6  7       5
//  1  3  -1  -3 [5  3  6] 7       6
//  1  3  -1  -3  5 [3  6  7]      7

// Example 2:

// Input: nums = [1], k = 1
// Output: [1]


vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        int n = nums.size();
        vector<int> windowMax;
        deque<int> dq;
        for (int i = 0; i < n; i++) {
        	//out of window i-(k-1) , st only [i-(k-1),i] in dq
            while (!dq.empty() && dq.front() < i - k + 1) 
                dq.pop_front();
            //a[x] <a[i] and x<i, then a[x] has no chance to be the "max" in 
            //[i-(k-1),i], or any other subsequent window: a[i] would always 
            // be a better candidate.
            while (!dq.empty() && nums[dq.back()] < nums[i]) //current > last of dq
                dq.pop_back();  
            dq.push_back(i);
            if (i >= k - 1) // head of dq is the max
            	windowMax.push_back(nums[dq.front()]);
        }
        return windowMax;
    }