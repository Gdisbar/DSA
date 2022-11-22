Largest sum subarray with at-least k numbers
================================================
// Given an array, find the subarray (containing at least k numbers) which has 
// the largest sum. 
// Examples: 
 

// Input : arr[] = {-4, -2, 1, -3} 
//             k = 2
// Output : -1
// The sub array is {-2, 1}

// Input : arr[] = {1, 1, 1, 1, 1, 1} 
//             k = 2
// Output : 6 
// The sub array is {1, 1, 1, 1, 1, 1}

int maxSumWithK(int a[], int n, int k)
{
    // maxSum[i] is going to store maximum sum
    // till index i such that a[i] is part of the
    // sum.
    int maxSum[n];
    maxSum[0] = a[0];
 
    // We use Kadane's algorithm to fill maxSum[]
    int curr_max = a[0];
    for (int i = 1; i < n; i++)
    {
        curr_max = max(a[i], curr_max+a[i]);
        maxSum[i] = curr_max;
    }
 
    // Sum of first k elements
    int sum = 0;
    for (int i = 0; i < k; i++)
        sum += a[i];
 
    // Use the concept of sliding window
    int result = sum;
    for (int i = k; i < n; i++)
    {
        // Compute sum of k elements ending
        // with a[i].
        sum = sum + a[i] - a[i-k];
 
        // Update result if required
        result = max(result, sum);
 
        // Include maximum sum till [i-k] also
        // if it increases overall max.
        result = max(result, sum + maxSum[i-k]);
    }
    return result;
}

//without extra space
long long int maxSumWithK(long long int a[], long long int n, long long int k)
{
    long long int sum = 0;
    for (long long int i = 0; i < k; i++) {
        sum += a[i];
    }
 
    long long int last = 0;
    long long int j = 0;
    long long int ans = LLONG_MIN;
    ans = max(ans, sum);
    for (long long int i = k; i < n; i++) {
        sum = sum + a[i];
        last = last + a[j++];
        ans = max(ans, sum);
        if (last < 0) {
            sum = sum - last;
            ans = max(ans, sum);
            last = 0;
        }
    }
    return ans;
}

862. Shortest Subarray with Sum at Least K
===============================================
// Given an integer array nums and an integer k, return the length of the 
// shortest non-empty subarray of nums with a sum of at least k. If there is no 
// such subarray, return -1.

// A subarray is a contiguous part of an array.

 

// Example 1:

// Input: nums = [1], k = 1
// Output: 1

// Example 2:

// Input: nums = [1,2], k = 4
// Output: -1

// Example 3:

// Input: nums = [2,-1,2], k = 3
// Output: 3

int shortestSubarray(vector<int> A, int K) {
        int N = A.size(), res = N + 1;
        deque<long> d; //indexes of increasing A[i]
        for (int i = 0; i < N; i++) {
            if (i > 0)
                A[i] += A[i - 1];
            if (A[i] >= K)
                res = min(res, i + 1);
            //For every A[i], we will compare A[i] - A[d[0]] with K 
            //(smallest prefix sum - A[i]), that before A[i], we didn't find a 
            //subarray whose sum at least K. A[i] is the first prefix sum that 
            //valid this condition. A[D[0]] ~ A[i-1] is the shortest subarray 
            //starting at A[D[0]] with sum at least K.
			// We have already find it for A[D[0]] and it can't be shorter, 
			//so we can drop it from our deque.
            while (d.size() > 0 && A[i] - A[d.front()] >= K) 
                res = min(res, i - d.front()), d.pop_front();

            //To keep A[D[i]] increasing in the deque , If A[i] <= A[d.back()] 
            //and moreover we already know that i > d.back(), it means that 
            //compared with d.back(),A[i] can help us make the subarray 
            //length shorter and sum bigger. So no need to keep d.back() in our 
            //deque.

            while (d.size() > 0 && A[i] <= A[d.back()]) 
                d.pop_back();
            d.push_back(i);
        }
        return res <= N ? res : -1;
    }