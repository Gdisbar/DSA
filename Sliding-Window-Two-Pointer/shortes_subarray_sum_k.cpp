862. Shortest Subarray with Sum at Least K
--------------------------------------------
Given an integer array nums and an integer k, return the length of the shortest non-empty subarray of nums with a sum of at least k. If there is no such subarray, return -1.

A subarray is a contiguous part of an array.

 

Example 1:

Input: nums = [1], k = 1
Output: 1

Example 2:

Input: nums = [1,2], k = 4
Output: -1

Example 3:

Input: nums = [2,-1,2], k = 3
Output: 3


Prepare

From @Sarmon:
"What makes this problem hard is that we have negative values.
If you haven't already done the problem with positive integers only,
I highly recommend solving it first"

    Minimum Size Subarray Sum

Explanation

Calculate prefix sum B of list A.
B[j] - B[i] represents the sum of subarray A[i] ~ A[j-1]
Deque d will keep indexes of increasing B[i].
For every B[i], we will compare B[i] - B[d[0]] with K.

Complexity:

Every index will be pushed exactly once.
Every index will be popped at most once.

Time O(N)
Space O(N)

How to think of such solutions?

Basic idea, for array starting at every A[i], find the shortest one with sum at leat K.
In my solution, for B[i], find the smallest j that B[j] - B[i] >= K.
Keep this in mind for understanding two while loops.

What is the purpose of first while loop?

For the current prefix sum B[i], it covers all subarray ending at A[i-1].
We want know if there is a subarray, which starts from an index, ends at A[i-1] and has 
at least sum K.So we start to compare B[i] with the smallest prefix sum in our deque, 
which is B[D[0]], hoping that [i] - B[d[0]] >= K.So if B[i] - B[d[0]] >= K, we can 
update our result res = min(res, i - d.popleft()).The while loop helps compare one by one, 
until this condition isn''t valid anymore.

Why we pop left in the first while loop?

This the most tricky part that improve my solution to get only O(N).
D[0] exists in our deque, it means that before B[i], we didn''t find a subarray whose sum 
at least K.
B[i] is the first prefix sum that valid this condition.
In other words, A[D[0]] ~ A[i-1] is the shortest subarray starting at A[D[0]] with sum at least K.
We have already find it for A[D[0]] and it can''t be shorter, so we can drop it from our deque.

What is the purpose of second while loop?

To keep B[D[i]] increasing in the deque.

Why keep the deque increase?

If B[i] <= B[d.back()] and moreover we already know that i > d.back(), it means that compared 
with d.back(),B[i] can help us make the subarray length shorter and sum bigger. So no need to 
keep d.back() in our deque.


    int shortestSubarray(vector<int> A, int K) {
        int N = A.size(), res = N + 1;
        deque<long> d;
        for (int i = 0; i < N; i++) {
            if (i > 0)
                A[i] += A[i - 1];
            if (A[i] >= K)
                res = min(res, i + 1);
            while (d.size() > 0 && A[i] - A[d.front()] >= K)
                res = min(res, i - d.front()), d.pop_front();
            while (d.size() > 0 && A[i] <= A[d.back()])
                d.pop_back();
            d.push_back(i);
        }
        return res <= N ? res : -1;
    }



Understanding with values:
B[d.back] = 7 ; B[i] =4 ; B[future id] = 10; k=3
if:
B[future id] - B[d.back()] >= k => 10 - 7 >= 3
and
B[d.back()] >= B[i] => 7>=4
then :
B[future id] - B[i] >= k => 10-4>=3

Length is shorter because [future id - i] is shorter than [future id - d.back]
sum is bigger as well because [10-4] is bigger than [10-7]

Q: Why keep the deque increase?
A: If B[i] <= B[d.back()] and moreover we already know that i > d.back(), it means that compared with d.back(),
B[i] can help us make the subarray length shorter and sum bigger. So no need to keep d.back() in our deque.

More detailed on this, we always add at the LAST position
B[d.back] <- B[i] <- ... <- B[future id]
B[future id] - B[d.back()] >= k && B[d.back()] >= B[i]
B[future id] - B[i] >= k too

so no need to keep B[d.back()]



209. Minimum Size Subarray Sum
---------------------------------
Given an array of positive integers nums and a positive integer target, return the 
minimal length of a contiguous subarray [numsl, numsl+1, ..., numsr-1, numsr] of which 
the sum is greater than or equal to target. If there is no such subarray, return 0 instead.

 

Example 1:

Input: target = 7, nums = [2,3,1,2,4,3]
Output: 2
Explanation: The subarray [4,3] has the minimal length under the problem constraint.

Example 2:

Input: target = 4, nums = [1,4,4]
Output: 1

Example 3:

Input: target = 11, nums = [1,1,1,1,1,1,1,1]
Output: 0


Intuition

    Shortest Subarray with Sum at Least K
    Actually I did this first, the same prolem but have negatives.
    I suggest solving this prolem first then take 862 as a follow-up.

Explanation

The result is initialized as res = n + 1.
One pass, remove the value from sum s by doing s -= A[j].
If s <= 0, it means the total sum of A[i] + ... + A[j] >= sum that we want.
Then we update the res = min(res, j - i + 1)
Finally we return the result res

Complexity

Time O(N)
Space O(1)


int minSubArrayLen(int s, vector<int>& A) {
        int i = 0, n = A.size(), res = n + 1;
        for (int j = 0; j < n; ++j) {
            s -= A[j];
            while (s <= 0) {
                res = min(res, j - i + 1);
                s += A[i++];
            }
        }
        return res % (n + 1);
    }