1005. Maximize Sum Of Array After K Negations
===============================================
Given an integer array nums and an integer k, modify the array in the following way:

    choose an index i and replace nums[i] with -nums[i].

You should apply this process exactly k times. You may choose the same index i multiple times.

Return the largest possible sum of the array after modifying it in this way.

 

Example 1:

Input: nums = [4,2,3], k = 1
Output: 5
Explanation: Choose index 1 and nums becomes [4,-2,3].

Example 2:

Input: nums = [3,-1,0,2], k = 3
Output: 6
Explanation: Choose indices (1, 2, 2) and nums becomes [3,1,0,2].

Example 3:

Input: nums = [2,-3,-1,5,-4], k = 2
Output: 13
Explanation: Choose indices (1, 4) and nums becomes [2,3,-1,5,4].


 Now for the return statement

    res is the total sum of the new array
    K % 2 check if the remaining K is odd because if it''s even it will have no effect (we will flip a number and then get it back to the original)
    flip the minimum number and remove twice its value from the result (twice because we already added it as positive in our sum operation)

You are not using the fact that -100<= value <= 100. In that case you could use Counting sort which is O(n)

class Solution:
    def largestSumAfterKNegations(self, A: List[int], K: int) -> int:
        c = collections.Counter(A)
        for i in range(-100, 0):
            if K == 0:
                break
            flips = min(K, c[i])
            c[i] -= flips
            c[-i] += flips //c[n-1-i]
            K -= flips
        return sum(c.elements()) - K % 2 * min(A, key=abs) * 2


int largestSumAfterKNegations(vector<int>& A, int K) {
        sort(A.begin(),A.end());
        for(int i = 0; i < A.size() && A[i] < 0 && K > 0; K--, i++){
            A[i] = -A[i];
        }
        int mn = INT_MAX;
        int sum = 0;
        for (int x : A){
            mn = min(x, mn);
            sum += x;
        }
        return K%2==0?sum:sum-mn*2;
    }


   class Solution {
public:
    int largestSumAfterKNegations(vector<int>& A, int K) {
        PriorityQueue<Integer> positive = new PriorityQueue<Integer>();
        PriorityQueue<Integer> negative = new PriorityQueue<Integer>();
        for(int num: A){
            if(num >= 0) positive.push(num);
            else negative.push(num);
        }
        for(int i = 0;i<K;i++){
            if(!negative.isEmpty()){
                Integer temp = negative.poll();
                positive.push(-temp);
            }else {
                Integer temp = positive.poll();
                negative.push(-temp);
            }
        }
        int sum = 0;
        for(int num:positive) sum+=num;
        for(int num:negative) sum+=num;
        return sum;
    }
};