795. Number of Subarrays with Bounded Maximum
================================================
Given an integer array nums and two integers left and right, return the number 
of contiguous non-empty subarrays such that the value of the maximum array element 
in that subarray is in the range [left, right].

The test cases are generated so that the answer will fit in a 32-bit integer.

 

Example 1:

Input: nums = [2,1,4,3], left = 2, right = 3
Output: 3
Explanation: There are three subarrays that meet the requirements: [2], [2, 1], [3].

Example 2:

Input: nums = [2,9,2,5,6], left = 2, right = 8
Output: 7

left=2,right=6,cnt= no of possible sub-array ending at h
        

case-1 : break point | [3(l) 1 4] 5(h) , left<=a[h]<=right

we can make subarray ending with 5 as : 
						5
					  4 5
					1 4 5
				  3 1 4 5
and sub-array ending at 4 :
		                4
		              1 4
		            3 1 4
so total cnt = 4(sub-array ending at 5)+3(sub-array ending at 4) 
                 + c(previous count for 3 & 1 using prev=h-l+1) = cnt + prev

case-2: break point | [3(l) 1 4] 1(h) , a[h]<left

we can make subarray ending with 1 as : 
						1 ---> will not be included as < left 
					  4 1
					1 4 1
				  3 1 4 1
so total cnt= cnt+prev(=h-l+1)

case-2: break point | [3(l) 1 4] 8(h) , a[h]>right

we can make subarray ending with 8 as : --> none of them will be included as 8 >right
						8 
					  4 8
					1 4 8
				  3 1 4 8

int numSubarrayBoundedMax(vector<int>& nums, int left, int right) {
        int n = nums.size(),l=0,h=0,cnt=0,prev=0;
        for(int h =0;h<n;h++){
            if(left<= nums[h]&&nums[h]<=right){
                prev=h-l+1;
                cnt+=prev;
            }
            else if(nums[h]<left){
                cnt+=prev; 
            }
            else {      
                l=h+1; //reset
                prev=0;
            }
        }
        return cnt;
    }


// Same approach but more optimized , takes least time among three


Explanation:

[left+1...right] is the longest subarray so far that has an ending number 
(A[right]) in range L...R.When A[i] > R, the update ensures this interval 
becomes empty again so the update to result can be handled uniformly in all 
situations.(making prev=0)

The initial case: we don''t have some A[right] in range L...R yet, 
so it''s also empty.


Helpful trace for

[|0| 3 |1| 4 5 |2 1| 5 |10| 6] , left=3,right=6

//res store possible combinations of subarray taking 1 to range at a time , 
// so we need to add current range in previous value as res = res + right-left
//each time we make range # of sub-array in total when we add a new element
//range = right-left

i:(0) >>> left:(-1), right:(-1), res:(0)
i:(1) >>> left:(-1), right:(1), res:(2) 
i:(2) >>> left:(-1), right:(1), res:(4)
i:(3) >>> left:(-1), right:(3), res:(8)
i:(4) >>> left:(-1), right:(4), res:(13)
i:(5) >>> left:(-1), right:(4), res:(18)
i:(6) >>> left:(-1), right:(4), res:(23)
i:(7) >>> left:(-1), right:(7), res:(31)
i:(8) >>> left:(8), right:(8), res:(31)
i:(9) >>> left:(8), right:(9), res:(32)


int numSubarrayBoundedMax(vector<int>& A, int L, int R) {
        int result=0, left=-1, right=-1;
        for (int i=0; i<A.size(); i++) {
            //a[8]=10 all are > R ,left= -1 -> 8
            if (A[i]>R) left=i; 
            //a[1]=3,a[3]=4,a[4]=5,a[7]=5,a[8]=10,a[9]=6 all are >=L
            //right= -1 -> 1 -> 3 -> 4 -> 7 -> 8 -> 9
            if (A[i]>=L) right=i; 
            result+=right-left; //as right covers left too,to get the range right-left
        }
        return result;
    }


//different approach,takes much more time


Let count(bound) is the number of subarrays which have all elements less than 
or equal to bound.
Finally, count(right) - count(left-1) is our result.
How to compute count(bound)?
Let ans is our answer
Let cnt is the number of consecutive elements less than or equal to bound so far
For index i in 0..n-1:
    If nums[i] <= bound then cnt = cnt + 1
    Else cnt = 0
    ans += cnt 
    // We have total cnt subarrays which end at index i_th and have all elements 
    //are less than or equal to bound

count(right) will count all the subarray with max <= right which means it will 
also count all the subarrays with max <= left - 1 And some remaining subarrays 
which will be our answer.
so we can write it as :
lets say our answer is ans
so count(right) = ans + rest = ans + count(left-1)
hence our equation : count(nums, right) - count(nums, left - 1) gives 
(ans + rest) - rest -> ans


int numSubarrayBoundedMax(vector<int>& nums, int left, int right) {
        return count(nums, right) - count(nums, left - 1);
    }
    int count(const vector<int>& nums, int bound) {
        int ans = 0, cnt = 0;
        for (int x : nums) {
            cnt = x <= bound ? cnt + 1 : 0;
            ans += cnt;
        }
        return ans;
    }