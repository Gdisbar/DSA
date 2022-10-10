// array need not to be sorted as we''re not performing BS on array rather than range
// where r >> max(a) , so by default natural numbers are sorted & mid will always be a
// +ve Integer



1283. Find the Smallest Divisor Given a Threshold
===================================================
// Given an array of integers nums and an integer threshold, we will choose a positive 
// integer divisor, divide all the array by it, and sum the division''s result. 
// Find the smallest divisor such that the result mentioned above is less than or 
// equal to threshold.

// Each result of the division is rounded to the nearest integer greater than or 
// equal to that element. (For example: 7/3 = 3 and 10/2 = 5).

// The test cases are generated so that there will be an answer.

 

// Example 1:

// Input: nums = [1,2,5,9], threshold = 6
// Output: 5
// Explanation: We can get a sum to 17 (1+2+5+9) if the divisor is 1. 
// If the divisor is 4 we can get a sum of 7 (1+1+2+3) and if the divisor is 5 the 
// sum will be 5 (1+1+1+2). 

// Example 2:

// Input: nums = [44,22,33,11,1], threshold = 5
// Output: 44


// Constraints:

//     1 <= nums.length <= 5 * 1e4
//     1 <= nums[i] <= 1e6
//     nums.length <= threshold <= 1e6

// 25% faster , 18% less memory ---> 84% faster , 91% less memory 
//TC : N*logM, where M = max(A)
int smallestDivisor(vector<int>& nums, int threshold) {
        int l=1,r=1000000; // if you put 1e6 grows up to 54% ,58% 
        while(l<r){
            int m=l+(r-l)/2;
            int total=0;
            for(int x : nums){ //total+=(x+m-1)/m; -> more than 3x faster & 5x less memory
                if(m!=0)
                    total+=x/m;
                if(m!=0&&x%m!=0) total+=1;
            }
            if(total<=threshold) r=m; //we've found answer but we're looking for lesser number
            else l=m+1;
        }
        return l;
    }