4. Median of Two Sorted Arrays
================================
Given two sorted arrays nums1 and nums2 of size m and n respectively, 
return the median of the two sorted arrays.

The overall run time complexity should be O(log (m+n)).

 

Example 1:

Input: nums1 = [1,3], nums2 = [2]
Output: 2.00000
Explanation: merged array = [1,2,3] and median is 2.

Example 2:

Input: nums1 = [1,2], nums2 = [3,4]
Output: 2.50000
Explanation: merged array = [1,2,3,4] and median is (2 + 3) / 2 = 2.5.


// Brute-force merge the two array & find median(avg of middle 2 elemnet)
// TC : m+n , SC : m+n



[2 3 | 5 7] --> median = (3+5)/2 = 4

[2 3 |(4/4) |5 7] --> both lower & upper subarray contains 4 with L=3,R=5

Relation between Index(L & R) with N :
-----------------------------------------------
N        Index of L / R       // L=(N-1)/2 , R= N/2
1               0 / 0
2               0 / 1
3               1 / 1  
4               1 / 2  [2 3 5 7]   = [2 3 | 5 7]    
5               2 / 2  [2 3 4 5 7] = [2 3 |(4/4) |5 7]
6               2 / 3
7               3 / 3
8               3 / 4



[6 9 13 18]  ->   [# 6 # 9 # 13 # 18 #]    (N = 4) , # imaginary positions
position index     0 1 2 3 4 5  6 7  8     (N_Position = 9)
          
[6 9 11 13 18]->   [# 6 # 9 # 11 # 13 # 18 #]   (N = 5)
position index      0 1 2 3 4 5  6 7  8 9 10    (N_Position = 11)

total positions = 2*N + 1 , so cut on left side (N-1)/2 & on right side N/2



Now for the two-array case:
---------------------------------
A1: [# 1 # 2 # 3 # 4 # 5 #]    (N1 = 5, N1_positions = 11)

A2: [# 1 # 1 # 1 # 1 #]     (N2 = 4, N2_positions = 9)

total positions = 2*N1 + 1 + 2*N2 + 1 , so exactly N1+N2 cuts in both side

if C2=K then C1=N1+N2-K

When the cuts are made, we''d have two L's and two R's. They are

     L1 = A1[(C1-1)/2]; R1 = A1[C1/2];
     L2 = A2[(C2-1)/2]; R2 = A2[C2/2];


        L1 |R1
x = [x1,x2,|x3,x4.x5.x6] ---> N1=6 , L1=(N1-1)/2=2,R1=N1/2=3

when we merge we get total x+y = N1+N2 elements so calculation of L2,R2 needs 
to be done accordingly

                 L2 |R2
y = [y1,y2,y3,y4,y5,|y6,y7,y8] --> N2=8 , L2=(N1+N2-2-1)/2=5,R2=(N1+N2-2)/2=6

target :  avg(max(x2,y5),min(x3,y6)) = (max(L1,L2)+min(R1,R2))/2
condition : x2<=y6,y5<=x3 ---> L1<=R2,L2<=R1


Now how do we decide if this cut is the cut we want? Because L1, L2 are the 
greatest numbers on the left halves and R1, R2 are the smallest numbers on the 
right, as A1 & A2 is sorted L1<=R1 & L2<=R2 is guaranteed so we need to check only

L1 <= R2 //&& L1 <= R1 
L2 <= R1 //&& L2 <= R2

//Using Binary Search on above range

L1 > R2 : too many large numbers on the left half of A1,we move C1 to left 
            (i.e. move C2 to the right)
l2 > R1 : too many large numbers on the left half of A2, and we must move C2 to the 
                left.
otherwise this cut is right one & mid = (max(L1,L2)+min(R1, R2)) / 2

we generally move C2 as it''s (i.e resulting search complexity log(min(N1,N2)) ) 
shorter as to move C1 a range check must be done 1st

A1=[1],A2=[2 3 4 5 6 7 8] --> we can''t do [2 | 3 4 5 6 7 8] beacause
A= [2] can''t balance rest of the element [3 4 5 6 7 8]

Edge case : when a cut falls in 0-th(first) or 2*N-th(last) pasition
i.e if C2=2*N2 then R2=A2[2*N2/2]=A2[N2],which exceeds boundary to overcome this 
we add INT_MIN to A[-1] & INT_MAX to A[N]


// TC : log(min(M,N)) , faster than 43% , less memory than 62%


double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
    int N1 = nums1.size();
    int N2 = nums2.size();
    // Make sure A2 is the shorter one.
    if (N1 < N2) return findMedianSortedArrays(nums2, nums1);  
    
    int lo = 0, hi = N2 * 2;
    while (lo <= hi) {
        int mid2 = (lo + hi) / 2;   // Try Cut 2 , C2
        int mid1 = N1 + N2 - mid2;  // Calculate Cut 1 accordingly , C1
        
        double L1 = (mid1 == 0) ? INT_MIN : nums1[(mid1-1)/2];  // Get L1, R1, L2, R2 respectively
        double L2 = (mid2 == 0) ? INT_MIN : nums2[(mid2-1)/2];
        double R1 = (mid1 == N1 * 2) ? INT_MAX : nums1[(mid1)/2];
        double R2 = (mid2 == N2 * 2) ? INT_MAX : nums2[(mid2)/2];
        
        if (L1 > R2) lo = mid2 + 1; // A1's lower half is too big; need to move C1 left (C2 right)
        else if (L2 > R1) hi = mid2 - 1; // A2's lower half too big; need to move C2 left.
        else return (max(L1,L2) + min(R1, R2)) / 2; // Otherwise, that's the right cut.
    }
    return -1;
} 