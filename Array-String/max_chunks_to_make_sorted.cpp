769. Max Chunks To Make Sorted
=================================
// You are given an integer array arr of length n that represents a permutation 
// of the integers in the range [0, n - 1].

// We split arr into some number of chunks (i.e., partitions), 
// and individually sort each chunk. After concatenating them, the result should 
// equal the sorted array.

// Return the largest number of chunks we can make to sort the array.

 

// Example 1:

// Input: arr = [4,3,2,1,0]
// Output: 1
// Explanation:
// Splitting into two or more chunks will not return the required result.
// For example, splitting into [4, 3], [2, 1, 0] will result in [3, 4, 0, 1, 2], 
// which isn''t sorted.

// Example 2:

// Input: arr = [1,0,2,3,4]
// Output: 4
// Explanation:
// We can split into two chunks, such as [1, 0], [2, 3, 4].
// However, splitting into [1, 0], [2], [3], [4] is the highest number of chunks 
// possible.

//Max no of chunks possible = n, Min no of chunks possible = 1 (taking total array)
//but here individually sort each chunk & concatenate them we can't do that 
//without modification (merge function of msort) so it''s different there from msort

//Chaining technique ------> find next greater element on left
// i=    0 1 2 3 |4 5| 6 7 8 
// arr=  3 0 1 2 |5 4| 8 6 7 
// idx=  3 3 3 3 |5 5| 8 8 8 ---> total chunk = 3

//TC : n 
int maxChunksToSorted(vector<int>& arr) {
        int n = arr.size(),idx=-1,cnt=0;
        for(int i = 0;i<n;i++){
            if(arr[i]>i){
                if(idx<arr[i]){
                    idx=arr[i];
                   // cnt++;
                } 
            }
            else{
                if(idx<i){
                    idx=i;
                  //  cnt++;
                }
            }
            if(i==idx) cnt++;
        }
        return cnt;
    }

//same approach but concise
int cnt = 0, idx = 0;
for (int i = 0; i < arr.size(); i++) {
    idx = max(idx, arr[i]);
    if (idx == i) {
        cnt++;
    }
}

768. Max Chunks To Make Sorted II
=====================================
// You are given an integer array arr.We split arr into some number of chunks 
// (i.e., partitions), and individually sort each chunk. After concatenating them, 
// the result should equal the sorted array.

// Return the largest number of chunks we can make to sort the array.

// //Here duplicates are allowed , that''s where it''s different from previous one

// Example 1:

// Input: arr = [5,4,3,2,1]
// Output: 1
// Explanation:
// Splitting into two or more chunks will not return the required result.
// For example, splitting into [5, 4], [3, 2, 1] will result in [4, 5, 1, 2, 3], 
// which isn''t sorted.

// Example 2:

// Input: arr = [2,1,3,4,4]
// Output: 4
// Explanation:
// We can split into two chunks, such as [2, 1], [3, 4, 4].
// However, splitting into [2, 1], [3], [4], [4] is the highest number of chunks 
// possible.

// i=          0 1  2  3  4  5  6  7
// a=        [30,10,20,40,60,50,75,70]
// leftmx=   [30 30 30 40 60 60 75 75]
// rightmn=  [10 10 20 40 50 50 70 70]

// leftmx[2]<rightmn[3],which means [30,10,20],[40,60,50,75,70] are the two chunks
// max impact of 1st group is upto i=2 because for 2nd group min=40 but  
// here in 1st group max=30

int maxChunksToSorted(vector<int>& arr) {
        int n = arr.size(),mx=INT_MIN,mn=INT_MAX,cnt=1;
        // we're not comparing leftmax[n-1] with rightmin[n]=INT_MAX,which is by default a chunk
        //so we start cnt=1 insted of cnt=0
        vector<int> leftmx(n,0),rightmn(n,INT_MAX);
        leftmx[0]=arr[0],rightmn[n-1]=arr[n-1];
        for(int i = 1;i<n-1;i++){
            rightmn[n-i-1]=min(rightmn[n-i],arr[n-i-1]);
            leftmx[i]=max(leftmx[i-1],arr[i]);
        }
        for(int i=0;i<n-1;i++){
            if(leftmx[i]<=rightmn[i+1]){
                cnt++;
            }
        }
        return cnt;
    }

//combining both in a single loop reduce running time to half 

// for(int i = 0;i<n;i++){
//         mx=max(mx,arr[i]);
//         leftmx[i]=mx;
//     }
    
//     for(int i = n-1;i>=0;i--){
//         mn=min(mn,arr[i]);
//         rightmn[i]=mn;
//     }


//more concise & faster ;) black magic,poor test cases , don't try this at home
// sum equal doesn''t mean that they are permutation of each other
// You could have [3,1,3] and [4,0,3]

// i=          0 1  2  3  4  5  6  7
// t=        [30,10,20,40,60,50,75,70]
// a=        [10,20,30,40,50,60,70,75]
// s1=		 [10 30 60 100 150 210 280 355 ] 
// s2=       [30 40 60 100 160 210 285 355 ] 

    int maxChunksToSorted(vector<int>& arr) {
        ll sum1 = 0, sum2 = 0, ans = 0;
        vector<int> t = arr;
        sort(t.begin(), t.end());
        for(int i = 0; i < arr.size(); i++) {
            sum1 += t[i];
            sum2 += arr[i];
            if(sum1 == sum2) ans++;
        }
	return ans;
    }