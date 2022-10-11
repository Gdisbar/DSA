Minimize the Heights II
===========================
Given an array arr[] denoting heights of N towers and a positive integer K.

For each tower, you must perform exactly one of the following operations exactly 
once.

    Increase the height of the tower by K.
    Decrease the height of the tower by K ( you can do this operation only if 
    the height of the tower is greater than or equal to K)

Find out the minimum possible difference between the height of the shortest and 
tallest towers after you have modified each tower.

You can find a slight modification of the problem here.
Note: It is compulsory to increase or decrease the height by K for each tower.


Example 1:

Input:
K = 2, N = 4
Arr[] = {1, 5, 8, 10}
Output:
5
Explanation:
The array can be modified as 
{3, 3, 6, 8}. The difference between 
the largest and the smallest is 8-3 = 5.

Example 2:

Input:
K = 3, N = 5
Arr[] = {3, 9, 12, 16, 20}
Output:
11
Explanation:
The array can be modified as
{6, 12, 9, 13, 17}. The difference between 
the largest and the smallest is 17-6 = 11. 

Expected Time Complexity: O(N*logN)
Expected Auxiliary Space: O(N)

// we can't increase & decrease each heights & check --> 2^n combinations

int getMinDiff(int a[], int n, int k) {
        sort(a,a+n); // a1>a2>a3>..>an => diff will be min if we pick(a1,a2),(a2,a3)
        int ans=a[n-1]-a[0];
        int largest=a[n-1]-k,smallest=a[0]+k;
        int mn=INT_MAX,mx=INT_MIN;
        for(int i=0;i<n-1;++i){
            mn=min(smallest,a[i+1]-k);
            mx=max(largest,a[i]+k);
            if(mn<0)continue; //otherwise we'll end up maximizing the difference
            ans=min(ans,mx-mn);
        }
        return ans;
    }

def getMinDiff(self, a, n, k):
        a.sort() #can''t use sorted(a) in that case need to write a=sorted(a)
        ans=a[n-1]-a[0]
        largest,smallest=a[n-1]-k,a[0]+k
        #mn,mx=float('inf'),float('-inf')
        for i in range(1,n):
            if a[i]<k:continue
            mn=min(smallest,a[i]-k)
            mx=max(largest,a[i-1]+k)
            #if(mn<0):continue
            ans=min(ans,mx-mn)
        return ans;