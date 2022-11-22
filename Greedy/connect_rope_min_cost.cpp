Connect N Ropes With Minimum Cost
========================================
// There are given N ropes of different lengths, we need to connect these 
// ropes into one rope. The cost to connect two ropes is equal to sum of 
// their lengths. The task is to connect the ropes with minimum cost.

// Example 1:

// Input:
// n = 4
// arr[] = {4, 3, 2, 6}
// Output: 
// 29
// Explanation:
// For example if we are given 4
// ropes of lengths 4, 3, 2 and 6. We can
// connect the ropes in following ways.
// 1) First connect ropes of lengths 2 and 3.
// Now we have three ropes of lengths 4, 6
// and 5.
// 2) Now connect ropes of lengths 4 and 5.
// Now we have two ropes of lengths 6 and 9.
// 3) Finally connect the two ropes and all
// ropes have connected.
// Total cost for connecting all ropes is 5
// + 9 + 15 = 29. This is the optimized cost
// for connecting ropes. Other ways of
// connecting ropes would always have same
// or more cost. For example, if we connect
// 4 and 6 first (we get three strings of 3,
// 2 and 10), then connect 10 and 3 (we get
// two strings of 13 and 2). Finally we
// connect 13 and 2. Total cost in this way
// is 10 + 13 + 15 = 38.

// Sample Input 2:

// 5
// 1 2 3 4 5

// Sample Output 2:

// 33

long long connectRopes(int* arr, int n)
{
    // Your code goes here
    //vector<int> v;
    long long s = 0,ans=0;
    priority_queue <long long, vector<long long>, greater<long long> > pq;
    for(int i=0;i<n;++i)
       pq.push((long)arr[i]);
    while(pq.size()>1){         // using !pq.empty() will not work , pq needs to hold the last element
        long long s1=pq.top();
        pq.pop();
        long long s2=pq.top();
        pq.pop();
        s=s1+s2;
        pq.push(s);
        ans+=s;
    }
    return ans;
}