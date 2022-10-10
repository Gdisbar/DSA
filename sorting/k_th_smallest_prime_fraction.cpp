786. K-th Smallest Prime Fraction
===================================
// You are given a sorted integer array arr containing 1 and prime numbers, 
// where all the integers of arr are unique. You are also given an integer k.

// For every i and j where 0 <= i < j < arr.length, we consider the fraction 
// arr[i] / arr[j].

// Return the kth smallest fraction considered. Return your answer as an array of 
// integers of size 2, where answer[0] == arr[i] and answer[1] == arr[j].

 

// Example 1:

// Input: arr = [1,2,3,5], k = 3
// Output: [2,5]
// Explanation: The fractions to be considered in sorted order are:
// 1/5, 1/3, 2/5, 1/2, 3/5, and 2/3.
// The third fraction is 2/5.

// Example 2:

// Input: arr = [1,7], k = 1
// Output: [1,7]

// Input: arr=[1,17,109,211,239,401,523,641,661,937,1171,1451,1493,1523,1637,1979,2081,2393,2423,2473,2767,2801,2843,2851,2999,3083,3253,3329,3463,3467,3761,3797,3881,3931,4007,4159,4243,4261,4363,4447,4513,4517,4651,4909,4933,4943,5081,5119,5279,5443,5689,5711,5717,5743,5861,5939,5981,6011,6199,6317,6469,6977,7129,7283,7681,7703,7727,7823,7907,7963,8011,8243,8573,8707,9043,9157,9181,9511,9923,9949,10099,10321,10589,10853,10903,11057,11549,11831,11927,12007,12119,12263,12373,12497,12841,12953,13003,13009,13033,13093,13469,14149,14197,14221,14249,14293,14461,14519,14551,14747,14867,15187,15263,15619,15749,15889,16127,16349,16561,16747,16829,16927,16979,17191,17599,17747,17903,17929,17957,18311,18451,18481,18679,19069,19259,19309,19421,19441,19553,19583,19699,19819,20089,20173,20441,20593,20939,21023,21169,21313,21397,21683,21841,22031,22073,22129,22291,22397,22501,22717,22937,23011,23021,23081,23431,23497,23687,23689,23857,23879,24029,24469,24659,24671,25127,25423,25447,25759,25841,26029,26161,26297,26479,26641,26687,26699,26777,27043,27073,27271,27827,28547,28621,28771,29399,29473,29501,29573,29633,29851]
// 		k=18055       //0.890115 : 19441/21841
// Output : [19441,21841]

//TLE , but answer is correct

vector<int> kthSmallestPrimeFraction(vector<int>& arr, int k) {
        map<double,pair<int,int>> mp; // stores in sorted order
        for(int i=0;i<arr.size()-1;++i){
            for(int j=i+1;j<arr.size();++j){
                mp[(double)arr[i]/arr[j]]=make_pair(arr[i],arr[j]);
            }
        }
        for(auto x : mp){
            k--;
            if(k==0){
                return vector<int> {x.second.first,x.second.second};
            }
        }
        return vector<int> {};
    }

// Binary Search - 94% faster, 89% less memory
//TC : n*log(max(A)^2)
// p/q will be smallest = min(A)/max(A)
  vector<int> kthSmallestPrimeFraction(vector<int>& A, int K) {
        double l = 0, r = 1;
        int p = 0, q = 1; // q=1 to avoid zero division error

        for (int n = A.size(), cnt = 0; true; cnt = 0, p = 0) { // while(true)
            double m = (l + r) / 2;

            for (int i = 0, j = n - 1; i < n; i++) { 
                while (j >= 0 && A[i] > m * A[n - 1 - j]) j--; // we need smaller ones , a[i]/a[n-1-j] > m , both +ve (given)
                cnt += (j + 1); // these elements are smaller than mid

                if (j >= 0 && p * A[n - 1 - j] < q * A[i]) { //we store current smallest A[i]/A[n-1-j] > p/q
                    p = A[i];
                    q = A[n - 1 - j];
                }
            }

            if (cnt < K) {
                l = m;
            } else if (cnt > K) {
                r = m;
            } else {
                return {p, q};
            }
        }
    }

// 45 % faster,60% less memory ,but using strict test cases can get TLE
//my idea was to use map but priority queue is faster
// TC : N*logN + K --> pushing every element in min heap(heapify takes logN)
consider an input of [n1, n2, n3, n4, n5], the possible factors are:
[n1/n5, n1/n4, n1/n3, n1/n2, n1/n1] ---> 1st time on pq
[n2/n5, n2/n4, n2/n3, n2/n2]
[n3/n5, n3/n4, n3/n3]
[n4/n5, n4/n4]
[n5/n5]
vector<int> kthSmallestPrimeFraction(vector<int>& A, int K) {
        int n = A.size();
        //priority_queue <Type, vector<Type>, ComparisonType > min_heap;
        priority_queue<pair<double, pair<int, int>>, vector<pair<double, pair<int, int>>>> pq;
        for (int i = 0; i < n; i++) {
            pq.push(make_pair(-1.0*A[i]/A[n-1], make_pair(i, n-1))); //higher fraction -ve is lowest,reverse of positive number without which it'll be max_heap
        }
        int i, j;
        while(K--) { //after remove 1st K-1 smallest, we're left with K-th smallest
            auto p = pq.top().second; pq.pop();
            i = p.first; //i=0 
            j = p.second; //n-1
            pq.push(make_pair(-1.0*A[i]/A[j-1], make_pair(i, j-1)));
        }
        return vector<int>{A[i], A[j]};
    }


// for each row i, all the numbers (call them A[j]) to the right of A[i]/m, are 
// the ones such that A[i]/A[j] will be smaller than m.
// sum them up so that you will know the total number of pairs A[i]/A[j] that are 
// smaller than m. Find a proper m such that the total number equals K, and then you 
// find the maximum A[i]/A[j] among all pairs that are smaller than A[i]/m, which is 
// the Kth smallest number.

// //Python3

// def kthSmallestPrimeFraction(self, A, K):
//         l, r, N = 0, 1, len(A)
//         while True:
//             m = (l + r) / 2
//             border = [bisect.bisect(A, A[i] / m) for i in range(N)]
//             cur = sum(N - i for i in border)
//             if cur > K:
//                 r = m
//             elif cur < K:
//                 l = m
//             else:
//                 return max([(A[i], A[j]) for i, j in enumerate(border) if j < N], key=lambda x: x[0] / x[1])

