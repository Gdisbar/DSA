973. K Closest Points to Origin
====================================
Given an array of points where points[i] = [x[i], y[i]] represents a point on 
the X-Y plane and an integer k, return the k closest points to the 
origin (0, 0).

The distance between two points on the X-Y plane is the Euclidean 
distance (i.e., √(x1 - x2)^2 + (y1 - y2)^2).

You may return the answer in any order. The answer is guaranteed to be 
unique (except for the order that it is in).

 

Example 1:

Input: points = [[1,3],[-2,2]], k = 1
Output: [[-2,2]]
Explanation:
The distance between (1, 3) and the origin is sqrt(10).
The distance between (-2, 2) and the origin is sqrt(8).
Since sqrt(8) < sqrt(10), (-2, 2) is closer to the origin.
We only want the closest k = 1 points from the origin, so the answer is 
just [[-2,2]].

Example 2:

Input: points = [[3,3],[5,-1],[-2,4]], k = 2
Output: [[3,3],[-2,4]]
Explanation: The answer [[-2,4],[3,3]] would also be accepted.

//complexity & runtime shows total different scenario  

// TC : n+n*log(n) , SC : n

    vector<vector<int>> kClosest(vector<vector<int>>& points, int k) {
        vector<pair<double,pair<int,int>>> v;
        vector<vector<int>> res(k);
        for(int i=0;i<points.size();++i){
            int x = points[i][0];
            int y = points[i][1];
            double d=sqrt(pow(x,2)+pow(y,2));
            v.push_back(make_pair(d,make_pair(x,y)));
        }
        sort(v.begin(),v.end());
        for(int i=0;i<k;++i){
            res[i].push_back(v[i].second.first);
            res[i].push_back(v[i].second.second);
        }
        return res;
    }

// multimap ---> TC : n*log(n)
vector<vector<int>> kClosest(vector<vector<int>>& points, int K) {
        multiset<vector<int>, compare> mset; //compare same as min heap
        for (vector<int>& point : points) {
            mset.insert(point);
            if (mset.size() > K) {
                mset.erase(mset.begin());
            }
        }
        vector<vector<int>> ans;
        copy_n(mset.begin(), K, back_inserter(ans));
        return ans;
    }


 vector<vector<int>> kClosest(vector<vector<int>>& points, int K) {
        multiset<vector<int>, compare> mset(points.begin(), points.end()); // compare same as maxheap
        vector<vector<int>> ans;
        copy_n(mset.begin(), K, back_inserter(ans));
        return ans;
    }

// priority_queue ---> TC : nlog(k)

struct compare {
        bool operator()(vector<int>& p, vector<int>& q) {
        	// for min heap this condition will be reversed
            return p[0] * p[0] + p[1] * p[1] < q[0] * q[0] + q[1] * q[1];
        }
    };
vector<vector<int>> kClosest(vector<vector<int>>& points, int K) {
        priority_queue<vector<int>, vector<vector<int>>, compare> pq;
        for (vector<int>& point : points) {
            pq.push(point);
            if (pq.size() > K) {
                pq.pop();
            }
        }
        vector<vector<int>> ans;
        while (!pq.empty()) {
            ans.push_back(pq.top());
            pq.pop();
        }
        return ans;
    }

// min heap --> n+ nlog(k)
// vector<vector<int>> kClosest(vector<vector<int>>& points, int K) {
//         priority_queue<vector<int>, vector<vector<int>>, compare> pq(points.begin(), points.end());
//         vector<vector<int>> ans;
//         for (int i = 0; i < K; i++) {
//             ans.push_back(pq.top());
//             pq.pop();
//         }
//         return ans;
//     }