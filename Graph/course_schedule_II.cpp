210. Course Schedule II
==========================
There are a total of numCourses courses you have to take, labeled from 0 
to numCourses - 1. You are given an array prerequisites where 
prerequisites[i] = [ai, bi] indicates that you must take course b[i] first 
if you want to take course a[i].

    For example, the pair [0, 1], indicates that to take course 0 you have to 
first take course 1.

Return the ordering of courses you should take to finish all courses. 
If there are many valid answers, return any of them. If it is impossible to 
finish all courses, return an empty array.

 

Example 1:

Input: numCourses = 2, prerequisites = [[1,0]]
Output: [0,1]
Explanation: There are a total of 2 courses to take. To take course 1 you 
should have finished course 0. So the correct course order is [0,1].

Example 2:

Input: numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]
Output: [0,2,1,3]
Explanation: There are a total of 4 courses to take. To take course 3 you should 
have finished both courses 1 and 2. Both courses 1 and 2 should be taken after 
you finished course 0.
So one correct course order is [0,1,2,3]. Another correct ordering is [0,2,1,3].

Example 3:

Input: numCourses = 1, prerequisites = []
Output: [0]


class Solution {
public:
    vector<int> findOrder(int n, vector<vector<int>>& pre) {
        
        vector<vector<int>> adj(n);
        vector<int> indegree(n, 0);
        for(auto i : pre){
            
            int u = i[1], v = i[0];
            
            adj[u].push_back(v);
            indegree[v]++; // [0(u),1(v)] ==> 0--->1 , increase the indegree of v
        }
        
        queue<int> q;
        
        for(int i = 0; i < indegree.size();i++)
        {
            if(indegree[i] == 0)
                q.push(i); // insert the nodes not indegree[i]
                
        }
        
        vector<int> res;
        
        while(!q.empty()){
            
            int s = q.front();
            q.pop();
            res.push_back(s);
            for(auto child : adj[s]){
                
                indegree[child]--;
                if(indegree[child] == 0 )
                    q.push(child);
            }
            
        }
        
        
        if(res.size() == n)
            return res;
        else
            return {}; 
    }
};


// TopoSort - DFS

class Solution:
    def findOrder(self, N, P):
        G, indegree, ans = defaultdict(list), [0]*N, []
        for nxt, pre in P:
            G[pre].append(nxt)
            indegree[nxt] += 1
        
        def dfs(cur):
            ans.append(cur)
            indegree[cur] = -1
            for nextCourse in G[cur]:
                indegree[nextCourse] -= 1
                if indegree[nextCourse] == 0: 
                    dfs(nextCourse)            
        for i in range(N):
            if indegree[i] == 0:
                dfs(i)

        return ans if len(ans) == N else []

