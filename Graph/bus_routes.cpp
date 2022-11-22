815. Bus Routes
==================
You are given an array routes representing bus routes where routes[i] 
is a bus route that the ith bus repeats forever.

    For example, if routes[0] = [1, 5, 7], this means that the 0th bus 
    travels in the sequence 1 -> 5 -> 7 -> 1 -> 5 -> 7 -> 1 -> ... forever.

You will start at the bus stop source (You are not on any bus initially), 
and you want to go to the bus stop target. You can travel between bus stops 
by buses only.

Return the least number of buses you must take to travel from source to 
target. Return -1 if it is not possible.

 

Example 1:

Input: routes = [[1,2,7],[3,6,7]], source = 1, target = 6
Output: 2
Explanation: The best strategy is take the first bus to the bus stop 7, t
hen take the second bus to the bus stop 6.

Example 2:

Input: routes = [[7,12],[4,5,15],[6],[15,19],[9,12,13]], source = 15, 
       target = 12
Output: -1

//https://www.youtube.com/watch?v=R58Q0J52qzI
// TC : V+E

int numBusesToDestination(vector<vector<int>>& routes, int source, int target) {
        int n=routes.size();
        unordered_map<int,unordered_set<int>> stop_routes;
        for(int i=0;i<n;++i){
            for(int j : routes[i])
                stop_routes[j].insert(i);
        }
        queue<pair<int,int>> to_visit; //stop_no,no_of_routes/no_of_buses
        unordered_set<int> stop_visited={source};
        to_visit.push({source,0});
        while(!to_visit.empty()){
            auto[stop,bus_n]=to_visit.front();
            to_visit.pop();
            if(stop==target) return bus_n;
            for(auto route : stop_routes[stop]){
                for(auto next_stop : routes[route]){
                    auto it=stop_visited.insert(next_stop);
                    if(it.second) to_visit.push({next_stop,bus_n+1});
                }
                 routes[route].clear();
            }
        }
        return -1;
    }

// TC: V+E

class Solution {
public:
    int numBusesToDestination(vector<vector<int>>& routes, int S, int T) {
        if(S == T)
            return 0;
        // For some reason leetcode decided that putting a testcase where
        // they concatenate several cycles together like 1->2->1->2->1->2
        // and also call it a bus route even though it doesn't really make sense.
        // So, use unordered_set as protection against this bs.
        unordered_map<int, unordered_set<int>> graph;
        for(int r = 0; r < routes.size(); r++){
            for(auto bs : routes[r]){
                graph[bs].insert(r);
            }
        }
        
        // visited has all the routes we have been to.
        unordered_set<int> visited;
        
        // queue contains integers that represent the index of the route.
        queue<int> q;
        
        // retrieve routes for the starting bus stop.
        for(auto route : graph[S]){
            q.push(route);
            visited.insert(route);
        }
        
        // all of them require you to take one bus.
        int dist = 1;
        
        while(q.size()){
            int size = q.size();
            for(int i = 0; i < size; i++){
                int route = q.front(); q.pop();
                int start = -1;
				
                // visit each bus stop on this route
                for(auto bs: routes[route]){
                    // this is just to avoid cycle iteration as 
                    // in the example above.
                    if(start == -1)
                        start = bs;
                    else if(start == bs)
                        break;
                    
                    // happens to be your destination
                    if(bs == T)
                        return dist;
                    
                    // to which routes can i transit from this bus stop or bullshit :)
                    for(auto access_routes : graph[bs]){
                        if(!visited.count(access_routes)){
                            visited.insert(access_routes);
                            q.push(access_routes);
                        }
                    }
                }
            }
            dist++;
        }
        return -1;
    }
};