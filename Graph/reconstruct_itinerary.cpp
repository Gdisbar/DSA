332. Reconstruct Itinerary
=============================
// You are given a list of airline tickets where tickets[i] = [from-i, to-i] 
// represent the departure and the arrival airports of one flight. 
// Reconstruct the itinerary in order and return it.

// All of the tickets belong to a man who departs from "JFK", thus, the 
// itinerary must begin with "JFK". If there are multiple valid itineraries, 
// you should return the itinerary that has the smallest lexical order when 
// read as a single string.

//     For example, the itinerary ["JFK", "LGA"] has a smaller lexical order 
//     than ["JFK", "LGB"].

// You may assume all tickets form at least one valid itinerary. You must use 
// all the tickets once and only once.

 

// Example 1:  
//                 "JFK" --> "MUC" ---> "LHR" --> "SFO" --> "SJC""


// Input: tickets = [["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]
// Output: ["JFK","MUC","LHR","SFO","SJC"]

// Example 2:					  |<--------|
// 							"SFO"---->"ATL"--|
//                                |       |     |
// 							   |       |     |
// 							   |---->"JFK"<---

// Input: tickets = [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],
//     ["ATL","SFO"]]
// Output: ["JFK","ATL","JFK","SFO","ATL","SFO"]
// Explanation: Another possible reconstruction is 
// ["JFK","SFO","ATL","JFK","ATL","SFO"] but it is larger in lexical order.

Eulearian Path
===================
//Is an algorithm that finds a path that uses every edge in a graph only once.

class Solution {
public:
    void dfs(unordered_map<string, multiset<string>>& graph,
             vector<string>& res, string start) {
        while (graph[start].size() > 0) {
            auto next = *graph[start].begin(); //next contain all adj list elements of head
            graph[start].erase(graph[start].begin()); //as we encounter new node , we remove them as we can use each edge once
            dfs(graph, res, next); //go to next node
        }
        res.push_back(start);
    }
    vector<string> findItinerary(vector<vector<string>>& tickets) {
        //store in sorted adj list using multiset
        unordered_map<string, multiset<string>> graph;
        for (const auto& t : tickets) graph[t[0]].insert(t[1]);
        vector<string> res;
        dfs(graph, res, "JFK");
        reverse(res.begin(), res.end());
        return res;
    }
};
