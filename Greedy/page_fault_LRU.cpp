Page Faults in LRU
====================
// Given a sequence of pages in an array pages[] of length N and memory capacity C, 
// find the number of page faults using Least Recently Used (LRU) Algorithm. 


// Input: N = 9, C = 4
// pages = {5, 0, 1, 3, 2, 4, 1, 0, 5}
// Output: 8
// Explaination: memory allocated with 4 pages 5, 0, 1, 
// 3: page fault = 4
// page number 2 is required, replaces LRU 5: 
// page fault = 4+1 = 5
// page number 4 is required, replaces LRU 0: 
// page fault = 5 + 1 = 6
// page number 1 is required which is already present: 
// page fault = 6 + 0 = 6
// page number 0 is required which replaces LRU 3: 
// page fault = 6 + 1 = 7
// page number 5 is required which replaces LRU 2: 
// page fault = 7 + 1  = 8.

// Expected Time Complexity: O(N*C)
// Expected Auxiliary Space: O(N)

 int pageFaults(int N, int C, int pages[]){
        // code here
        deque<int> q(C); // waiting list
        q.clear();
        int page_fault=0;
        for(int i = 0;i < N;++i){
            auto it = find(q.begin(),q.end(),pages[i]);
            if(it==q.end()){  // page not present in set of pages in cache
                ++page_fault;
                if(q.size()==C){ // set can hold equal pages
                    q.erase(q.begin()); //pop LRU page
                    q.push_back(pages[i]); //push current page in waiting list
                }
                else{
                    q.push_back(pages[i]);
                }
            }
            else{
                q.erase(it); // page present , remove
                q.push_back(pages[i]);
            }
        }
        return page_fault;
    }


146. LRU Cache
======================
// Design a data structure that follows the constraints of a Least Recently 
// Used (LRU) cache.

// Implement the LRUCache class:

//     LRUCache(int capacity) Initialize the LRU cache with positive size capacity.
//     int get(int key) Return the value of the key if the key exists, otherwise 
//     return -1.
//     void put(int key, int value) Update the value of the key if the key exists. 
//     Otherwise, add the key-value pair to the cache. If the number of keys exceeds 
//     the capacity from this operation, evict the least recently used key.

// The functions get and put must each run in O(1) average time complexity.

 

// Example 1:

// Input
// ["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
// [[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
// Output
// [null, null, null, 1, null, -1, null, -1, 3, 4]

// Explanation
// LRUCache lRUCache = new LRUCache(2);
// lRUCache.put(1, 1); // cache is {1=1}
// lRUCache.put(2, 2); // cache is {1=1, 2=2}
// lRUCache.get(1);    // return 1
// lRUCache.put(3, 3); // LRU key was 2, evicts key 2, cache is {1=1, 3=3}
// lRUCache.get(2);    // returns -1 (not found)
// lRUCache.put(4, 4); // LRU key was 1, evicts key 1, cache is {4=4, 3=3}
// lRUCache.get(1);    // return -1 (not found)
// lRUCache.get(3);    // return 3
// lRUCache.get(4);    // return 4

class LRUCache{
    size_t m_capacity;
    unordered_map<int,list<pair<int, int>>::iterator> m_map; //m_map_iter->first: key, m_map_iter->second: list iterator;
    list<pair<int, int>> m_list;    //m_list_iter->first: key, m_list_iter->second: value;
public:
    LRUCache(size_t capacity):m_capacity(capacity) {
    }
    int get(int key) {
        auto found_iter = m_map.find(key);
        if (found_iter == m_map.end()) //key doesn't exist
            return -1;
        m_list.splice(m_list.begin(), m_list, found_iter->second); //move the node corresponding to key to front
        return found_iter->second->second;      //return value of the node
    }
    void set(int key, int value) {
        auto found_iter = m_map.find(key);
        if (found_iter != m_map.end()) //key exists
        {
            m_list.splice(m_list.begin(), m_list, found_iter->second); //move the node corresponding to key to front
            found_iter->second->second = value;    //update value of the node
            return;
        }
        if (m_map.size() == m_capacity) //reached capacity
        {
           int key_to_del = m_list.back().first; 
           m_list.pop_back();            //remove node in list;
           m_map.erase(key_to_del);      //remove key in map
        }
        m_list.emplace_front(key, value);  //create new node in list
        m_map[key] = m_list.begin();       //create correspondence between key and node
    }
};