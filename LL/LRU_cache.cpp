146. LRU Cache
====================
Implement the LRUCache class:

 
Example 1:

Input
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
Output
[null, null, null, 1, null, -1, null, -1, 3, 4]

Explanation
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // cache is {1=1}
lRUCache.put(2, 2); // cache is {1=1, 2=2}
lRUCache.get(1);    // return 1
lRUCache.put(3, 3); // LRU key was 2, evicts key 2, cache is {1=1, 3=3}
lRUCache.get(2);    // returns -1 (not found)
lRUCache.put(4, 4); // LRU key was 1, evicts key 1, cache is {4=4, 3=3}
lRUCache.get(1);    // return -1 (not found)
lRUCache.get(3);    // return 3
lRUCache.get(4);    // return 4




class Node {
public:
    int key, value;
    Node *prev, *next;
    Node() : key(0), value(0), prev(NULL), next(NULL) {}
    Node(int k, int v): key(k), value(v), prev(NULL), next(NULL) {}
};

class LRUCache {
private:
    int capacity, size;
    unordered_map<int, Node*> nodes;
    Node *head = new Node(), *tail = new Node();
    
    void add(Node* node) {
        Node* next = head -> next;
        head -> next = node;
        node -> next = next;
        node -> prev = head;
        next -> prev = node;
    }
    
    void remove(Node* node) {
        Node *prev = node -> prev, *next = node -> next;
        prev -> next = next;
        next -> prev = prev;
    }
    
    void update(Node* node) {
        remove(node);
        add(node);
    }
    
    void popBack() {
        Node* node = tail -> prev;
        remove(node);
        nodes.erase(node -> key);
        delete node;
    }
public:
    LRUCache(int capacity) {   // initates with +ve capacity value
        this -> capacity = capacity;
        this -> size = 0;
        head -> next = tail;
        tail -> prev = head;
    }
    
    int get(int key) { 
        if (nodes.find(key) == nodes.end()) { //key doesn't exist
            return -1;
        }
        update(nodes[key]);
        return nodes[key] -> value; //return key
    }
    
    void put(int key, int value) { 
        if (nodes.find(key) != nodes.end()) {
            nodes[key] -> value = value;
            update(nodes[key]);     //update key if it exist
        } else {
            Node* node = new Node(key, value); //add key-value pair
            nodes[key] = node;
            add(node);
            if (size == capacity) { //If the number of keys exceeds the capacity from this operation, evict the least recently used key.
                popBack();
            } else {
                size++;
            }
        }
    }
};

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache* obj = new LRUCache(capacity);
 * int param_1 = obj->get(key);
 * obj->put(key,value);
 */