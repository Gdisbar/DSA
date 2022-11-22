1268. Search Suggestions System
=================================
// You are given an array of strings products and a string searchWord.

// Design a system that suggests at most three product names from products 
// after each character of searchWord is typed. Suggested products should have 
// common prefix with searchWord. If there are more than three products with a 
// common prefix return the three lexicographically minimums products.

// Return a list of lists of the suggested products after each character of 
// searchWord is typed.

 

// Example 1:

// Input: products = ["mobile","mouse","moneypot","monitor","mousepad"], 
// searchWord = "mouse"
// Output: [["mobile","moneypot","monitor"],["mobile","moneypot","monitor"],
// ["mouse","mousepad"],["mouse","mousepad"],["mouse","mousepad"]]
// Explanation: products sorted lexicographically = ["mobile","moneypot",
// "monitor","mouse","mousepad"].
// After typing m and mo all products match and we show user ["mobile",
// "moneypot","monitor"].
// After typing mou, mous and mouse the system suggests ["mouse","mousepad"].

// Example 2:

// Input: products = ["havana"], searchWord = "havana"
// Output: [["havana"],["havana"],["havana"],["havana"],["havana"],["havana"]]
// Explanation: The only word "havana" will be always suggested while 
// typing the search word.

// In a sorted list of words,
// for any word A[i],
// all its sugested words must following this word in the list.

// For example, if A[i] is a prefix of A[j],
// A[i] must be the prefix of A[i + 1], A[i + 2], ..., A[j]

// Explanation

// With this observation,
// we can binary search the position of each prefix of search word,
// and check if the next 3 words is a valid suggestion.

// Complexity

// Time O(NlogN) for sorting
// Space O(logN) for quick sort.

// Time O(logN) for each query
// Space O(query) for each query
// where I take word operation as O(1)

vector<vector<string>> suggestedProducts(vector<string>& A, string searchWord) {
        auto it = A.begin();
        sort(it, A.end());
        vector<vector<string>> res;
        string cur = "";
        for (char c : searchWord) {
            cur += c;
            vector<string> suggested;
            it = lower_bound(it, A.end(), cur);
            for (int i = 0; i < 3 && it + i != A.end(); i++) {
                string& s = *(it + i);
                if (s.find(cur)) break;
                suggested.push_back(s);
            }
            res.push_back(suggested);
        }
        return res;
    }


// using Trie 

class Solution {
private:
    struct Node{
        unordered_map<int, Node*> map;
        vector<string> sugg;
        
        Node(){
            map = {};
            sugg = {};
        }
    };
    
    struct Node *root;
    
    void insert(string s){
        Node *curr = root;
        
        for(char c : s){
            if(curr->map.find(c) == curr->map.end()){
                curr->map[c] = new Node;
            }
            curr = curr->map[c];
            if(curr->sugg.size() < 3) curr->sugg.push_back(s);
        }
    }
    
    vector<string> search(char c, Node *&curr){
        
        if(curr->map.find(c) == curr->map.end()){
            curr = NULL;
            return {};
        }
        curr = curr->map[c];
        return curr->sugg;
    }
    
    
public:
    vector<vector<string>> suggestedProducts(vector<string>& products, 
    	string searchWord) {
        
        root = new Node;
        vector<vector<string>> out;
        
        sort(products.begin(), products.end());
        
        for(string s : products){
            insert(s);
        }
        
        Node *curr = root;
        
        for(char c : searchWord){
            if(!curr) out.push_back({});
            else out.push_back(search(c, curr));
        }
        
        return out;
    }
};



// Trie + Sort

// Sort
// n = number of charcters in the products list
// Time: O(nlog(n))
// Build Trie
// k = 3
// m = number of characters in the longest product
// Time: O(n)
// Space: O(nkm)
// Output Result
// s = number of characters in searchword
// Time: O(s)
// Space: O(sk)

class Solution:
    def suggestedProducts(self, products: List[str], searchWord: str) -> List[List[str]]:
        class TrieNode:
            def __init__(self):
                self.children = collections.defaultdict(TrieNode)
                self.suggestion = []
            
            def add_sugestion(self, product):
                if len(self.suggestion) < 3:
                    self.suggestion.append(product)
        
        products = sorted(products)
        root = TrieNode()
        for p in products:
            node = root
            for char in p:
                node = node.children[char]
                node.add_sugestion(p)
        
        result, node = [], root
        for char in searchWord:
            node = node.children[char]
            result.append(node.suggestion)
        return result

// Trie + Heap

// Build Trie
// k = 3
// n = number of charcters in the products list
// m = number of characters in the longest product
// Time: O(nklog(k))
// Space: O(nkm)
// Output Result
// s = number of characters in searchword
// Time: O(sklog(k))
// Space: O(sk)

class Solution:
    def suggestedProducts(self, products: List[str], searchWord: str) -> List[List[str]]:
        class TrieNode:
            def __init__(self):
                self.children = collections.defaultdict(TrieNode)
                self.h = []
            
            def add_sugesstion(self, product):
                if len(self.h) < 3:
                    heapq.heappush(self.h, MaxHeapStr(product))
                else:
                    heapq.heappushpop(self.h, MaxHeapStr(product))
            
            def get_suggestion(self):
                return sorted(self.h, reverse = True)
        
        class MaxHeapStr(str):
            def __init__(self, string): self.string = string
            def __lt__(self,other): return self.string > other.string
            def __eq__(self,other): return self.string == other.string
        
        root = TrieNode()
        for p in products:
            node = root
            for char in p:
                node = node.children[char]
                node.add_sugesstion(p)
        
        result, node = [], root
        for char in searchWord:
            node = node.children[char]
            result.append(node.get_suggestion())
        return result
