208. Implement Trie (Prefix Tree)
======================================
A trie (pronounced as "try") or prefix tree is a tree data structure used to efficiently store and retrieve keys in a dataset of strings. There are various applications of this data structure, such as autocomplete and spellchecker.

Implement the Trie class:

    Trie() Initializes the trie object.
    void insert(String word) Inserts the string word into the trie.
    boolean search(String word) Returns true if the string word is in the trie (i.e., was inserted before), and false otherwise.
    boolean startsWith(String prefix) Returns true if there is a previously inserted string word that has the prefix prefix, and false otherwise.

 

Example 1:

Input
["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
[[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]
Output
[null, null, true, false, true, null, true]

Explanation
Trie trie = new Trie();
trie.insert("apple");
trie.search("apple");   // return True
trie.search("app");     // return False
trie.startsWith("app"); // return True
trie.insert("app");
trie.search("app");     // return True


//TC : max string length * queries , SC : single node(217 B) * # of nodes

#define MAX_NODES 10000
class Trie {
    struct Trienode
    {
        char val;  //1B
        int count;  //4B
        int endsHere; //4B
        Trienode *child[26]; //8*26=208B
    };
    Trienode *root;
    
    Trienode *getNode(int index) //say we add "a" 
    {
        Trienode *newnode = new Trienode;
        newnode->val = 'a'+index;
        //we're adding "a" for 1st time so count is 0 & "a" has no child i.e "a" ends here
        newnode->count = newnode->endsHere = 0; 
        for(int i=0;i<26;++i)
            newnode->child[i] = NULL; //a->[...] 26 positions to add next char that might form a word
        return newnode;       //a->p[...] => ap->p[...] => appl->[...] => apple'\0'
    }
public:
    /** Initialize your data structure here. */
    Trie() {
        ios_base::sync_with_stdio(false);
        cin.tie(NULL);
        root = getNode('/'-'a');
    }
    
    /** Inserts a word into the trie. */
    void insert(string word) {
        Trienode *curr = root;
        int index;
        for(int i=0;word[i]!='\0';++i)
        {
            index = word[i]-'a';
            if(curr->child[index]==NULL) //add next char as child of previous node
                curr->child[index] = getNode(index);
            curr->child[index]->count +=1; //if we've encountered the char in some other word
            curr = curr->child[index]; 
        }
        curr->endsHere +=1; //at the end of each word increase endsHere,so that if we insert any new word later we can separate them based on endsHere value
    }
    
    /** Returns if the word is in the trie. */
    bool search(string word) {
        Trienode *curr = root;
        int index;
        for(int i=0;word[i]!='\0';++i)
        {
            index = word[i]-'a';
            if(curr->child[index]==NULL)
                return false;
            curr = curr->child[index];
        }
        //if endsHere=0 then we might have found our string as a substring 
        // of other word but it''s not a complete string but part of a word
        return (curr->endsHere > 0);  
                                    
    }
    
    /** Returns if there is any word in the trie that starts with the given prefix. */
    bool startsWith(string prefix) {
        Trienode *curr = root;
        int index;
        for(int i=0;prefix[i]!='\0';++i)
        {
            index = prefix[i]-'a';
            if(curr->child[index]==NULL)
                return false;
            curr = curr->child[index];
        }
        return (curr->count > 0);
    }
};

/**
 * Your Trie object will be instantiated and called as such:
 * Trie* obj = new Trie();
 * obj->insert(word);
 * bool param_2 = obj->search(word);
 * bool param_3 = obj->startsWith(prefix);
 */