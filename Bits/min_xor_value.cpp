Min XOR value
================
Given an array of integers. Find the pair in an array that has a minimum XOR value. 

Examples : 

Input : arr[] =  {9, 5, 3}
Output : 6
        All pair with xor value (9 ^ 5) => 12, 
        (5 ^ 3) => 6, (9 ^ 3) => 10.
        Minimum XOR value is 6

Input : arr[] = {1, 2, 3, 4, 5}
Output : 1 

// Returns minimum xor value of pair in arr[0..n-1]
int minXOR(int arr[], int n)
{
    int min_xor = INT_MAX; // Initialize result
 
    // Generate all pair of given array
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
 
            // update minimum xor value if required
            min_xor = min(min_xor, arr[i] ^ arr[j]);
 
    return min_xor;
}
 


 // Function to find minimum XOR pair
int minXOR(int arr[], int n)
{
    // Sort given array
    sort(arr, arr + n);
 
    int minXor = INT_MAX;
    int val = 0;
 
    // calculate min xor of consecutive pairs
    for (int i = 0; i < n - 1; i++) {
        val = arr[i] ^ arr[i + 1];
        minXor = min(minXor, val);
    }
 
    return minXor;
}
 
Output

6

Time Complexity: O(N*logN) 
Space Complexity: O(1) 

A further more Efficient solution can solve the above problem in O(n) time under the assumption that integers take fixed number of bits to store. The idea is to use Trie Data Structure.

    Algorithm:

        Create an empty trie. Every node of trie contains two children for 0 and 1 bits.
        Initialize min_xor = INT_MAX, insert arr[0] into trie
        Traversal all array element one-by-one starting from second.
            First find minimum setbet difference value in trie 
                do xor of current element with minimum setbit diff that value 
            update min_xor value if required
            insert current array element in trie 
        return min_xor
        
#define INT_SIZE 32
 
// A Trie Node
struct TrieNode {
    int value; // used in leaf node
    TrieNode* Child[2];
};
 
// Utility function to create a new Trie node
TrieNode* getNode()
{
    TrieNode* newNode = new TrieNode;
    newNode->value = 0;
    newNode->Child[0] = newNode->Child[1] = NULL;
    return newNode;
}
 
// utility function insert new key in trie
void insert(TrieNode* root, int key)
{
    TrieNode* temp = root;
 
    // start from the most significant bit, insert all
    // bit of key one-by-one into trie
    for (int i = INT_SIZE - 1; i >= 0; i--) {
        // Find current bit in given prefix
        bool current_bit = (key & (1 << i));
 
        // Add a new Node into trie
        if (temp->Child[current_bit] == NULL)
            temp->Child[current_bit] = getNode();
 
        temp = temp->Child[current_bit];
    }
 
    // store value at leafNode
    temp->value = key;
}
 
// Returns minimum XOR value of an integer inserted
// in Trie and given key.
int minXORUtil(TrieNode* root, int key)
{
    TrieNode* temp = root;
 
    for (int i = INT_SIZE - 1; i >= 0; i--) {
        // Find current bit in given prefix
        bool current_bit = (key & (1 << i));
 
        // Traversal Trie, look for prefix that has
        // same bit
        if (temp->Child[current_bit] != NULL)
            temp = temp->Child[current_bit];
 
        // if there is no same bit.then looking for
        // opposite bit
        else if (temp->Child[1 - current_bit] != NULL)
            temp = temp->Child[1 - current_bit];
    }
 
    // return xor value of minimum bit difference value
    // so we get minimum xor value
    return key ^ temp->value;
}
 
// Returns minimum xor value of pair in arr[0..n-1]
int minXOR(int arr[], int n)
{
    int min_xor = INT_MAX; // Initialize result
 
    // create a True and insert first element in it
    TrieNode* root = getNode();
    insert(root, arr[0]);
 
    // Traverse all array element and find minimum xor
    // for every element
    for (int i = 1; i < n; i++) {
        // Find minimum XOR value of current element with
        // previous elements inserted in Trie
        min_xor = min(min_xor, minXORUtil(root, arr[i]));
 
        // insert current array value into Trie
        insert(root, arr[i]);
    }
    return min_xor;
}

