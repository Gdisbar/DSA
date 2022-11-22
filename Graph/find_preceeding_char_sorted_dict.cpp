Given a sorted dictionary of an alien language, find order of characters
=========================================================================
// Given a sorted dictionary (array of words) of an alien language, 
// find order of characters in the language.

// Examples:  

// Input:  words[] = {"baa", "abcd", "abca", "cab", "cad"}
// Output: Order of characters is 'b', 'd', 'a', 'c'
// Note that words are sorted and in the given language "baa" 
// comes before "abcd", therefore 'b' is before 'a' in output.
// Similarly we can find other orders.

// Input:  words[] = {"caa", "aaa", "aab"}
// Output: Order of characters is 'c', 'a', 'b'

// Approach 1: 

// The idea is to create a graph of characters and then find topological sorting 
// of the created graph. Following are the detailed steps.
// 1) Create a graph g with number of vertices equal to the size of alphabet in 
// the given alien language. For example, if the alphabet size is 5, then there 
// can be 5 characters in words. Initially there are no edges in graph.

// 2) Do following for every pair of adjacent words in given sorted array. 
// …..a) Let the current pair of words be word1 and word2. One by one compare 
// characters of both words and find the first mismatching characters. 
// …..b) Create an edge in g from mismatching character of word1 to that of word2.

// 3) Print topological sorting of the above created graph.

// TC :
//  there would be alpha vertices and at-most (n-1) edges in the graph. 
//  The time complexity of topological sorting is O(V+E) which is O(n + alpha) here. 
//  So overall time complexity is O(n + alpha) + O(n + alpha) which is O(n + alpha).

//  For example {“aba”, “bba”, “aaa”} is not valid, because from first two words, 
//  we can deduce ‘a’ should appear before ‘b’, but from last two words, 
//  we can deduce ‘b’ should appear before ‘a’ which is not possible. 

// Algorithm:

// (1) Compare 2 adjacent words at a time (i.e, word1 with word2, word2 with 
//     word3, … , word(startIndex) and word(startIndex + 1)

// (2) Then we compare one character at a time for the 2 words selected. 

// (2a) If both characters are different, we stop the comparison here and 
// conclude that the character from word(startIndex) comes before the other.  

// (2b) If both characters are the same, we continue to compare until (2a) 
// occurs or if either of the words has been exhausted. 

// (3) We continue to compare each word in this fashion until we have compared 
// all words. 

// If ‘N’ is the number of words in the input alien vocabulary/dictionary, 
// ‘L’ length of the max length word, and ‘C’ is the final number of unique 
// characters,

// Time Complexity: O(N * L) 

// Space Complexity: O(C)


 
// Class to represent a graph
class Graph
{
    int V;    // No. of vertices'
 
    // Pointer to an array containing adjacency listsList
    list<int> *adj;
 
    // A function used by topologicalSort
    void topologicalSortUtil(int v, bool visited[], stack<int> &Stack);
public:
    Graph(int V);   // Constructor
 
    // function to add an edge to graph
    void addEdge(int v, int w);
 
    // prints a Topological Sort of the complete graph
    void topologicalSort();
};
 
Graph::Graph(int V)
{
    this->V = V;
    adj = new list<int>[V];
}
 
void Graph::addEdge(int v, int w)
{
    adj[v].push_back(w); // Add w to v’s list.
}
 
// A recursive function used by topologicalSort
void Graph::topologicalSortUtil(int v, bool visited[], stack<int> &Stack)
{
    // Mark the current node as visited.
    visited[v] = true;
 
    // Recur for all the vertices adjacent to this vertex
    list<int>::iterator i;
    for (i = adj[v].begin(); i != adj[v].end(); ++i)
        if (!visited[*i])
            topologicalSortUtil(*i, visited, Stack);
 
    // Push current vertex to stack which stores result
    Stack.push(v);
}
 
// The function to do Topological Sort. It uses recursive topologicalSortUtil()
void Graph::topologicalSort()
{
    stack<int> Stack;
 
    // Mark all the vertices as not visited
    bool *visited = new bool[V];
    for (int i = 0; i < V; i++)
        visited[i] = false;
 
    // Call the recursive helper function to store Topological Sort
    // starting from all vertices one by one
    for (int i = 0; i < V; i++)
        if (visited[i] == false)
            topologicalSortUtil(i, visited, Stack);
 
    // Print contents of stack
    while (Stack.empty() == false)
    {
        cout << (char) ('a' + Stack.top()) << " ";
        Stack.pop();
    }
}
 
 
// This function finds and prints order of character from a sorted
// array of words. n is size of words[].  alpha is set of possible
// alphabets.
// For simplicity, this function is written in a way that only
// first 'alpha' characters can be there in words array.  For
// example if alpha is 7, then words[] should have only 'a', 'b',
// 'c' 'd', 'e', 'f', 'g'
void printOrder(string words[], int n, int alpha)
{
    // Create a graph with 'alpha' edges
    Graph g(alpha);
 
    // Process all adjacent pairs of words and create a graph
    for (int i = 0; i < n-1; i++)
    {
        // Take the current two words and find the first mismatching
        // character
        string word1 = words[i], word2 = words[i+1];
        for (int j = 0; j < min(word1.length(), word2.length()); j++)
        {
            // If we find a mismatching character, then add an edge
            // from character of word1 to that of word2
            if (word1[j] != word2[j])
            {
                g.addEdge(word1[j]-'a', word2[j]-'a');
                break;
            }
        }
    }
 
    // Print topological sort of the above created graph
    g.topologicalSort();
}
 
// Driver program to test above functions
int main()
{
    string words[] = {"caa", "aaa", "aab"};
    printOrder(words, 3, 3);
    return 0;
}