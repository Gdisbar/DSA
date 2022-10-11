Length of the longest substring without repeating characters
=============================================================
// Given a string str, find the length of the 
// longest substring without repeating characters. 

// Example:

//     For “ABDEFGABEF”, the longest substring are “BDEFGA” and “DEFGAB”, 
//     with length 6.

//     For “BBBB” the longest substring is “B”, with length 1.

//     For “GEEKSFORGEEKS”, there are two longest substrings shown in the 
//     below diagrams, with length 7


// n^3

// This function returns true if all characters in str[i..j]
// are distinct, otherwise returns false
bool areDistinct(string str, int i, int j)
{
 
    // Note : Default values in visited are false
    vector<bool> visited(26);
 
    for (int k = i; k <= j; k++) {
        if (visited[str[k] - 'a'] == true)
            return false;
        visited[str[k] - 'a'] = true;
    }
    return true;
}
 
// Returns length of the longest substring
// with all distinct characters.
int longestUniqueSubsttr(string str)
{
    int n = str.size();
    int res = 0; // result
    for (int i = 0; i < n; i++)
        for (int j = i; j < n; j++)
            if (areDistinct(str, i, j))
                res = max(res, j - i + 1);
    return res;
}
 

// n^2

int longestUniqueSubsttr(string str)
{
    int n = str.size();
    int res = 0; // result
 
    for (int i = 0; i < n; i++) {
         
        // Note : Default values in visited are false
        vector<bool> visited(256);  
 
        for (int j = i; j < n; j++) {
 
            // If current character is visited
            // Break the loop
            if (visited[str[j]] == true)
                break;
 
            // Else update the result if
            // this window is larger, and mark
            // current character as visited.
            else {
                res = max(res, j - i + 1);
                visited[str[j]] = true;
            }
        }
 
        // Remove the first character of previous
        // window
        visited[str[i]] = false;
    }
    return res;
}


// n

#define NO_OF_CHARS 256
 
int longestUniqueSubsttr(string str)
{
    int n = str.size();
 
    int res = 0; // result
 
    // last index of all characters is initialized
    // as -1
    vector<int> lastIndex(NO_OF_CHARS, -1);
 
    // Initialize start of current window
    int i = 0;
 
    // Move end of current window
    for (int j = 0; j < n; j++) {
 
        // Find the last index of str[j]
        // Update i (starting index of current window)
        // as maximum of current value of i and last
        // index plus 1
        i = max(i, lastIndex[str[j]] + 1);
 
        // Update result if we get a larger window
        res = max(res, j - i + 1);
 
        // Update last index of j.
        lastIndex[str[j]] = j;
    }
    return res;
}

# Python3 program to find the length
# of the longest substring
# without repeating characters
def longestUniqueSubsttr(string):
 
    # last index of every character
    last_idx = {}
    max_len = 0
 
    # starting index of current
    # window to calculate max_len
    start_idx = 0
 
    for i in range(0, len(string)):
       
        # Find the last index of str[i]
        # Update start_idx (starting index of current window)
        # as maximum of current value of start_idx and last
        # index plus 1
        if string[i] in last_idx:
            start_idx = max(start_idx, last_idx[string[i]] + 1)
 
        # Update result if we get a larger window
        max_len = max(max_len, i-start_idx + 1)
 
        # Update last index of current char.
        last_idx[string[i]] = i
 
    return max_len