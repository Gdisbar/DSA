Friends Pairing Problem
==========================
// Given n friends, each one can remain single or can be paired up with some 
// other friend. Each friend can be paired only once. Find out the total number 
// of ways in which friends can remain single or can be paired up. 

// Examples: 

// Input  : n = 3
// Output : 4
// Explanation:
// {1}, {2}, {3} : all single
// {1}, {2, 3} : 2 and 3 paired but 1 is single.
// {1, 2}, {3} : 1 and 2 are paired but 3 is single.
// {1, 3}, {2} : 1 and 3 are paired but 2 is single.
// Note that {1, 2} and {2, 1} are considered same.

// Mathematical Explanation:
// The problem is simplified version of how many ways we can divide n elements 
// into multiple groups.
// (here group size will be max of 2 elements).
// In case of n = 3, we have only 2 ways to make a group: 
//     1) all elements are individual(1,1,1)
//     2) a pair and individual (2,1)
// In case of n = 4, we have 3 ways to form a group:
//     1) all elements are individual (1,1,1,1)
//     2) 2 individuals and one pair (2,1,1)
//     3) 2 separate pairs (2,2)

int countFriendsPairings(int n)
{
    int dp[n + 1];
 
    // Filling dp[] in bottom-up manner using
    // recursive formula explained above.
    for (int i = 0; i <= n; i++) {
        if (i <= 2)
            dp[i] = i;
        else
            dp[i] = dp[i - 1] + (i - 1) * dp[i - 2]; //individal + (i-1) pairs
    }
 
    return dp[n];
}
 

// Space optimized

int countFriendsPairings(int n)
{
    int a = 1, b = 2, c = 0;
    if (n <= 2) {
        return n;
    }
    for (int i = 3; i <= n; i++) {
        c = b + (i - 1) * a;
        a = b;
        b = c;
    }
    return c;
}
 
1583. Count Unhappy Friends
============================
// You are given a list of preferences for n friends, where n is always even.

// For each person i, preferences[i] contains a list of friends sorted in the 
// order of preference. In other words, a friend earlier in the list is more 
// preferred than a friend later in the list. Friends in each list are denoted 
// by integers from 0 to n-1.

// All the friends are divided into pairs. The pairings are given in a list pairs, 
// where pairs[i] = [xi, yi] denotes xi is paired with yi and yi is paired with xi.

// However, this pairing may cause some of the friends to be unhappy. A friend x 
// is unhappy if x is paired with y and there exists a friend u who is paired 
// with v but:

//     x prefers u over y, and
//     u prefers x over v.

// Return the number of unhappy friends.

 

// Example 1:

// Input: n = 4, preferences = [[1, 2, 3], [3, 2, 0], [3, 1, 0], [1, 2, 0]], 
// pairs = [[0, 1], [2, 3]]

// Output: 2
// Explanation:
// Friend 1 is unhappy because:
// - 1 is paired with 0 but prefers 3 over 0, and
// - 3 prefers 1 over 2.
// Friend 3 is unhappy because:
// - 3 is paired with 2 but prefers 1 over 2, and
// - 1 prefers 3 over 0.
// Friends 0 and 2 are happy.

// Example 2:

// Input: n = 2, preferences = [[1], [0]], pairs = [[1, 0]]
// Output: 0
// Explanation: Both friends 0 and 1 are happy.

// Example 3:

// Input: n = 4, preferences = [[1, 3, 2], [2, 3, 0], [1, 3, 0], [0, 2, 1]], 
// pairs = [[1, 3], [0, 2]]

// Output: 4

// Important thing to note:
// When
// x prefers u over y, and
// u prefers x over v,
// Then
// Both x and u are unhappy.
// NOT just x.

// The problem statement mentions that x is unhappy. We have to figure out 
// that u is also unhappy.

// --

// You have to try each pair with every other pair to identify the unhappiness. 
// That leads to O(n^2) complexity already. So how do we optimize the inner 
// operation to constant time so that the overall complexity remains O(n^2)?

// To know whether x prefers y over z, we need to know their positions in xs list.
// If you want to find the position of y in xs list, you might have to search 
// the entire list and get the index. Similarly for z.
// This would be O(n) and solution will become O(n^3) which will lead to TLE.

// Hence, we first store the position of each friend in every other friends 
// list, in a map at the beginning.

class Solution {

    // We store the index of each person in every other person's list in a 
    // map at the beginning.
    
    // `positions[i][j]` should be read as position of i in the list of j 
    // is positions[i][j].
    unordered_map<int, unordered_map<int, int>> positions;
    // Stores unhappy people. In the end, we will return it's size.
    unordered_set<int> unhappy; 
public:
    void checkHappiness(int x, int y, int u, int v) {
        if (positions[u][x] < positions[y][x] &&
            positions[x][u] < positions[v][u]) {
            unhappy.insert(x);
            unhappy.insert(u);
        }
    }
    
    int unhappyFriends(int n, vector<vector<int>>& preferences, 
                            vector<vector<int>>& pairs) {
        for (int i=0; i<n; i++) {
            for (int j=0; j<n-1; j++) {
                positions[preferences[i][j]][i] = j;
            }
        }
        int n = pairs.size();
        for (int i=0; i<n-1; i++) {
            for (int j=i+1; j<n; j++) {
                int x = pairs[i][0], y = pairs[i][1], u = pairs[j][0], v = pairs[j][1];
                checkHappiness(x, y, u, v); // If x prefers u over y,  and u prefers x over v
                checkHappiness(x, y, v, u); // If x prefers v over y,  and v prefers x over u
                checkHappiness(y, x, u, v); // If y prefers u over x,  and u prefers y over v
                checkHappiness(y, x, v, u); // If y prefers v over x,  and v prefers y over u
            }
        }
        
        return unhappy.size();
    }
};


