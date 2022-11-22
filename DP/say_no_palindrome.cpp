D. Say No to Palindromes #Educational Codeforces Round 112 (Rated for Div. 2)
===============================================================================
// Let's call the string beautiful if it does not contain a substring of length at 
// least 2 , which is a palindrome. 

// Let's define cost of a string as the minimum number of operations so that the 
// string becomes beautiful, if in one operation it is allowed to change any 
// character of the string to one of the first 3 letters of the Latin 
// alphabet (in lowercase).

// You are given a string s of length n, each character of the string is one of 
// the first 3 letters of the Latin alphabet (in lowercase).

// You have to answer m queries — calculate the cost of the substring of the 
// string s from l-th to r-th position, inclusive.


// Input

// The first line contains two integers n and m (1≤n,m≤2⋅105) — the length of the 
// string s and the number of queries.

// The second line contains the string s, it consists of n characters, 
// each character one of the first 3 Latin letters.

// The following m lines contain two integers l and r (1≤l≤r≤n) — parameters 
// of the i-th query.

// Output

// For each query, print a single integer — the cost of the substring of the 
// string s from l-th to r-th position, inclusive.

// 5 4
// baacb
// 1 3
// 1 5
// 4 5
// 2 3

// 1
// 2
// 0
// 1

// Note

// Consider the queries of the example test.

// in the first query, the substring is baa, which can be changed to bac in one 
// operation;
// in the second query, the substring is baacb, which can be changed to cbacb in 
// two operations;
// in the third query, the substring is cb, which can be left unchanged;
// in the fourth query, the substring is aa, which can be changed to ba in one
// operation. 



// Note that in the beautiful string s[i]≠s[i−1] (because it is a palindrome of 
// length 2) and s[i]≠s[i−2] (because it is a palindrome of length 3). 
// This means s[i]=s[i−3], i.e. a beautiful string has the form abcabcabc..., up to 
// the permutation of the letters a, b and c.

// For each permutation of the letters a, b and c, we will construct a string t, 
// of the form abcabcabc... of length n. 
// Let''s define an array a of length n as follows: 
// a[i]=0 if s[i]=t[i] (i.e. the character at the i-th position does not 
// need to be changed) and a[i]=1 otherwise. 
// Let''s build an array pr of prefix sums of the array a. 
// Now you can process a query of the number of positions that need to be 
// replaced for the current line t in O(1).

// t   cur
// abc 1
// acb 2
// bac 3
// bca 4
// cab 5
// cba 6

// pr 
// 0 1 2 3 4 4 (abca|b|c & baac|b|} -> only match b)
// 0 1 2 3 4 5 
// 0 0 0 1 2 3 (|ba|cbac & |ba|acb)
// 0 0 1 1 2 3 (|b|c|a|bca & |b|a|a|cb)
// 0 1 1 2 2 3 
// 0 1 2 2 2 2 

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL); 
  int n, m;
  cin >> n >> m;
  string s;
  cin >> s;
  vector<vector<int>> pr(6, vector<int>(n + 1));
  string t = "abc";
  int cur = 0;
  do {
    for (int i = 0; i < n; ++i) // check column-wise
      pr[cur][i + 1] = pr[cur][i] + (s[i] != t[i % 3]);
    ++cur;
  } while (next_permutation(t.begin(), t.end()));
  while (m--) {
    int l, r;
    cin >> l >> r;
    int ans = n;
    for (int i = 0; i < 6; ++i)
      ans = min(ans, pr[i][r] - pr[i][l - 1]);
    cout << ans << "\n";
  }
}