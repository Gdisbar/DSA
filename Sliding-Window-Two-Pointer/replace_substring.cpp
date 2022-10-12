1234. Replace the Substring for Balanced String
--------------------------------------------------
You are given a string containing only 4 kinds of characters 'Q', 'W', 'E' and 'R'.

A string is said to be balanced if each of its characters appears n/4 times where n 
is the length of the string.

Return the minimum length of the substring that can be replaced with any other string of 
the same length to make the original string s balanced.

Return 0 if the string is already balanced.

 

Example 1:

Input: s = "QWER"
Output: 0
Explanation: s is already balanced.

Example 2:

Input: s = "QQWE"
Output: 1
Explanation: We need to replace a 'Q' to 'R', so that "RQWE" (or "QRWE") is balanced.


Intuition

We want a minimum length of substring,
which leads us to the solution of sliding window.
Specilly this time we don''t care the count of elements inside the window,
we want to know the count outside the window.

Explanation

One pass the all frequency of "QWER".
Then slide the windon in the string s.

Imagine that we erase all character inside the window,
as we can modyfy it whatever we want,
and it will always increase the count outside the window.

So we can make the whole string balanced,
as long as max(count[Q],count[W],count[E],count[R]) <= n / 4.

Important

Does i <= j + 1 makes more sense than i <= n.
Strongly don''t think, and i <= j + 1 makes no sense.

Answer the question first:
Why do we need such a condition in sliding window problem?

Actually, we never need this condition in sliding window solution
(Check all my other solutions link at the bottom).

Usually count the element inside sliding window,
and i won''t be bigger than j because nothing left in the window.

The only reason that we need a condition is in order to prevent index out of range.
And how do we do that? Yes we use i < n

    Does i <= j + 1 even work?
    When will i even reach j + 1?
    Does i <= j + 1 work better than i <= j?

Please upvote for this important tip.
Also let me know if there is any unclear, glad to hear different voices.
But please, have a try, and show the code if necessary.

Some people likes to criticize without even a try,
and solve the problem by talking.
Why talk to me? Talk to the white board.

Complexity

Time O(N), one pass for counting, one pass for sliding window
Space O(1)


class Solution {
public:
    int balancedString(string s) {
        int n = s.length(),m=n/4,ans=n,j=0;
        unordered_map<char,int> mp;
        for(int i = 0;i<n;i++)
            mp[s[i]]++;
        for(int i =0;i<n;i++){
            mp[s[i]]--;
            while(j<n&&mp['Q']<=m&&mp['W']<=m&&mp['E']<=m&&mp['R']<=m){
                ans=min(ans,i-j+1);
                //mp[s[j++]]++;
                mp[s[j]]++;
                j++;
            }
                
        }
            
            return ans;
    }
};