899. Orderly Queue
===================
// You are given a string s and an integer k. You can choose one of the first k 
// letters of s and append it at the end of the string.Return the lexicographically 
// smallest string you could have after applying the mentioned step any number of moves.


// Example 1:

// Input: s = "cba", k = 1
// Output: "acb"
// Explanation: 
// In the first move, we move the 1st character 'c' to the end, obtaining the string 
// "bac".
// In the second move, we move the 1st character 'b' to the end, obtaining the 
// final result "acb".

// Example 2:

// Input: s = "baaca", k = 3
// Output: "aaabc"
// Explanation: 
// In the first move, we move the 1st character 'b' to the end, obtaining the string 
// "aacab".
// In the second move, we move the 3rd character 'c' to the end, obtaining the final 
// result "aaabc".

    string orderlyQueue(string S, int K) {
        if (K > 1) {
            sort(S.begin(), S.end());
            return S;
        }
        string res = S;
        for (int i = 1; i < S.length(); i++)
            res = min(res, S.substr(i) + S.substr(0, i));
        return res;
    }

// another approach - Lyndon factorization

string orderlyQueue(string S, int K)
{
    if(K >= 2)
    {
        sort(S.begin(), S.end());
        return S;
    }
    int N = (int)S.size();
    int i = 0, ans = 0;
    while(i < N)
    {
        ans = i;
        int k = i, j = i + 1;
        while(j - i < N && S[k] <= S[j % N])
        {
            if(S[k] < S[j % N]) k = i;
            else k++;
            j++;
        }
        while(i <= k) i += j - k;
    }
    return S.substr(ans) + S.substr(0, ans);
}

def orderlyQueue(self, s: str, k: int) -> str:
    if k==1:
        tmp=s
        for i in range(len(s)-1):
            s=s[1:]+s[0]
            tmp=min(tmp,s)
        return tmp
    else:
        return "".join([str(x)for x in sorted(s)])

def orderlyQueue(self, S, K):
    return "".join(sorted(S)) if K > 1 else min(S[i:] + S[:i] for i in range(len(S)))