139. Word Break
=================
// Given a string s and a dictionary of strings wordDict, return true if s can be 
// segmented into a space-separated sequence of one or more dictionary words.

// Note that the same word in the dictionary may be reused multiple times in 
// the segmentation.

 

// Example 1:

// Input: s = "leetcode", wordDict = ["leet","code"]
// Output: true
// Explanation: Return true because "leetcode" can be segmented as "leet code".

// Example 2:

// Input: s = "applepenapple", wordDict = ["apple","pen"]
// Output: true
// Explanation: Return true because "applepenapple" can be segmented as 
// "apple pen apple".
// Note that you are allowed to reuse a dictionary word.

// Example 3:

// Input: s = "catsandog", wordDict = ["cats","dog","sand","and","cat"]
// Output: false

// // fail for :
// // "bb"
// // ["a","b","bbb","bbbb"] 

// // correct : true - wrong : false

// s="abcd"   
// dict=["a","b","c","ab","bc","abc"]


//                                   "abcd"
//                                    |
//                   --------------------------------------------------
//         (contain "a") |        (contain "ab")   |           |      |
//                     "bcd"                      "cd"        "d"    ""
//                  	  |
//                  -----------
//                  |    |    |
//                 "cd"  "d"  ""
//                  |     |
//                  |     ""
//              --------
//              |      |
//             "d"     ""
//              |
//              ""



bool wordBreak(string s, vector<string>& wordDict) {
	    int prev_idx=-1;
        for(int i=0;i<wordDict.size();++i){
        	if(s.find(wordDict[i])!=string::npos){
        		int cur_idx=s.find(wordDict[i]);
        	    if(prev_idx<cur_idx) prev_idx=cur_idx;
        	    else{
        	    	int last_idx=s.find_last_of(wordDict[i]);
        	    	if(last_idx<prev_idx) return false;
        	    }
        	}
        	else return false;
        }
        return true;
    }

// Memoization , 36% faster , 20% less memory

class Solution {
private:
    unordered_set<string> st;
	unordered_map<int,bool> dp;
    bool helper(string s,int pos){
        if(s.size()==pos) return true;
        if(dp.count(pos)) return dp[pos];
        for(int i=pos;i<s.size();++i){
            if(st.count(s.substr(pos,i-pos+1))&&helper(s,i+1))
                return dp[pos]=true;
        }
        return dp[pos]=false;
    }
public:
bool wordBreak(string s, vector<string>& wordDict) {
        for(int i=0;i<wordDict.size();++i)
            st.insert(wordDict[i]);
        return helper(s,0);
    }
};

//Tabulation , faster than 12% , 26% less memory

class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        int n = s.size();
        bool dp[n+1][n+1];
        
        unordered_set<string> myset;
        for(auto word: wordDict)
            myset.insert(word);
        
        for(int length=1;length<=n;++length) //Window Size
        {
            int start = 1;
            int end = length;
            while(end<=n) //Sliding Window
            {
                string temp = s.substr(start-1,length);
                if(myset.find(temp)!=myset.end())
                    dp[start][end] = true;
                else
                {
                    bool flag = false;
                    for(int i=start;i<end;++i)
                        if(dp[start][i] and dp[i+1][end])
                        {
                            flag = true;
                            break;
                        }
                    dp[start][end] = flag;
                }
                start += 1;
                end += 1;
            }
        }
        return dp[1][n];
    }
};

// BFS , 90% faster, 77% less memory

class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        queue<int> q;//Store start position for each new search query
        q.push(0);//We must start from pos 0
        unordered_set<int> visited;//Keeps already processed positions
        unordered_set<string> dict;//wordDict set
        
        //Store wordDict in dict-set for easy search
        for(string it: wordDict)
            dict.insert(it);
        
        while(!q.empty()){
            int curr = q.front();
            q.pop();
            
            if(!visited.count(curr)){//Don't re-process a pos
                visited.insert(curr);
                string temp="";
                for(int i=curr;i<s.size();++i){
                    temp.push_back(s[i]);
                    if(dict.count(temp)){
                        q.push(i+1);
                        if(i==s.size()-1)
                            return true;
                    }
                }
            }
        }
        return false;
    }
};


140. Word Break II
=====================
// Given a string s and a dictionary of strings wordDict, add spaces in s to 
// construct a sentence where each word is a valid dictionary word. Return all 
// such possible sentences in any order.

// Note that the same word in the dictionary may be reused multiple times in 
// the segmentation.

 

// Example 1:

// Input: s = "catsanddog", wordDict = ["cat","cats","and","sand","dog"]
// Output: ["cats and dog","cat sand dog"]

// Example 2:

// Input: s = "pineapplepenapple", wordDict = ["apple","pen","applepen","pine",
//                 "pineapple"]
// Output: ["pine apple pen apple","pineapple pen apple","pine applepen apple"]
// Explanation: Note that you are allowed to reuse a dictionary word.

// Example 3:

// Input: s = "catsandog", wordDict = ["cats","dog","sand","and","cat"]
// Output: []

// 55% faster , 60% less memory


class Solution {
    vector<string> ans;//Store all valid sentences
    struct trienode{
        char c;
        int we;
        trienode *child[26];
        trienode(char c){
            we = 0;
            this->c = c;
            for(int i=0;i<26;++i)
                child[i]=NULL;
        }
    };
    trienode *root;//root of TRIE
    void insertInTrie(const string &word){
        trienode *curr = root;
        int idx;
        for(int i=0;i<word.size();++i){
            idx = word[i]-'a';
            if(!curr->child[idx])
                curr->child[idx] = new trienode(char(97+idx));
            curr = curr->child[idx];
        }
        curr->we += 1;
    }
    bool searchInTrie(string s){
        trienode *curr = root;
        int idx;
        for(int i=0;i<s.size();++i){
            idx = s[i]-'a';
            if(!curr->child[idx])
                return false;
            curr = curr->child[idx];
        }
        return curr->we>0;
    }
    
    void solve(const string &s,string st,int pos){
        if(pos==s.size()){
            ans.push_back(st);
            return;
        }
        st += " ";
        for(int i=pos;i<s.size();++i){
            if(searchInTrie(s.substr(pos,i+1-pos)))
                solve(s,st+s.substr(pos,i+1-pos),i+1);
        }
    }
public:
    vector<string> wordBreak(string s, vector<string>& wordDict) {
        root = new trienode('/');
        for(auto word: wordDict)
            insertInTrie(word);
        
        for(int i=0;i<s.size();++i){
            if(searchInTrie(s.substr(0,i+1)))
                solve(s,s.substr(0,i+1),i+1);
        }
        return ans;
    }
};

// TC : n^2  , 100% faster , 17% less memory

class Solution {
public:
    vector<string> wordBreak(string s, vector<string>& wordDict) {
        //insert all the words in the set
        unordered_set<string> set;
        vector<string> res;
        for(auto word:wordDict)
            set.insert(word);
        //to store the current string 
        string curr="";
        findHelper(0,s,curr,set,res);
        return res;
    }
    
    void findHelper(int ind,string s,string curr,unordered_set<string> set,
                            vector<string>& res)
    {
        if(ind==s.length())
        {
            //we have reached end
            curr.pop_back(); //remove the trailing space
            res.push_back(curr);
        }
        string str="";
        for(int i=ind;i<s.length();i++)
        {
            //get every substring and check if it exists in set
            str.push_back(s[i]);
            if(set.count(str))
            {
                //we have got a word in dict 
                //explore more and get other substrings
                findHelper(i+1,s,curr+str+" ",set,res);
            }
        }
    }
};
