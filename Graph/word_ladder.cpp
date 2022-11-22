127. Word Ladder
==================
// A transformation sequence from word beginWord to word endWord using a 
// dictionary wordList is a sequence of words 
// beginWord -> s1 -> s2 -> ... -> sk such that:

//     Every adjacent pair of words differs by a single letter.
//     Every si for 1 <= i <= k is in wordList. Note that beginWord does not 
//     need to be in wordList.
//     sk == endWord

// Given two words, beginWord and endWord, and a dictionary wordList, 
// return the number of words in the shortest transformation sequence 
// from beginWord to endWord, or 0 if no such sequence exists.

 

// Example 1:

// Input: beginWord = "hit", endWord = "cog", 
// wordList = ["hot","dot","dog","lot","log","cog"]

// Output: 5
// Explanation: One shortest transformation sequence is 
// "hit" -> "hot" -> "dot" -> "dog" -> cog", which is 5 words long.

// Example 2:

// Input: beginWord = "hit", endWord = "cog", 
// wordList = ["hot","dot","dog","lot","log"]
// Output: 0
// Explanation: The endWord "cog" is not in wordList, therefore there is no valid 
// transformation sequence.

int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
        bool flag=false;
        unordered_set<string> set1;
        for(int i=0;i<wordList.size();++i){
            if(wordList[i].compare(endWord)==0)
                flag=true;
            set1.insert(wordList[i]);
        }
        if(flag==false) return 0;
        int cnt=0;
        queue<string> q;
        q.push(beginWord);
        while(!q.empty()){
            cnt++;
            int sz=q.size();
            for(int i=0;i<sz;++i){
                string s=q.front();
                q.pop();
                for(int j=0;j<s.size();++j){
                    string tmp=s;
                    for(char c='a';c<='z';++c){
                        tmp[j]=c;
                        if(tmp.compare(s)==0)continue;

                        if(tmp.compare(endWord)==0)
                            return cnt+1;
                        if(set1.find(tmp)!=set1.end()){
                            q.push(tmp);
                            set1.erase(tmp);

                        }
                    }
                }   
        
            }
            
        }
        return 0;
    }


class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        wordList=set(wordList)
        q=collections.deque([[beginWord,1]])
        while q:
            word,length=q.popleft()
            if word==endWord:
                return length
            for i in range(len(word)):
                for c in "abcdefghijklmnopqrstuvwxyz":
                    next_word=word[:i]+c+word[i+1:]
                    if next_word in wordList:
                        wordList.remove(next_word)
                        q.append([next_word,length+1])
        return 0
