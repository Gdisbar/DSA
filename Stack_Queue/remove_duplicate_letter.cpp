316. Remove Duplicate Letters
==============================
// Given a string s, remove duplicate letters so that every letter appears 
// once and only once. You must make sure your result is the smallest in 
// lexicographical order among all possible results.


// Example 1:

// Input: s = "bcabc"
// Output: "abc"

s = "bcabc"
i =  01234
freq = a-1,b-2,c-2
st = 0 , b-1,b-T
st = 10, c-1 -> c > b = c-F,st=0
st = 0 , a-0 -> b > a = b-F,st=2
st=432,a-T,b-T,c-T

// Example 2:

// Input: s = "cbacdcbc"
// Output: "acdb"


string removeDuplicateLetters(string s) {
        int n = s.size();
        unordered_map<char, int> freq; //if repeated char
        unordered_map<char, bool> visited;
        stack<int> st; //lexicographical checking
        string res = "";

        for(char ch:s)	freq[ch]++;

        for(int i = 0; i < n; i++){
        	//if already present then, we'll just decrement the freq value of that 
        	//char which has come again in the string but what rightly placed already
            freq[s[i]]--;
            if(visited[s[i]])	continue;
            // keep popping elements, if they are present later and 
            // lexicographically greater than present element
            while(!st.empty() &&  s[i] < s[st.top()] && freq[s[st.top()]] > 0){
                visited[s[st.top()]] = false;
                st.pop();
            }
            st.push(i);
            visited[s[i]] = true;
        }

        while(!st.empty()){
            res = s[st.top()] + res;
            st.pop();
        }

        return res;
    }