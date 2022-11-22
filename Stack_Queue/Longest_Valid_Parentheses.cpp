// generally parentheses validation problem :
// 1. got opening parentheses push
// 2. got closing preantheses process according to condition 
//     //pop() involved Base case checking must
//     2.1.1 pop & match 
//     2.1.2 find length/span ---> res=max(res,i-s.top()) //forward
//     2.2. there is alphaneumeric characters or stack empty push them 

32. Longest Valid Parentheses
===============================
// Given a string containing just the characters '(' and ')', 
// find the length of the longest valid (well-formed) parentheses substring.

 

// Example 1:

// Input: s = "(()"
// Output: 2
// Explanation: The longest valid parentheses substring is "()".

// Example 2:

// Input: s = ")()())"
// Output: 4
// Explanation: The longest valid parentheses substring is "()()".

// Example 3:

// Input: s = ""
// Output: 0


int longestValidParentheses(string s) {
        stack<int> stk;
        int res=0,i=0;
        stk.push(-1);
        for(int i=0;i<s.size();++i){
            if(s[i]=='(') stk.push(i);
            else{
                if(!stk.empty()) { // Pop the previous opening bracket's index
                    stk.pop();
                }
                // Check if this length formed with base of current valid substring 
                if(!stk.empty()) res=max(res,i-stk.top());
                else stk.push(i); // If stack is empty. push current index as
            					 // base for next valid substring (if any)
            }
        }
        return res;
    }

// well there is this alternate vector version of stack

int longestValidParentheses(string s) {
        if (s.size()<=1) return 0;
        //length of the longest valid substring ending at that index
        vector<int> a(s.size(),0); 
        int res=0;
        for(int i=1;i<s.size();++i){
            if (s[i]==')'&&i-a[i-1]-1>=0&&s[i-a[i-1]-1]=='('){
                a[i]=a[i-1]+2+((i-a[i-1]-2>=0)?a[i-a[i-1]-2]:0);
                res=max(res,a[i]);
            }
        }
        return res;
    }