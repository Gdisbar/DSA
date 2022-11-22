921. Minimum Add to Make Parentheses Valid
==============================================
// Intuition:

// To make a string valid,
// we can add some ( on the left,
// and add some ) on the right.
// We need to find the number of each.

// Explanation:

// left records the number of ( we need to add on the left of S.
// right records the number of ) we need to add on the right of S,
// which equals to the number of current opened parentheses.



// Time Complexity:

// Time O(N)
// Space O(1)
// left --> not_opened , right -> not_closed
int minAddToMakeValid(string s) {
       int left = 0, right = 0;
        for (char c : s)
            if (c == '(')
                right++;
            else if (right > 0) //close which were not_closed
                right--;
            else             //we open when resultant not_closed > 0
                left++;
        return left + right;
    }

// Using stack --> Intutive
    // if encounter '(', push '(' into stack;
    // otherwise, ')', check if there is a matching '(' on top of stack; if no, 
    // increase the counter by 1; if yes, pop it out;
    // after the loop, count in the un-matched characters remaining in the stack.

int minAddToMakeValid(string s) {
        int count = 0;
        stack<char> st;
        for (char c : s) {
            if (c == '(') { 
                st.push(c); 
            }else if (st.empty()) {  //we got ')' & not matching '(' found in stack
                ++count;  
            }else {       //matching ')' found
                st.pop(); 
            }
        }
        return count + stk.size(); //count unmatched
    }





1249. Minimum Remove to Make Valid Parentheses
===============================================
// Given a string s of '(' , ')' and lowercase English characters.

// Your task is to remove the minimum number of parentheses ( '(' or ')', 
// in any positions ) so that the resulting parentheses string is valid and 
// return any valid string.

// Formally, a parentheses string is valid if and only if:

//     It is the empty string, contains only lowercase characters, or
//     It can be written as AB (A concatenated with B), where A and B are valid 
//     strings, or
//     It can be written as (A), where A is a valid string.

 

// Example 1:

// Input: s = "lee(t(c)o)de)"
// Output: "lee(t(c)o)de"
// Explanation: "lee(t(co)de)" , "lee(t(c)ode)" would also be accepted.

// Example 2:

// Input: s = "a)b(c)d"
// Output: "ab(c)d"

// Example 3:

// Input: s = "))(("
// Output: ""
// Explanation: An empty string is also valid.


class Solution {
public:
    string minRemoveToMakeValid(string s) {
        string ans = "";
        stack<int> st;
        int n = s.size();
        int i=0;
        while(i < n) {
            if(s[i] == '(') {
                st.push(i);
            } else if(s[i] == ')') {
                if(st.empty()) {
                    s[i] = '#';
                } else {
                    st.pop();
                }
            }
            i++;
        }
        while(!st.empty()) {
            s[st.top()] = '#';
            st.pop();
        }
        i = 0;
        while(i<n) {
            if(s[i] != '#') {
                ans += s[i];
            }
            i++;
        }
        return ans;
    }
};

Minimum number of bracket reversals needed to make an expression balanced
=============================================================================
// Given an expression with only ‘}’ and ‘{‘. The expression may not be balanced. 
// Find minimum number of bracket reversals to make the expression balanced.


// Examples: 

// Input:  exp = "}{"
// Output: 2
// We need to change '}' to '{' and '{' to
// '}' so that the expression becomes balanced, 
// the balanced expression is '{}'

// Input:  exp = "{{{"
// Output: Can''t be made balanced using reversals

// Input:  exp = "{{{{"
// Output: 2 

// Input:  exp = "{{{{}}"
// Output: 1 

// Input:  exp = "}{{}}{{{"
// Output: 3

int countMinReversals(string s){
    if(s.size()&1) return s.size();
        stack<char> st;
        int res=0;
        for(int i=0;i<s.size();++i){
            if(s[i]=='(') st.push(s[i]);
            else{
                if(!st.empty()){
                    st.pop();
                }
                else{
                    st.push('('); //we've found ')'
                    res++; //}|{{}}|{{{ ,res=1(i=0),st.size()=4
                }   
            }
        }
        if (st.size() % 2 != 0) return -1; //odd
        res+=st.size()/2; // res=res(=1)+4/2
        return res;
}


//SC : 1

int countMinReversals(string expr)
{
    int len = expr.length();
  
    // Expressions of odd lengths
    // cannot be balanced
    if (len % 2 != 0) {
        return -1;
    }
    int left_brace = 0, right_brace = 0;
    int ans;
    for (int i = 0; i < len; i++) {
  
        // If we find a left bracket then we simply
        // increment the left bracket
        if (expr[i] == '{') {
            left_brace++;
        }
  
        // Else if left bracket is 0 then we find
        // unbalanced right bracket and increment
        // right bracket or if the expression
        // is balanced then we decrement left
        else {
            if (left_brace == 0) {
                right_brace++;
            }
            else {
                left_brace--;
            }
        }
    }
    ans = ceil(left_brace / 2.0) + ceil(right_brace / 2.0);
    return ans;
}
  