Find if an expression has duplicate parenthesis or not
===========================================================
// Given a balanced expression, find if it contains duplicate parenthesis or not. 
// A set of parenthesis are duplicate if the same subexpression is surrounded by 
// multiple parenthesis. 

// Examples: 

// Below expressions have duplicate parenthesis -      
// ((a+b)+((c+d)))
// The subexpression "c+d" is surrounded by two
// pairs of brackets.

// (((a+(b)))+(c+d))
// The subexpression "a+(b)" is surrounded by two 
// pairs of brackets.

// (((a+(b))+c+d))
// The whole expression is surrounded by two 
// pairs of brackets.

// ((a+(b))+(c+d))
// (b) and ((a+(b)) is surrounded by two
// pairs of brackets.

// Below expressions don''t have any duplicate parenthesis -
// ((a+b)+(c+d)) 
// No subsexpression is surrounded by duplicate
// brackets.

// It may be assumed that the given expression is valid and there are not 
// any white spaces present. 


// (((a+(b))+c+d))
bool findDuplicateparenthesis(string s){
    stack<char> st;
    for(char c : s){
        if(c==')'){
            if(st.top()=='(') return true; // there is nothing between '(' and ')'
            else{
                while(st.top()!='('){
                    st.pop(); // all operator & operand in between '(' and ')'
                }
                st.pop(); // we need to pop the ')' inside stack
            }
            
        }else{
            st.push(c); // anything except ')' , push into stack
        }
    }
    return false; // no duplicate present 
}