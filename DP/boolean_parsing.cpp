1106. Parsing A Boolean Expression
====================================
// Return the result of evaluating a given boolean expression, 
// represented as a string.

// An expression can either be:

//     "t", evaluating to True;
//     "f", evaluating to False;
//     "!(expr)", evaluating to the logical NOT of the inner expression expr;
//     "&(expr1,expr2,...)", evaluating to the logical AND of 2 or more inner 
//     expressions expr1, expr2, ...;
//     "|(expr1,expr2,...)", evaluating to the logical OR of 2 or more inner 
//     expressions expr1, expr2, ...

 

// Example 1:

// Input: expression = "!(f)"
// Output: true

// Example 2:

// Input: expression = "|(f,t)"
// Output: true

// Example 3:

// Input: expression = "&(t,f)"
// Output: false

// Intuition

// Well, we can see that &, | and ! are just three functions.
// And in python, they are function all, any and keyword not.

// Explanation

// Following the description,
// it demands us to evaluate the expression.
// So no recursion and no stack, I just eval the expression.

// Complexity

// Time O(N)
// Space O(N)
// I guess it's not fast compared with string parse, but wow it's O(N).

// Python:

    def parseBoolExpr(self, S, t=True, f=False):
        return eval(S.replace('!', 'not |').replace('&(', 'all([').replace('|(', 'any([').replace(')', '])'))


def parseBoolExpr(self, expression: str) -> bool:
        
        def parse(e: str, lo: int, hi: int) -> bool:
            if hi - lo == 1: # base case
                return e[lo] == 't'               
            ans = e[lo] == '&' # only when the first char is '&', ans assigned True.
            level, start = 0, lo + 2 # e[lo + 1] must be '(', so start from lo + 2 to delimit sub-expression.
            for i in range(lo + 2, hi):
                if level == 0  and e[i] in [',', ')']: # found a sub-expression.
                    cur = parse(e, start, i) # recurse to sub-problem.
                    start = i + 1 # start point of next sub-expression.
                    if e[lo] == '&':
                        ans &= cur
                    elif e[lo] == '|':
                        ans |= cur
                    else: # e[lo] is '!'.
                        ans = not cur
                if e[i] == '(':
                    level = level + 1
                elif e[i] == ')':
                    level = level - 1
            return ans;        
        
        return parse(expression, 0, len(expression))

//

        stack = []
        for i in range(len(e)):
            c = e[i]
            if c == ',':
                continue
            elif c == 't':
                stack.append(True)
            elif c == 'f':
                stack.append(False)
            elif c in ['&', '|', '!','(']:
                stack.append(c)
            elif c == ')':
                operands = []
                while stack[-1] != '(':
                    operands.append(stack.pop())
                stack.pop() # pop out '('.
                operator = stack.pop()
                if operator == '&':
                    stack.append(all(operands))
                elif operator == '|':
                    stack.append(any(operands))
                elif operator == '!':
                    stack.append(not operands[0])
        return stack.pop()

Boolean Parenthesization Problem //Extra
============================================

// Given a boolean expression with the following symbols. 

// Symbols
//     'T' ---> true 
//     'F' ---> false 

// And following operators filled between symbols 

// Operators
//     &   ---> boolean AND
//     |   ---> boolean OR
//     ^   ---> boolean XOR 

// Count the number of ways we can parenthesize the expression so that the 
// value of expression evaluates to true. 
// Let the input be in form of two arrays one contains the symbols (T and F) 
// in order and the other contains operators (&, | and ^}

// Examples: 

// Input: symbol[]    = {T, F, T}
//        operator[]  = {^, &}
// Output: 2
// The given expression is "T ^ F & T", it evaluates true
// in two ways "((T ^ F) & T)" and "(T ^ (F & T))"

// Input: symbol[]    = {T, F, F}
//        operator[]  = {^, |}
// Output: 2
// The given expression is "T ^ F | F", it evaluates true
// in two ways "( (T ^ F) | F )" and "( T ^ (F | F) )". 

// Input: symbol[]    = {T, T, F, T}
//        operator[]  = {|, &, ^}
// Output: 4
// The given expression is "T | T & F ^ T", it evaluates true
// in 4 ways ((T|T)&(F^T)), (T|(T&(F^T))), (((T|T)&F)^T) 
// and (T|((T&F)^T)). 

int dp[101][101][2];
int parenthesis_count(string s,int i,int j,int isTrue){
    // Base Condition
    if (i > j)
        return false;
    if (i == j) {
        if (isTrue == 1)
            return s[i] == 'T';
        else
            return s[i] == 'F';
    }
 
    if (dp[i][j][isTrue] != -1)
        return dp[i][j][isTrue];
    int ans = 0;
    for (int k = i + 1; k <= j - 1; k = k + 2){
        int leftF, leftT, rightT, rightF;
        if (dp[i][k - 1][1] == -1){
            leftT = parenthesis_count(s, i, k - 1, 1);
        } // Count no. of T in left partition
        else {
            leftT = dp[i][k - 1][1];
        }
 
        if (dp[k + 1][j][1] == -1){
            rightT = parenthesis_count(s, k + 1, j, 1);
        } // Count no. of T in right partition
        else{
            rightT = dp[k + 1][j][1];
        }
 
        if (dp[i][k - 1][0] == -1){
            // Count no. of F in left partition
            leftF = parenthesis_count(s, i, k - 1, 0);
        }
        else{
            leftF = dp[i][k - 1][0];
        }
 
        if (dp[k + 1][j][0] == -1){
            // Count no. of F in right partition
            rightF = parenthesis_count(s, k + 1, j, 0);
        }
        else{
            rightF = dp[k + 1][j][0];
        }
 
        if (s[k] == '&'){
            if (isTrue == 1)
                ans += leftT * rightT;
            else
                ans += leftF * rightF + leftT * rightF
                       + leftF * rightT;
        }
        else if (s[k] == '|'){
            if (isTrue == 1)
                ans += leftT * rightT + leftT * rightF
                       + leftF * rightT;
            else
                ans = ans + leftF * rightF;
        }
        else if (s[k] == '^'){
            if (isTrue == 1)
                ans = ans + leftF * rightT + leftT * rightF;
            else
                ans = ans + leftT * rightT + leftF * rightF;
        }
        dp[i][j][isTrue] = ans;
    }
    return ans;
}
 
// Driver Code
int main()
{
    string symbols = "TTFT";
    string operators = "|&^";
    string s;
    int j = 0;
 
    for (int i = 0; i < symbols.length(); i++)
    {
        s.push_back(symbols[i]);
        if (j < operators.length())
            s.push_back(operators[j++]);
    }
     
    // We obtain the string  T|T&F^T
    int n = s.length();
     
    // There are 4 ways
    // ((T|T)&(F^T)), (T|(T&(F^T))), (((T|T)&F)^T) and
    // (T|((T&F)^T))
    memset(dp, -1, sizeof(dp));
    cout << parenthesis_count(s, 0, n - 1, 1);
    return 0;
}