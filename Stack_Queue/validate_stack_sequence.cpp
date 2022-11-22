946. Validate Stack Sequences
==============================
// Given two integer arrays pushed and popped each with distinct values, 
// return true if this could have been the result of a sequence of push 
// and pop operations on an initially empty stack, or false otherwise.

 
// Example 1:

// Input: pushed = [1,2,3,4,5], popped = [4,5,3,2,1]
// Output: true
// Explanation: We might do the following sequence:
// push(1), push(2), push(3), push(4),
// pop() -> 4,
// push(5),
// pop() -> 5, pop() -> 3, pop() -> 2, pop() -> 1

// Example 2:

// Input: pushed = [1,2,3,4,5], popped = [4,3,5,1,2]
// Output: false
// Explanation: 1 cannot be popped before 2.


bool validateStackSequences(vector<int>& pushed, vector<int>& popped) {
        stack<int> st; // Create a stack
        
        int j = 0; // Intialise one pointer pointing on popped array
        
        for(auto val : pushed){
            st.push(val); // insert the values in stack
            while(st.size() > 0 && st.top() == popped[j]){ // if st.peek() values equal to popped[j];
                st.pop(); // then pop out
                j++; // increment j
            }
        }
        return st.size() == 0; // check if stack is empty return true else false
    }

// space optimized
    
bool validateStackSequences(vector<int>& pushed, vector<int>& popped) {
        int i = 0; // Intialise one pointer pointing on pushed array
        int j = 0; // Intialise one pointer pointing on popped array
        
        for(auto val : pushed){
            pushed[i++] = val; // using pushed as the stack.
            while(i > 0 && pushed[i - 1] == popped[j]){ // pushed[i - 1] values equal to popped[j];
                i--; // decrement i
                j++; // increment j
            }
        }
        return i == 0; // Since pushed is a permutation of popped so at the end we are supposed to be left with an empty stack
    }