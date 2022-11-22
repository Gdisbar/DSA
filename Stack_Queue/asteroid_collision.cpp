// lesser element in right --> pop until s.top()<-x
// problem specifice condition 
// push --> 
// pop  --> (s.top()==-x)+opposite sign+Base case (!s.empty())
// do nothing --> lesser value (s.top()>-x) + Base case


735. Asteroid Collision
==========================
// We are given an array asteroids of integers representing asteroids in a row.

// For each asteroid, the absolute value represents its size, and the sign represents 
// its direction (positive meaning right, negative meaning left). Each asteroid moves 
// at the same speed.

// Find out the state of the asteroids after all collisions. If two asteroids meet, 
// the smaller one will explode. If both are the same size, both will explode. 
// Two asteroids moving in the same direction will never meet.

 

// Example 1:

// Input: asteroids = [5,10,-5]
// Output: [5,10]
// Explanation: The 10 and -5 collide resulting in 10. The 5 and 10 never collide.

// Example 2:

// Input: asteroids = [8,-8]
// Output: []
// Explanation: The 8 and -8 collide exploding each other.

// Example 3:

// Input: asteroids = [10,2,-5]
// Output: [10]
// Explanation: The 2 and -5 collide resulting in -5. The 10 and -5 collide resulting 
// in 10.

//TC : n , SC : n , each element inserted & popped atmost once 2*n operatios
//97% faster, 65% less memory

// if a[i] > 0 we push in stack
// else we pop until a[i]>s.top() & different sign 
// after that we check few individual conditions
// equal value + different sign we pop
// lesser a[i] value + different sign,it'll be destroyed no need to do anything 
// same or higher a[i] value + same sign push 

vector<int> asteroidCollision(vector<int>& a) {
        stack<int> s;
        for(int x : a){
            if(x>0) s.push(x);
            else{
                while(!s.empty()&&s.top()>0&&s.top()<-x){ //smaller+opposite
                    s.pop();
                }
                if(!s.empty()&&s.top()>0&&s.top()==-x){ //equal+opposite
                    s.pop();
                }
                else if(!s.empty()&&s.top()>-x){/*continue to next a[i]*/}
                else{ //same direction or no element in stack,push part of while
                    s.push(x); 
                }
            }
        }
        vector<int> ans(s.size()); //start filling from end
        for(int i=s.size()-1;i>=0&&!s.empty();--i){
            ans[i]=s.top();
            s.pop();
        }
        
        return ans;
    }