// Converting 2 stack -> 1 stack solution
2-stack solution s2 store min element at each step & s1 store normal element,
when we call top() we mean s1.top()

s2.top() // output of getMin() , min element
if(s2.empty()||val<=getMin()) s2.push(val) //during push operation
if(s1.top()==getMin()) s2.pop(); // during pop operation


for 1-stack we can''t store min element at each stage so we modify our value
in hasing like way so we can get min/element whichever is required

if(s.empty()) s.push(0L),mn=val
else s.push(val-mn),mn=min(mn,val)

x = s.top() ,s.pop();
mn=abs(x-mn) //if(x<0)

max(s.top()+mn,mn) //s.top()

155. Min Stack
===================

Input
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]

Output
[null,null,null,null,-3,null,0,-2]

Explanation
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin(); // return -3
minStack.pop();
minStack.top();    // return 0
minStack.getMin(); // return -2

//Using 2 stack

class MinStack {
private:
    stack<int> s1;
    stack<int> s2;    
public:

    void push(int val) {
        s1.push(val);
        if(s2.empty()||val<=getMin()) s2.push(val);
    }
    
    void pop() {
        if(s1.top()==getMin()) s2.pop();
        s1.pop();
    }
    
    int top() {
        return s1.top();
    }
    
    int getMin() {
        return s2.top();
    }
};

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack* obj = new MinStack();
 * obj->push(val);
 * obj->pop();
 * int param_3 = obj->top();
 * int param_4 = obj->getMin();
 */

//Using 1 stack
// almost 2x slower,memory use should be half but doesn''t show so

class MinStack {
private:
    stack<long> s;
    long mn;
public:
    void push(int val) {
        if(s.empty()){
            s.push(0L);
            mn=val;
        }
        else{
            s.push(val-mn);
            if(val<mn) mn=val;
        }
    }
    
    void pop() {
        if(s.empty()) return;
        long x = s.top();
        s.pop();
        if(x<0) mn=mn-x;
    }
    
    int top() {
        return (int)max(s.top()+mn,mn);
    }
    
    int getMin() {
        return (int)mn;
    }
};
