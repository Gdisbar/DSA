// Map Based solution
unordered_map freq stores the <value,freq[value]> , we maintain a global variable
maxfreq to get most frequent element in the stack + m[freq[val]]=val
unordered_map<int, stack<int>> m
int x = m[maxfreq].top() , m[maxfreq].pop();
if (!m[freq[x]--].size()) maxfreq--;

//Priority_Queue --> get maxfreq element + break tie

priority_queue<tuple<Freq, Timestamp, val>> pq //max-heap
 Freq : most frequent one (priority-1)
 timestamp :most recent one on the top (priority-2)
 val : timestamps are unique, we won''t use it to compare


pq.push({f, timestamp++, val})
//tie() : unpack the tuple values into separate variables.
//tuple_cat() : concatenates two tuples and returns a new tuple.
tie(f, tstamp, val) = pq.top().pq.pop()

895. Maximum Frequency Stack
=================================
Design a stack-like data structure to push elements to the stack and pop 
the most frequent element from the stack.

Implement the FreqStack class:

    FreqStack() constructs an empty frequency stack.
    void push(int val) pushes an integer val onto the top of the stack.
    int pop() removes and returns the most frequent element in the stack.
        If there is a tie for the most frequent element, the element closest 
        to the stack''s top is removed and returned.

 

Example 1:

Input
["FreqStack", "push", "push", "push", "push", "push", "push", "pop", "pop", "pop", 
   "pop"]
[[], [5], [7], [5], [7], [4], [5], [], [], [], []]
Output
[null, null, null, null, null, null, null, 5, 7, 5, 4]

Explanation
FreqStack freqStack = new FreqStack();
freqStack.push(5); // The stack is [5]
freqStack.push(7); // The stack is [5,7]
freqStack.push(5); // The stack is [5,7,5]
freqStack.push(7); // The stack is [5,7,5,7]
freqStack.push(4); // The stack is [5,7,5,7,4]
freqStack.push(5); // The stack is [5,7,5,7,4,5]
freqStack.pop();   // return 5, as 5 is the most frequent. The stack becomes [5,7,5,7,4].
freqStack.pop();   // return 7, as 5 and 7 is the most frequent, but 7 is closest to the top. The stack becomes [5,7,5,4].
freqStack.pop();   // return 5, as 5 is the most frequent. The stack becomes [5,7,4].
freqStack.pop();   // return 4, as 4, 5 and 7 is the most frequent, but 4 is closest to the top. The stack becomes [5,7].


//map based solution , 80% faster

    unordered_map<int, int> freq;
    unordered_map<int, stack<int>> m;
    int maxfreq = 0;

    void push(int x) {
        maxfreq = max(maxfreq, ++freq[x]); 
        m[freq[x]].push(x);
    }

    int pop() {
        int x = m[maxfreq].top();
        m[maxfreq].pop();
        if (!m[freq[x]--].size()) maxfreq--;
        return x;
    }
    
// Priority Queue Based solution

using Freq = int;
using Timestamp = int;
class FreqStack {
public:
    // max pq
    priority_queue<tuple<Freq, Timestamp, int>> pq;
    unordered_map<int, Freq> map;
    int timestamp;
    FreqStack() {
        timestamp = 0;
    }
    
    void push(int val) {
        int f = ++map[val];
        // Explanation:
        // Ex.        [1, 2, 3, 1, 1]
        // timestamp:  0  1  2  3  4 
        // freq        1  1  1  2  3
        
        // Freq : most frequent one (priority-1)
        // timestamp :most recent one on the top (priority-2)
        // val : timestamps are unique, we won't use it to compare
        pq.push({f, timestamp++, val});
    }
    
    int pop() {
        int f, tidx, val;
        tie(f, tidx, val) = pq.top();
        pq.pop();
        map[val]--;
        return val;
    }
};


