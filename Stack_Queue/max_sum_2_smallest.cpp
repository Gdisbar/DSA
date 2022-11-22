Maximum sum of smallest and second smallest in an array
==========================================================
// Given an array, find maximum sum of smallest and second smallest elements 
// chosen from all possible subarrays. More formally, if we write all (nC2) 
// subarrays of array of size >=2 and find the sum of smallest and second smallest, 
// then our answer will be maximum sum among them. 


// Examples: 

// Input : arr[] = [4, 3, 1, 5, 6]
// Output : 11
// Subarrays with smallest and second smallest are,
// [4, 3]        smallest = 3    second smallest = 4
// [4, 3, 1]    smallest = 1    second smallest = 3
// [4, 3, 1, 5]    smallest = 1    second smallest = 3
// [4, 3, 1, 5, 6]    smallest = 1    second smallest = 3
// [3, 1]         smallest = 1    second smallest = 3
// [3, 1, 5]     smallest = 1    second smallest = 3
// [3, 1, 5, 6]    smallest = 1    second smallest = 3
// [1, 5]        smallest = 1    second smallest = 5
// [1, 5, 6]    smallest = 1    second smallest = 5
// [5, 6]         smallest = 5    second smallest = 6
// Maximum sum among all above choices is, 5 + 6 = 11

// Input : arr[] =  {5, 4, 3, 1, 6}
// Output : 9

// An efficient solution is based on the observation that this problem reduces to 
// finding a maximum sum of two consecutive elements in array. 

int pairWithMaxSum(int arr[], int N)
{
   if (N < 2)
     return -1;
 
   // Find two consecutive elements with maximum
   // sum.
   int res = arr[0] + arr[1];
   for (int i=1; i<N-1; i++)
      res = max(res, arr[i] + arr[i+1]);
 
   return res;
}

Reversing the first K elements of a Queue
===========================================
// Given an integer k and a queue of integers, we need to reverse the order 
// of the first k elements of the queue, leaving the other elements in the same 
// relative order.
// Only following standard operations are allowed on queue. 

//     enqueue(x) : Add an item x to rear of queue
//     dequeue() : Remove an item from front of queue
//     size() : Returns number of elements in queue.
//     front() : Finds front item.

// Examples: 

// Input : Q = [10, 20, 30, 40, 50, 60,70, 80, 90, 100] k = 5
// Output : Q = [50, 40, 30, 20, 10, 60, 70, 80, 90, 100]

// Input : Q = [10, 20, 30, 40, 50, 60, 70 , 80, 90, 100] k = 4
// Output : Q = [40, 30, 20, 10, 50, 60,70, 80, 90, 100]

/* Function to reverse the first
   K elements of the Queue */
void reverseQueueFirstKElements(int k, queue<int>& Queue)
{
    if (Queue.empty() || k > Queue.size())
        return;
    if (k <= 0)
        return;
 
    stack<int> Stack;
 
    /* Push the first K elements
       into a Stack*/
    for (int i = 0; i < k; i++) {
        Stack.push(Queue.front());
        Queue.pop();
    }
 
    /* Enqueue the contents of stack
       at the back of the queue*/
    while (!Stack.empty()) {
        Queue.push(Stack.top());
        Stack.pop();
    }
 
    /* Remove the remaining elements and
       enqueue them at the end of the Queue*/
    for (int i = 0; i < Queue.size() - k; i++) {
        Queue.push(Queue.front());
        Queue.pop();
    }
}
 
