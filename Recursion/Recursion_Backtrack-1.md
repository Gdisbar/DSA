## Part A — Recursion

| Pattern | Recognition | Core Idea |
|---|---|---|
| Linear Recursion | Factorial, Reverse, Power, Sum | One recursive call per step | 
| Binary Recursion | Tree, Fibonacci, Merge, Split | Each call creates multiple children — watch for repeated work → Memoization |
| Divide & Conquer | Split, Merge, Sort, Half | Solve left, solve right, merge |
| Tree DFS | Tree, Path, Height, Traversal | Each node is a smaller version of the same problem | 94, 104, 112, 124, 543 |
| Recursive Simulation | Parser, Expression, Nested | Recursion mirrors the nested structure directly | 394, 385 |

## Pattern 1 - Linear Recursion

### Sort a Stack

```

Stack visual:
| 2 |  <- TOP
| 1 |
| 3 |  <- BOTTOM

sort_stack([3, 1, 2])
│
├── Pop 2
├── sort_stack([3, 1])
│   │
│   ├── Pop 1
│   ├── sort_stack([3])
│   │   │
│   │   ├── Pop 3
│   │   ├── sort_stack([])  --> BASE CASE REACHED (Returns)
│   │   │
│   │   └── Call insert_in_order(s, 3)  [Stack is now: []]
│   │
│   └── Call insert_in_order(s, 1)      [Stack is now: [3]]
│
└── Call insert_in_order(s, 2)          [Stack is now: [1, 3]]

// sub-tree
insert_in_order([1, 3], 2)
│
├── s[-1] (which is 3) > 2 -> pop 3, hold temp = 3
├── insert_in_order([1], 2)
│   └── s[-1] (which is 1) <= 2 -> BASE CASE REACHED!
│       s.append(2)
│       Stack becomes: [1, 2]
│
└── Push back temp = 3
    Stack becomes: [1, 2, 3]  (3 is top)

```
```java

//TC: O(N*N) - Popping all elements takes $n$ recursive steps. For each element popped, inserting it at the bottom takes 1 + 2 + 3 + ... + n operations in total, giving n*(n+1)/2 = n^2.
//SC: O(N) recursive.

import java.util.Stack;
// Inserts 'elem' into 's' such that 's' remains sorted 
// (ascending: top is largest)
public static void insertInStackByOrder(Stack<Integer> s,int elem){
    // Base case: stack is empty OR top of stack is <= elem
    // In python : s.peek() => s[-1], s.push() => s.append()
    if(s.isEmpty() || s.peek() <= elem){
        s.push(elem);
        return;
    }
    // Pop the top element and recurse
    int temp = s.pop();
    insertInStackByOrder(s,elem);
    // Push back the popped element
    s.push(temp);
    
}
// Helper method to empty the stack recursively and insert back in sorted order
public static void helper(Stack<Integer> s){
    if(s.isEmpty()){
        return;
    }
    int elem = s.pop();
    helper(s);
    // Place popped element at its correct sorted position
    insertInStackByOrder(s,elem);
}
static void reverse(Stack<Integer> s){
    helper(s);
}

//TC : still O(N^2) - Even if the stack is already sorted, popping 
// all elements takes n steps and placing each element 
// back takes - n*(n+1)/2 steps
// SC : O(N)
// Removing this condition makes it to reversing stack - s.peek() <= elem

```

### Reverse Linked List

```python
def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
    if not head or not head.next: return head
    last = self.reverseList(head.next)
    head.next.next = head
    head.next = None
    return last

```

### Calculate pow(x,n)

```java
//TC: log n — The exponent n is halved at each recursive step.
//SC: log n — Call stack depth for the recursive calls.

class Solution {
    public double myPow(double x, int n) {
        long N = n;
        if (N < 0) {
            x = 1 / x;
            N = -N;
        }
        return fastPow(x, N);
    }

    private double fastPow(double x, long n) {
        // Base case: x^0 = 1
        if (n == 0) {
            return 1.0;
        }

        // Recursively compute x^(n / 2)
        double half = fastPow(x, n / 2);

        // If n is even: (x^(n/2))^2
        if (n % 2 == 0) {
            return half * half;
        } 
        // If n is odd: x * (x^(n/2))^2
        else {
            return x * half * half;
        }
    }
}
```

## Pattern-2 : Binary Recursion

### 95. Unique Binary Search Trees II

Given an integer n, return all the structurally unique BST's (binary search trees), which has exactly n nodes of unique values from 1 to n. Return the answer in any order.

**Example 1**:

    Input: n = 3
    Output: [[1,null,2,null,3],[1,null,3,2],[2,1,3],[3,1,null,null,2],[3,2,null,1]]
      1        1         2          3         3
       \        \      /   \       /         /
        2        3    1     3     1         2
         \      /                  \       /
          3    2                    2     1
        
**Example 2**:
    Input: n = 1
    Output: [[1]]

**Constraints**: 1 <= n <= 8 , tells us we can use recursion

**For only counting number of BST**

```
int dp[20]{};
int numTrees(int n) {
    if(n <= 1) return 1;
    if(dp[n]) return dp[n];
    for(int i = 1; i <= n; i++) 
        dp[n] += numTrees(i-1) * numTrees(n-i);
    return dp[n];
}

f(5)  [Solve for n=5]
│
├── Root = 1  ─── f(0) * f(4)
│   │
│   └── f(4)  [Solve for n=4]
│       │
│       ├── Root = 1  ─── f(0) * f(3)
│       │   │
│       │   └── f(3)  [Solve for n=3]
│       │       │
│       │       ├── Root = 1  ─── f(0) * f(2)
│       │       │   │
│       │       │   └── f(2)  [Solve for n=2]
│       │       │       │
│       │       │       ├── Root = 1 ─── f(0) * f(1)  --> [BASE CASE: 1 * 1 = 1]
│       │       │       └── Root = 2 ─── f(1) * f(0)  --> [BASE CASE: 1 * 1 = 1]
│       │       │       └── Result: f(2) = 1 + 1 = 2
│       │       │
│       │       ├── Root = 2 ─── f(1) * f(1)  --> [BASE CASE: 1 * 1 = 1]
│       │       └── Root = 3 ─── f(2) * f(0)  --> [MEMO LOOKUP: f(2)=2, f(0)=1 -> 2 * 1 = 2]
│       │       └── Result: f(3) = 2 + 1 + 2 = 5
│       │
│       ├── Root = 2 ─── f(1) * f(2)  --> [MEMO LOOKUP: f(1)=1, f(2)=2 -> 1 * 2 = 2]
│       ├── Root = 3 ─── f(2) * f(1)  --> [MEMO LOOKUP: f(2)=2, f(1)=1 -> 2 * 1 = 2]
│       └── Root = 4 ─── f(3) * f(0)  --> [MEMO LOOKUP: f(3)=5, f(0)=1 -> 5 * 1 = 5]
│       └── Result: f(4) = 5 + 2 + 2 + 5 = 14
│
├── Root = 2 ─── f(1) * f(3)  --> [MEMO LOOKUP: f(1)=1, f(3)=5 -> 1 * 5 = 5]
├── Root = 3 ─── f(2) * f(2)  --> [MEMO LOOKUP: f(2)=2, f(2)=2 -> 2 * 2 = 4]
├── Root = 4 ─── f(3) * f(1)  --> [MEMO LOOKUP: f(3)=5, f(1)=1 -> 5 * 1 = 5]
└── Root = 5 ─── f(4) * f(0)  --> [MEMO LOOKUP: f(4)=14, f(0)=1 -> 14 * 1 = 14]
└── Final Result: f(5) = 14 + 5 + 4 + 5 + 14 = 42

```
**Storing the BST(s)**
```code
buildTree(1, 3)
│
├── Root i = 1
│   ├── Left:  buildTree(1, 0) ──> [ null ]
│   └── Right: buildTree(2, 3)
│       │   ├── Root i = 2
│       │   │   ├── Left:  buildTree(2, 1) ──> [ null ]
│       │   │   └── Right: buildTree(3, 3) ──> [ (3) ]
│       │   │   └── Result: [ 2->(R:3) ]
│       │   │
│       │   └── Root i = 3
│       │       ├── Left:  buildTree(2, 2) ──> [ (2) ]
│       │       └── Right: buildTree(4, 3) ──> [ null ]
│       │       └── Result: [ 3->(L:2) ]
│       │
│       └── Returned List: [ 2->(R:3) , 3->(L:2) ]
│
│   └── Combined Trees for Root 1:
│       • 1 -> (R: 2->(R:3))  [Tree 1]
│       • 1 -> (R: 3->(L:2))  [Tree 2]
│
├── Root i = 2
│   ├── Left:  buildTree(1, 1) ──> [ (1) ]
│   └── Right: buildTree(3, 3) ──> [ (3) ]
│   └── Combined Trees for Root 2:
│       • 2 -> (L:1, R:3)     [Tree 3]
│
└── Root i = 3
    ├── Left:  buildTree(1, 2)
    │   │   ├── Root i = 1
    │   │   │   ├── Left:  buildTree(1, 0) ──> [ null ]
    │   │   │   └── Right: buildTree(2, 2) ──> [ (2) ]
    │   │   │   └── Result: [ 1->(R:2) ]
    │   │   │
    │   │   └── Root i = 2
    │   │       ├── Left:  buildTree(1, 1) ──> [ (1) ]
    │   │       └── Right: buildTree(2, 1) ──> [ null ]
    │   │       └── Result: [ 2->(L:1) ]
    │   │
    │   └── Returned List: [ 1->(R:2) , 2->(L:1) ]
    │
    └── Right: buildTree(4, 3) ──> [ null ]
    │
    └── Combined Trees for Root 3:
        • 3 -> (L: 1->(R:2))  [Tree 4]
        • 3 -> (L: 2->(L:1))  [Tree 5]
```
```java
/*
TC: O(C_n) or O(4^n/n^(3/2)), where C_n is the n-th Catalan number representing the total number of unique BSTs generated.

SC: O(C_n) to store all unique tree structures in memory, plus O(n) recursion call stack depth.
*/
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public List<TreeNode> buildTree(int start,int end){
        List<TreeNode> ans = new ArrayList<>();

        // Base case : start > end, add NULL to list & return
        if (start > end){
            ans.add(null);
            return ans;
        }
        // Iterate through all values from start to end to construct left and right subtree recursively
        for (int i = start;i <= end;i++){
            List<TreeNode> leftSubTree = this.buildTree(start,i-1);
            List<TreeNode> rightSubTree = this.buildTree(i+1,end);
            // loop through all left and right subtrees and connect them to i-th root  
            for (int j = 0;j < leftSubTree.size();j++){
                for (int k = 0;k < rightSubTree.size();k++){
                    TreeNode root = new TreeNode(i,null,null);
                    root.left = leftSubTree.get(j); // Connect left subtree rooted at leftSubTree[j]
                    root.right = rightSubTree.get(k);
                     // Add this tree(rooted at i)
                     ans.add(root);
                }
            }
           
        }
        return ans;
    }
    public List<TreeNode> generateTrees(int n) {
        if (n == 0) {
            return new ArrayList<>();
        }
        List<TreeNode> ans = this.buildTree(1,n);
        return ans;
    }
}
```

## Pattern-3 : Divide & Conquer 

### 148. Sort List

Given the head of a linked list, return the list after sorting it in ascending order.

**Example 1**:

    Input: head = [4,2,1,3]
    Output: [1,2,3,4]

```java

/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode sortList(ListNode head) {
        // Base case: If list contains 0 or 1 node
        if (head == null || head.next == null) {
            return head;
        }

        ListNode temp = null;
        ListNode slow = head;
        ListNode fast = head;

        // Two-pointer approach / Turtle-Hare Algorithm to find the middle
        while (fast != null && fast.next != null) {
            temp = slow;
            slow = slow.next;        // slow moves by 1
            fast = fast.next.next;   // fast moves by 2
        }

        temp.next = null;            // End of first (left) half

        ListNode l1 = sortList(head); // Recursive call for left half
        ListNode l2 = sortList(slow); // Recursive call for right half

        return mergeList(l1, l2);     // Merge sorted halves
    }

    // Merge function
    private ListNode mergeList(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(0);
        ListNode curr = dummy;

        while (l1 != null && l2 != null) {
            if (l1.val <= l2.val) {
                curr.next = l1;
                l1 = l1.next;
            } else {
                curr.next = l2;
                l2 = l2.next;
            }
            curr = curr.next;
        }

        // Attach the remaining nodes (in Java, attaching the remaining sublist head is sufficient)
        if (l1 != null) {
            curr.next = l1;
        }

        if (l2 != null) {
            curr.next = l2;
        }

        return dummy.next;
    }
}
```

### 315. Count of Smaller Numbers After Self

Given an integer array nums, return an integer array counts where counts[i] is the number of smaller elements to the right of nums[i].

**Example 1**:

    Input: nums = [5,2,6,1]
    Output: [2,1,1,0]
    Explanation:
    To the right of 5 there are 2 smaller elements (2 and 1).
    To the right of 2 there is only 1 smaller element (1).
    To the right of 6 there is 1 smaller element (1).
    To the right of 1 there is 0 smaller element.

**Example 2**:

    Input: nums = [-1,-1]
    Output: [0,0]

Merge Sort naturally splits the array into a Left Half and a Right Half. When merging two sorted halves:

- Every element in the Right Half was originally to the right of every element in the Left Half.

- If we pick an element from the Right Half (``nums[right]``) because it is smaller than ``nums[left]``, that right element is smaller than ``nums[left]`` and all remaining unmerged elements in the Left Half.

- We track this with a counter: nRightLessThanLeft.

- Whenever an element from the Left Half gets placed into the sorted temp array, we add nRightLessThanLeft to its original index in indices[].

```code
Let's trace the input array: nums = [5, 2, 6, 1]

Initial state with original indices {val, original_idx}:
newNums = [ (5,0), (2,1), (6,2), (1,3) ]

mergeSort(0, 3)  [nums: (5,0), (2,1), (6,2), (1,3)]
│
├── mergeSort(0, 1)  [Left Half: (5,0), (2,1)]
│   │
│   ├── mergeSort(0, 0)  --> Base Case: Returns (5,0)
│   ├── mergeSort(1, 1)  --> Base Case: Returns (2,1)
│   │
│   └── MERGE(0, 0, 1)  [Left: (5,0) | Right: (2,1)]
│       │
│       ├── Compare 5 > 2:
│       │   ├── Pick (2,1) from Right -> nRightLessThanLeft = 1
│       │   └── Temp: [(2,1)]
│       │
│       ├── Left half remaining: (5,0)
│       │   ├── Pick (5,0) from Left -> indices[0] += 1  (indices[0] becomes 1)
│       │   └── Temp: [(2,1), (5,0)]
│       │
│       └── Sorted Subarray: [(2,1), (5,0)]
│
├── mergeSort(2, 3)  [Right Half: (6,2), (1,3)]
│   │
│   ├── mergeSort(2, 2)  --> Base Case: Returns (6,2)
│   ├── mergeSort(3, 3)  --> Base Case: Returns (1,3)
│   │
│   └── MERGE(2, 2, 3)  [Left: (6,2) | Right: (1,3)]
│       │
│       ├── Compare 6 > 1:
│       │   ├── Pick (1,3) from Right -> nRightLessThanLeft = 1
│       │   └── Temp: [(1,3)]
│       │
│       ├── Left half remaining: (6,2)
│       │   ├── Pick (6,2) from Left -> indices[2] += 1  (indices[2] becomes 1)
│       │   └── Temp: [(1,3), (6,2)]
│       │
│       └── Sorted Subarray: [(1,3), (6,2)]
│
└── FINAL MERGE(0, 1, 3)  [Left: (2,1), (5,0) | Right: (1,3), (6,2)]
    │
    ├── Compare (2,1) vs (1,3) -> 2 > 1:
    │   ├── Pick (1,3) from Right -> nRightLessThanLeft = 1
    │   └── Temp: [(1,3)]
    │
    ├── Compare (2,1) vs (6,2) -> 2 <= 6:
    │   ├── Pick (2,1) from Left -> indices[1] += 1  (indices[1] becomes 1)
    │   └── Temp: [(1,3), (2,1)]
    │
    ├── Compare (5,0) vs (6,2) -> 5 <= 6:
    │   ├── Pick (5,0) from Left -> indices[0] += 1  (indices[0] becomes 1 + 1 = 2)
    │   └── Temp: [(1,3), (2,1), (5,0)]
    │
    ├── Left half empty. Remaining Right element: (6,2)
    │   └── Temp: [(1,3), (2,1), (5,0), (6,2)]
    │
    └── FINAL RESULT IN INDICES:
        - indices[0] (for 5) = 2  (smaller on right: 2, 1)
        - indices[1] (for 2) = 1  (smaller on right: 1)
        - indices[2] (for 6) = 1  (smaller on right: 1)
        - indices[3] (for 1) = 0  (smaller on right: none)
        
        Output: [2, 1, 1, 0]

```

```java
import java.util.*;

// Helper class to store element value along with its original index
private static class Pair {
    int val;
    int idx;

    Pair(int val, int idx) {
        this.val = val;
        this.idx = idx;
    }
}

private void mergeSort(int start, int end, 
                        Pair[] nums, int[] indices, 
                        Pair[] temp) {
    if (start >= end) {
        return;
    }

    int mid = start + (end - start) / 2;

    mergeSort(start, mid, nums, indices, temp);
    mergeSort(mid + 1, end, nums, indices, temp);

    int left = start;
    int right = mid + 1;
    int idx = start;
    int nRightLessThanLeft = 0;

    // Merge two sorted halves
    while (left <= mid && right <= end) {
        // Using <= ensures stability for equal elements
        if (nums[left].val <= nums[right].val) {
            indices[nums[left].idx] += nRightLessThanLeft;
            temp[idx++] = nums[left++];
        } else {
            temp[idx++] = nums[right++];
            nRightLessThanLeft++;
        }
    }

    // Process remaining elements on the left side
    while (left <= mid) {
        indices[nums[left].idx] += nRightLessThanLeft;
        temp[idx++] = nums[left++];
    }

    // Process remaining elements on the right side
    while (right <= end) {
        temp[idx++] = nums[right++];
    }

    // Copy merged elements back to original array
    for (int i = start; i <= end; i++) {
        nums[i] = temp[i];
    }
}

public List<Integer> countSmaller(int[] nums) {
    int n = nums.length;
    if (n == 0) return new ArrayList<>();

    Pair[] newNums = new Pair[n];
    Pair[] temp = new Pair[n];
    int[] indices = new int[n];

    for (int i = 0; i < n; i++) {
        newNums[i] = new Pair(nums[i], i);
    }

    mergeSort(0, n - 1, newNums, indices, temp);

    // Convert int[] result to List<Integer>
    List<Integer> result = new ArrayList<>(n);
    for (int count : indices) {
        result.add(count);
    }

    return result;
}

```

## Pattern-4 : Tree DFS

## Pattern-5 : Recursive Simulation

