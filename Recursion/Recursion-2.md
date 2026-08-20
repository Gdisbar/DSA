## Part A — Recursion

| Pattern | Recognition | Core Idea |
|---|---|---|
| Linear Recursion | Factorial, Reverse, Power, Sum | One recursive call per step | 
| Binary Recursion | Tree, Fibonacci, Merge, Split | Each call creates multiple children — watch for repeated work → Memoization |
| Divide & Conquer | Split, Merge, Sort, Half | Solve left, solve right, merge |
| Tree DFS | Tree, Path, Height, Traversal | Each node is a smaller version of the same problem |
| Recursive Simulation | Parser, Expression, Nested | Recursion mirrors the nested structure directly | 

## Pattern-4 : Tree DFS

```java

/**
 * TC : O(N)
 * SC : O(N) - skew tree, O(logN) - balanced tree
 * 
 * 
 * 
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

```

### 94. Binary Tree Inorder Traversal

Given the root of a binary tree, return the inorder traversal of its nodes' values.

```java

public static void inorderHelper(TreeNode root,List<Integer> traversal){
    if (root==null) return;
    inorderHelper(root.left,traversal);
    traversal.add(root.val);
    inorderHelper(root.right,traversal);
}
public List<Integer> inorderTraversal(TreeNode root) {
    List<Integer> traversal = new ArrayList<>();
    inorderHelper(root,traversal);
    return traversal;
}
```

### 104. Maximum Depth of Binary Tree

Given the root of a binary tree, return its maximum depth.

A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

**Example 1**:

                   3
                  /  \
                9     20
                     /  \
                    15   7

        Input: root = [3,9,20,null,null,15,7]
        Output: 3

```java
public int maxDepth(TreeNode root) {
    if (root==null) return 0;
    // covered in recursive step
    // if (root.left==null && root.right==null) return 1; 
    return 1 + Math.max(this.maxDepth(root.left),this.maxDepth(root.right));
}
```

### 112. Path Sum

Given the root of a binary tree and an integer targetSum, return true if the tree has a root-to-leaf path such that adding up all the values along the path equals targetSum.

A leaf is a node with no children.

**Example 1**:

                  5
                /   \
               /     \
              4       8
             /       / \
            /       /   \
           11      13    4
          /  \            \
         7    2            1
    Input: root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22
    Output: true
    Explanation: The root-to-leaf path [2 -> 11 -> 4-> 5] with the target sum is shown.

```java
public boolean hasPathSum(TreeNode root, int targetSum) {
    if (root==null) return false;
    if (root.left==null && root.right==null) return root.val==targetSum;
    int remainingSum = targetSum - root.val;
    return hasPathSum(root.left,remainingSum) || hasPathSum(root.right,remainingSum);
}
```

### 543. Diameter of Binary Tree

Given the root of a binary tree, return the length of the diameter of the tree.

The diameter of a binary tree is the length of the longest path between any two nodes in a tree. This path may or may not pass through the root.

The length of a path between two nodes is represented by the number of edges between them.

**Example 1**:

                1
               / \
              2   3
             / \
            4   5    
    Input: root = [1,2,3,4,5]
    Output: 3
    Explanation: 3 is the length of the path [4,2,1,3] or [5,2,1,3].

```java
class Solution {
    public int diameter = 0;
    public int heightOfBinaryTree(TreeNode root){
        if (root==null) return 0;
        int leftHeight = heightOfBinaryTree(root.left);
        int rightHeight = heightOfBinaryTree(root.right);
        diameter = Math.max(leftHeight+rightHeight,diameter);
        return 1+Math.max(leftHeight,rightHeight);
    }
    public int diameterOfBinaryTree(TreeNode root) {
        if (root==null) return 0;
        int heightOfTree = this.heightOfBinaryTree(root);
        return diameter;
    }
}
```

### 114. Flatten Binary Tree to Linked List

Given the root of a binary tree, flatten the tree into a "linked list":

The "linked list" should use the same TreeNode class where the right child pointer points to the next node in the list and the left child pointer is always null.
The "linked list" should be in the same order as a pre-order traversal of the binary tree.

**Example 1**:

             1                          1
            / \                          \
           /   \                          2
          2     5      ========>           \
         / \     \                          3
        3   4     6                          \
                                              4
                                               \
                                                5
                                                 \
                                                  6

    Input: root = [1,2,5,3,4,null,6]
    Output: [1,null,2,null,3,null,4,null,5,null,6]


```java
/*

Reverse post order traversal - Right -> Left -> Root
By visiting the right child first, then the left, and processing the current node last, the code processes nodes in reverse pre-order sequence 
(6 -> 5 -> 4 -> 3 -> 2 -> 1)

*/
class Solution {
    public TreeNode prev=null;
    public void flatten(TreeNode root) {
        if (root==null) return;
        flatten(root.right);
        flatten(root.left);
        root.right = prev;
        root.left=null;
        prev = root;
    }
}

```

## Pattern-5 : Recursive Simulation

### 394. Decode String

Given an encoded string, return its decoded string.

The encoding rule is: k[encoded_string], where the encoded_string inside the square brackets is being repeated exactly k times. Note that k is guaranteed to be a positive integer.

You may assume that the input string is always valid; there are no extra white spaces, square brackets are well-formed, etc. Furthermore, you may assume that the original data does not contain any digits and that digits are only for those repeat numbers, k. For example, there will not be input like 3a or 2[4].

The test cases are generated so that the length of the output will never exceed 105.

 

**Example 1**:

    Input: s = "3[a]2[bc]"
    Output: "aaabcbc"

**Example 2**:

    Input: s = "3[a2[c]]"
    Output: "accaccacc"

**Example 3**:

    Input: s = "2[abc]3[cd]ef"
    Output: "abcabccdcdcdef"

```python
def decodeString(self, s: str) -> str:
    stack = []
    for i in range(len(s)):
        if s[i]!=']':
            stack.append(s[i])
        else:
            substr = ""
            while stack[-1]!='[':
                substr = stack.pop() + substr
            # pop remaining '[' - no need for empty check, already done
            stack.pop()
            digit_k = ""
            while stack and stack[-1].isdigit():
                digit_k = stack.pop() + digit_k
            stack.append(int(digit_k)*substr)
    return "".join(stack)
```

```java
import java.util.Stack;

public String decodeString(String s) {
    Stack<String> stack = new Stack<>();
    
    for (int i = 0; i < s.length(); i++) {
        char c = s.charAt(i);
        
        if (c != ']') {
            stack.push(String.valueOf(c));
        } else {
            // 1. Extract the substring inside the brackets
            StringBuilder substr = new StringBuilder();
            while (!stack.peek().equals("[")) {
                substr.insert(0, stack.pop());
            }
            
            // Pop the '['
            stack.pop();
            
            // 2. Extract the multiplier k
            StringBuilder digitK = new StringBuilder();
            while (!stack.isEmpty() && Character.isDigit(stack.peek().charAt(0))) {
                digitK.insert(0, stack.pop());
            }
            
            int count = Integer.parseInt(digitK.toString());
            
            // 3. Multiply and push back onto the stack
            StringBuilder expanded = new StringBuilder();
            for (int j = 0; j < count; j++) {
                expanded.append(substr);
            }
            stack.push(expanded.toString());
        }
    }
    
    // Combine everything remaining in the stack
    StringBuilder result = new StringBuilder();
    for (String str : stack) {
        result.append(str);
    }
    
    return result.toString();
}

```

### 385. Mini Parser

Given a string s represents the serialization of a nested list, implement a parser to deserialize it and return the deserialized NestedInteger.

Each element is either an integer or a list whose elements may also be integers or other lists.

**Example 1**:

    Input: s = "324"
    Output: 324
    Explanation: You should return a NestedInteger object which contains a single integer 324.

**Example 2**:

    Input: s = "[123,[456,[789]]]"
    Output: [123,[456,[789]]]
    Explanation: Return a NestedInteger object containing a nested list with 2 elements:
    1. An integer containing value 123.
    2. A nested list containing two elements:
        i.  An integer containing value 456.
        ii. A nested list with one element:
            a. An integer containing value 789

**Constraints**:

    1 <= s.length <= 5 * 10^4
    s consists of digits, square brackets "[]", negative sign '-', and commas ','.
    s is the serialization of valid NestedInteger.
    All the values in the input are in the range [-10^6, 10^6].

```java

/**
 * // This is the interface that allows for creating nested lists.
 * // You should not implement it, or speculate about its implementation
 * public interface NestedInteger {
 *     // Constructor initializes an empty nested list.
 *     public NestedInteger();
 *
 *     // Constructor initializes a single integer.
 *     public NestedInteger(int value);
 *
 *     // @return true if this NestedInteger holds a single integer, rather than a nested list.
 *     public boolean isInteger();
 *
 *     // @return the single integer that this NestedInteger holds, if it holds a single integer
 *     // Return null if this NestedInteger holds a nested list
 *     public Integer getInteger();
 *
 *     // Set this NestedInteger to hold a single integer.
 *     public void setInteger(int value);
 *
 *     // Set this NestedInteger to hold a nested list and adds a nested integer to it.
 *     public void add(NestedInteger ni);
 *
 *     // @return the nested list that this NestedInteger holds, if it holds a nested list
 *     // Return empty list if this NestedInteger holds a single integer
 *     public List<NestedInteger> getList();
 * }
 */

import java.util.Stack;

public NestedInteger deserialize(String s) {
    if (s == null || s.isEmpty()) {
        return null;
    }

    // Case 1: The input string is just a single integer (e.g., "324")
    if (s.charAt(0) != '[') {
        return new NestedInteger(Integer.parseInt(s));
    }

    Stack<NestedInteger> stack = new Stack<>();
    NestedInteger current = null;
    int l = 0; // Pointer to track the start of a number

    for (int r = 0; r < s.length(); r++) {
        char ch = s.charAt(r);

        if (ch == '[') {
            NestedInteger ni = new NestedInteger();
            if (current != null) {
                stack.push(current);
            }
            current = ni;
            l = r + 1;
        } else if (ch == ']' || ch == ',') {
            // If we hit a closing bracket or comma right after a number, parse it
            if (r > l) {
                int val = Integer.parseInt(s.substring(l, r));
                current.add(new NestedInteger(val));
            }
            l = r + 1; // Move start pointer past the comma/bracket

            // If closing a list, attach current NestedInteger to its parent
            if (ch == ']' && !stack.isEmpty()) {
                NestedInteger parent = stack.pop();
                parent.add(current);
                current = parent;
            }
        }
    }

    return current;
}

```