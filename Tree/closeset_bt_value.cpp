900 · Closest Binary Search Tree Value
=========================================
// Given a non-empty binary search tree and a target value, find the value in the 
// BST that is closest to the target.


//     Given target value is a floating point.
//     You are guaranteed to have only one unique value in the BST that 
//     is closest to the target.

// Example

// Input: root = {5,4,9,2,#,8,10} and target = 6.124780

// Output: 5

// Explanation：

// Binary tree {5,4,9,2,#,8,10},  denote the following structure:

//         5

//        / \

//      4    9

//     /    / \

//    2    8  10

// Example2

// Input: root = {3,2,4,1} and target = 4.142857

// Output: 4

// Explanation：

// Binary tree {3,2,4,1},  denote the following structure:

//      3

//     / \

//   2    4

//  /

// 1

public class Solution {
    int goal;
    double min = Double.MAX_VALUE;
 
    public int closestValue(TreeNode root, double target) {
        helper(root, target);
        return goal;
    }
 
    public void helper(TreeNode root, double target){
        if(root==null)
            return;
 
        if(Math.abs(root.val - target) < min){
            min = Math.abs(root.val-target);
            goal = root.val;
        } 
 
        if(target < root.val){
            helper(root.left, target);
        }else{
            helper(root.right, target);
        }
    }
}

901 · Closest Binary Search Tree Value II
===========================================
// Given a non-empty binary search tree and a target value, 
// find k values in the BST that are closest to the target.


// Input:

// {1}

// 0.000000

// 1

// Output:

// [1]

// Explanation：

// Binary tree {1},  denote the following structure:

//  1


// Input:

// {3,1,4,#,2}

// 0.275000

// 2

// Output:

// [1,2]

// Explanation：

// Binary tree {3,1,4,#,2},  denote the following structure:

//   3

//  /  \

// 1    4

//  \

//   2

// TC : logn
  
public List<Integer> closestKValues(TreeNode root, double target, int k) {
        LinkedList<Integer> list = new LinkedList<Integer>();
        closestKValuesHelper(list, root, target, k);
        return list;
    }

    /**
     * @return <code>true</code> if result is already found.
     */
    private boolean closestKValuesHelper(LinkedList<Integer> list, 
    				TreeNode root, double target, int k) {
        if (root == null) {
            return false;
        }

        if (closestKValuesHelper(list, root.left, target, k)) {
            return true;
        }

        if (list.size() == k) {
            if (Math.abs(list.getFirst() - target) < Math.abs(root.val - target)) {
                return true;
            } else {
                list.removeFirst();
            }
        }

        list.addLast(root.val);
        return closestKValuesHelper(list, root.right, target, k);
    }