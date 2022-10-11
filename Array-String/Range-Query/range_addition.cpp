903 · Range Addition
=====================
// Assume you have an array of length n initialized with all 0's and are given k 
// update operations.Each operation is represented as a 
// triplet: [startIndex, endIndex, inc] which increments each element of subarray 
// A[startIndex ... endIndex] (startIndex and endIndex inclusive) with inc.

// Return the modified array after all k operations were executed.


// Example

// Input : length = 5,updates = [[1,  3,  2],[2,  4,  3],[0,  2, -2]]
// Output: [-2, 0, 3, 5, 3]

// Explanation:

// Initial state:                      [ 0, 0, 0, 0, 0 ]
// After applying operation [1, 3, 2]: [ 0, 2, 2, 2, 0 ]
// After applying operation [2, 4, 3]: [ 0, 2, 5, 5, 3 ]
// After applying operation [0, 2, -2]:[-2, 0, 3, 5, 3 ]


//using heap - nlogn

public int[] getModifiedArray(int length, int[][] updates) {
    int result[] = new int[length];
    if(updates==null || updates.length==0)
        return result;
 
    //sort updates by starting index
    Arrays.sort(updates, new Comparator<int[]>(){
        public int compare(int[] a, int [] b){
            return a[0]-b[0];
        }
    });
 
    ArrayList<int[]> list = new ArrayList<int[]>();
 
    //create a heap sorted by ending index
    PriorityQueue<Integer> queue = new PriorityQueue<Integer>(new Comparator<Integer>(){
        public int compare(Integer a, Integer b){
            return updates[a][1]-updates[b][1];
        }
    });
 
    int sum=0;
    int j=0;
    for(int i=0; i<length; i++){
        //substract value from sum when ending index is reached
        while(!queue.isEmpty() && updates[queue.peek()][1] < i){
            int top = queue.poll();
            sum -= updates[top][2];    
        }
 
        //add value to sum when starting index is reached
        while(j<updates.length && updates[j][0] <= i){
           sum = sum+updates[j][2];
           queue.offer(j);
           j++;
        }
 
        result[i]=sum;    
    }
 
    return result;
}

// We can track each range's start and end when iterating over the ranges. 
// And in the final result array, adjust the values on the change points. 

//after update
// ans [ 0 2 0 0 -2 0 ]
// ans [ 0 2 3 0 -2 -3 ]
// ans [ -2 2 3 2 -2 -3 ]
//suffix sum before removing last element
// -2 2 3 2 -2 -3 
// -2 0 3 5 3 -3 


vector<int> getModifiedArray(int length,vector<vector<int>>&update){
	vector<int> ans(length+1,0);
	for (int i=0;i<update.size();++i){
		ans[update[i][0]]+=update[i][2];
		ans[update[i][1]+1]-=update[i][2];
	}
	for(int i=1;i<length;++i){
		ans[i]+=ans[i-1];
	}
	ans.pop_back();
	return ans;
}

598. Range Addition II
=========================
// You are given an m x n matrix M initialized with all 0's and an array of 
// operations ops, where ops[i] = [ai, bi] means M[x][y] should be incremented by 
// one for all 0 <= x < ai and 0 <= y < bi.

// Count and return the number of maximum integers in the matrix after performing 
// all the operations.

 

// Example 1:

// Input: m = 3, n = 3, ops = [[2,2],[3,3]]
// Output: 4
// Explanation: The maximum integer in M is 2, and there are four of it in M. 
// So return 4.

// Example 2:

// Input: m = 3, n = 3, ops = [[2,2],[3,3],[3,3],[3,3],[2,2],[3,3],[3,3],[3,3],
// 							[2,2],[3,3],[3,3],[3,3]]
// Output: 4

// Example 3:

// Input: m = 3, n = 3, ops = []
// Output: 9


// An operation [a,b] add by one for all 0 <= i < a and 0 <= j < b.
// So the number of maximum integers in the matrix after performing all the operations 
// or the integers in matrix that get added by 1 by all operations are the integers 
// that in 0 <=i<min_a and 0<=i<min_b or min_a * min_b

    def maxCount(self, m, n, ops):
        """
        :type m: int
        :type n: int
        :type ops: List[List[int]]
        :rtype: int
        """
        if not ops:
            return m*n
        return min(op[0] for op in ops)*min(op[1] for op in ops)