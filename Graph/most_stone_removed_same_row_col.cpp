947. Most Stones Removed with Same Row or Column
===================================================
// On a 2D plane, we place n stones at some integer coordinate points. 
// Each coordinate point may have at most one stone.

// A stone can be removed if it shares either the same row or the same column 
// as another stone that has not been removed.

// Given an array stones of length n where stones[i] = [xi, yi] represents the 
// location of the ith stone, return the largest possible number of stones that 
// can be removed.

 

// Example 1:

// Input: stones = [[0,0],[0,1],[1,0],[1,2],[2,1],[2,2]]
// Output: 5
// Explanation: One way to remove 5 stones is as follows:
// 1. Remove stone [2,2] because it shares the same row as [2,1].
// 2. Remove stone [2,1] because it shares the same column as [0,1].
// 3. Remove stone [1,2] because it shares the same row as [1,0].
// 4. Remove stone [1,0] because it shares the same column as [0,0].
// 5. Remove stone [0,1] because it shares the same row as [0,0].
// Stone [0,0] cannot be removed since it does not share a row/column with 
// another stone still on the plane.

// Example 2:

// Input: stones = [[0,0],[0,2],[1,1],[2,0],[2,2]]
// Output: 3
// Explanation: One way to make 3 moves is as follows:
// 1. Remove stone [2,2] because it shares the same row as [2,0].
// 2. Remove stone [2,0] because it shares the same column as [0,0].
// 3. Remove stone [0,2] because it shares the same row as [0,0].
// Stones [0,0] and [1,1] cannot be removed since they do not share a row/column 
// with another stone still on the plane.

// Example 3:

// Input: stones = [[0,0]]
// Output: 0
// Explanation: [0,0] is the only stone on the plane, so you cannot remove it.

// One sentence to solve:

// Connected stones can be reduced to 1 stone,
// the maximum stones can be removed = stones number - islands number.
// so just count the number of "islands".

// 1. Connected stones

// Two stones are connected if they are in the same row or same col.
// Connected stones will build a connected graph.
// It's obvious that in one connected graph,
// we can't remove all stones.

// We have to have one stone left.
// An intuition is that, in the best strategy, we can remove until 1 stone.

// I guess you may reach this step when solving the problem.
// But the important question is, how?

// 2. A failed strategy

// Try to remove the least degree stone
// Like a tree, we try to remove leaves first.
// Some new leaf generated.
// We continue this process until the root node left.

// However, there can be no leaf.
// When you try to remove the least in-degree stone,
// it won''t work on this "8" like graph:
// [[1, 1, 0, 0, 0],
// [1, 1, 0, 0, 0],
// [0, 1, 1, 0, 0],
// [0, 0, 1, 1, 1],
// [0, 0, 0, 1, 1]]

// The stone in the center has least degree = 2.
// But if you remove this stone first,
// the whole connected stones split into 2 parts,
// and you will finish with 2 stones left.

// 3. A good strategy

// In fact, the proof is really straightforward.
// You probably apply a DFS, from one stone to next connected stone.
// You can remove stones in reversed order.
// In this way, all stones can be removed but the stone that you start your DFS.

// One more step of explanation:
// In the view of DFS, a graph is explored in the structure of a tree.
// As we discussed previously,
// a tree can be removed in topological order,
// from leaves to root.

// 4. Count the number of islands

// We call a connected graph as an island.
// One island must have at least one stone left.
// The maximum stones can be removed = stones number - islands number

// The whole problem is transferred to:
// What is the number of islands?

// You can show all your skills on a DFS implementation,
// and solve this problem as a normal one.

// 5. Unify index

// Struggle between rows and cols?
// You may duplicate your codes when you try to the same thing on rows and cols.
// In fact, no logical difference between col index and rows index.

// An easy trick is that, add 10000 to col index.
// So we use 0 ~ 9999 for row index and 10000 ~ 19999 for col.

// 6. Search on the index, not the points

// When we search on points,
// we alternately change our view on a row and on a col.

// We think:
// a row index, connect two stones on this row
// a col index, connect two stones on this col.

// In another view：
// A stone, connect a row index and col.

// Have this idea in mind, the solution can be much simpler.
// The number of islands of points,
// is the same as the number of islands of indexes.

// 7. Union-Find

// I use union find to solve this problem.
// As I mentioned, the elements are not the points, but the indexes.

//     for each point, union two indexes.
//     return points number - union number

// Copy a template of union-find,
// write 2 lines above,
// you can solve this problem in several minutes.

// Complexity

// union and find functions have worst case O(N), amortize O(1)
// The whole union-find solution with path compression,
// has O(N) Time, O(N) Space

// You can write it like this: ~j == -(j+1). In other words this is more like 
// an alter ego of a number. The only issue is we have 0, which is nor positive 
// or negative so it doesn''t have an alter ego that's why we use ~ or -(j+1) 
// instead of just -

// Here are some examples:

// i=0	~i=-1	-(i+1)=-1
// i=1	~i=-2	-(i+1)=-2
// i=2	~i=-3	-(i+1)=-3
// i=3	~i=-4	-(i+1)=-4
// i=4	~i=-5	-(i+1)=-5
// i=5	~i=-6	-(i+1)=-6
// i=6	~i=-7	-(i+1)=-7
// i=7	~i=-8	-(i+1)=-8
// i=8	~i=-9	-(i+1)=-9
// i=9	~i=-10	-(i+1)=-10

// Update About Union Find Complexity

// I have 3 main reasons that always insist O(N), on all my union find solutions.

// The most important, union find is really a common knowledge for algorithm.
// Using both path compression, splitting, or halving and union by rank or size 
// ensures
// that the amortized time per operation is only O(1).
// So it's fair enough to apply this conclusion.

// It's really not my job to discuss how union find works or the definition of 
// big O.
// I bet everyone can find better resource than my post on this part.
// You can see the core of my solution is to transform the problem as a union 
// find problem.
// The essence is the thinking process behind.
// People can have their own template and solve this problem with 2-3 more lines.
// But not all the people get the point.

//     I personally manually write this version of union find every time.
//     It is really not worth a long template.
//     The version with path compression can well handle all cases on leetcode.
//     What‘s the benefit here to add more lines?

//     In this problem, there is N union operation, at most 2 * sqrt(N) node.
//     When N get bigger, the most operation of union operation is amortize O(1).

//     I knew there were three good resourse of union find:

//         top down analusis of path compression
//         wiki
//         stackexchange

//     But they most likely give a upper bound time complexity of union find,
//     not a supreme.
//     If anyone has a clear example of union find operation sequence,
//     to make it larger than O(N), I am so glad to know it.


class Solution {
public:
    int removeStones(vector<vector<int>>& stones) {
        for (int i = 0; i < stones.size(); ++i)
            union_(stones[i][0], ~stones[i][1]);
        return stones.size() - islands;
    }

    unordered_map<int, int> f;
    int islands = 0;

    int find(int x) {
        if (!f.count(x)) f[x] = x, islands++;
        if (x != f[x]) f[x] = find(f[x]);
        return f[x];
    }

    void union_(int x, int y) {
        x = find(x), y = find(y);
        if (x != y) f[x] = y, islands--;
    }
};

//Python

 def removeStones(self, points):
        UF = {}
        def find(x):
            if x != UF[x]:
                UF[x] = find(UF[x])
            return UF[x]
        def union(x, y):
            UF.setdefault(x, x)
            UF.setdefault(y, y)
            UF[find(x)] = find(y)

        for i, j in points:
            union(i, ~j)
        return len(points) - len({find(x) for x in UF})