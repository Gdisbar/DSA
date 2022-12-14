305 Number of Islands II
===========================
A 2d grid map of m rows and n columns is initially filled with water. 
We may perform an addLand operation which turns the water at position 
(row, col) into a land. Given a list of positions to operate, count the number of islands after each addLand operation. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

Example:

Given m = 3, n = 3, positions = [[0,0], [0,1], [1,2], [2,1]]. Initially, the 2d grid grid is filled with water. (Assume 0 represents water and 1 represents land).

0 0 0 
0 0 0
0 0 0

Operation #1: addLand(0, 0) turns the water at grid[0][0] into a land.

1 0 0
0 0 0   Number of islands = 1
0 0 0

Operation #2: addLand(0, 1) turns the water at grid[0][1] into a land.

1 1 0
0 0 0   Number of islands = 1
0 0 0

Operation #3: addLand(1, 2) turns the water at grid[1][2] into a land.

1 1 0
0 0 1   Number of islands = 2
0 0 0

Operation #4: addLand(2, 1) turns the water at grid[2][1] into a land.

1 1 0
0 0 1   Number of islands = 3
0 1 0

We return the result as an array: [1, 1, 2, 3]

//Union found

    public List<Integer> numIslands2(int m, int n, int[][] positions) {
        List<Integer> res = new ArrayList();
        if(m == 0 || n == 0 || positions.length == 0) return res;

        int union[] = new int[m * n];
        int size = 0;

        Arrays.fill(union, -1);

        for(int []pos : positions){
            int i = pos[0], j = pos[1];
            union[i * n + j] = i * n + j;
            size += findUnion(union, i, j, m, n);
            res.add(size);
        }

        return res;
    }

    private int[] x = {-1,1,0,0};
    private int[] y = {0,0,-1,1}; 

    private int findUnion(int [] union, int i, int j, int m, int n){

        int diff = 1;

        for(int dir = 0; dir < 4; dir ++){

            int target = i * n + j, curr = (i + x[dir]) * n + j + y[dir];

            if(i >= - x[dir] && i < m - x[dir] && j >= - y[dir] && j < n - y[dir] && union[curr] != -1){

                while(union[target] != target){
                    target = union[target];
                }

                while(union[curr] != curr){
                    curr = union[curr];
                }

                if(target != curr){
                    union[curr] = target;
                    diff --;
                }
            }
        }

        return diff;
    }

Use Union-Find Set to solve this problem. to use this alogrithm, we need to design a set of APIs we can follow while updating the required data structure, this API + data structure will:

    assign a id for each element(the data structure), in this case, each island/water area initially has -1 as its id.
    API-1 find(map, i...), return the id for certain element in the map.

    If 2 id returned is not the same, then use API-2 union() try to merge elements with these two ids.

    Quick-Find solution, O(kmn)

public class Solution {
    //this is a quick_find solution;
    public List<Integer> numIslands2(int m, int n, int[][] positions) {
        int[] lands = new int[m*n];
        ArrayList<Integer> result = new ArrayList<>();
        int count =0;
        int[][] neighbors = {{1,0},{-1,0},{0,1},{0,-1}};
        for(int i=0; i< m*n; i++){
            lands[i] = -1;
        }
        for(int i=0; i<positions.length;i++){

            int pX= positions[i][0];
            int pY = positions[i][1];
            if( lands[pX*n+pY]!= -1) continue;

            count++;
            lands[pX*n+pY] = pX*n+pY;
            for(int k=0; k<neighbors.length; k++){
                int nX = pX+ neighbors[k][0];
                int nY = pY + neighbors[k][1];

                if(nX >=0 && nX<m && nY >=0 && nY <n && lands[nX*n+nY]!=-1 && lands[nX*n+nY] != lands[pX*n+pY]){
                    count--;
                    union(lands, lands[nX*n+nY], lands[pX*n+pY]);
                }
            }

            result.add(count);
        }

        return result;
    }

    private void union(int[] lands, int pId, int qId){
        for(int i=0;i<lands.length; i++){
            if(lands[i] == qId) lands[i] = pId;
        }
    }
}

    Quick-Union, O(klgmn)

public class Solution {
    //this is a quick_union solution;
    public List<Integer> numIslands2(int m, int n, int[][] positions) {
        int[] lands = new int[m*n];
        ArrayList<Integer> result = new ArrayList<>();
        int count =0;
        int[][] neighbors = {{1,0},{-1,0},{0,1},{0,-1}};
        for(int i=0; i< m*n; i++){
            lands[i] = -1;
        }
        for(int i=0; i<positions.length;i++){

            int pX= positions[i][0];
            int pY = positions[i][1];
            if(lands[pX*n+pY]!= -1) continue;
            count++;
            lands[pX*n+pY] = pX*n+pY;
            for(int k=0; k<neighbors.length; k++){
                int nX = pX+ neighbors[k][0];
                int nY = pY + neighbors[k][1];

                if(nX >=0 && nX<m && nY >=0 && nY <n && lands[nX*n+nY]!=-1){
                    int pRoot = find(lands, pX*n+pY);
                    int nRoot = find(lands, nX*n+nY);
                    if(pRoot != nRoot){
                        count--;
                        lands[pRoot] = nRoot;// union happens here
                    }

                }
            }

            result.add(count);
        }

        return result;
    }

    private int find(int[] lands, int index){
        while(index != lands[index]) index = lands[index];
        return index;
    }
}

the problem with above solution is that when union happens, each set connects randomly, which cause the next find may be a deep traverse along the tree. by using a weighted method this can be reduced.

    Weighted Quick-Union, reduce the worst case. time consumption reduce a bit.

for(int i=0; i< m*n; i++){
   lands[i] = -1;
   sz[i] = 1;
}
...// same as above;
count--;
// make smaller tree points to larget tree.
if(sz[pRoot] < sz[nRoot]){
    lands[pRoot] = nRoot;
    sz[nRoot] += sz[pRoot];
}else{
    lands[nRoot] = pRoot;
    sz[pRoot] += sz[nRoot];
}

we can still improve the performance using path compression, simply update the root information in the find() method

//method 1, the entire tree is flatten
int old = index;
while(index != lands[index]) index = lands[index]; 
while(old != index){
     int tmp = lands[old];
     lands[old] = index;
     old = tmp;
}
//method 2.
for(;index != lands[index]; index = lands[index])
   lands[index] = lands[lands[index]];
