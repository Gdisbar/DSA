54. Spiral Matrix
========================
Given an m x n matrix, return all elements of the matrix in spiral order.

 

Example 1:

Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [1,2,3,6,9,8,7,4,5]

Example 2:

Input: matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
Output: [1,2,3,4,8,12,11,10,9,5,6,7]


    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        int r = matrix.size(),c=matrix[0].size();
        vector<int> ans;
        int top = 0,bottom = r-1,left=0,right=c-1;
        int dir=1;
        while(top<=bottom&&left<=right){
            if(dir==1){
                //left-->right
                for(int i = left;i<=right;++i) 
                    ans.push_back(matrix[top][i]);
                //top row is complete,move down to next row
                ++top;
                dir=2;
            }
            else if(dir==2){
                //top-->bottom
                for(int i = top;i<=bottom;++i) 
                    ans.push_back(matrix[i][right]);
                //right column is complete,move left to previous column
                --right;
                dir=3;
            }
            else if(dir==3){
                //right-->left
                for(int i = right;i>=left;--i) 
                    ans.push_back(matrix[bottom][i]);
                //bottom row is complete,move up to previous row
                --bottom;
                dir=4;
            }
            else if(dir==4){
                //bottom-->top
                for(int i = bottom;i>=top;--i) 
                    ans.push_back(matrix[i][left]);
                //left column is complete,move right to next column
                ++left;
                dir=1;
            }
        }
        return ans;
    }

//Easier approach

    // If the candidate is in the bounds of the matrix and unseen, then it 
    // becomes our next position; otherwise, our next position is the one after 
    // performing a clockwise turn.


vector<int> spiralOrder(vector<vector<int> >& matrix){
    int m = matrix.size(), n = matrix[0].size();
    vector<int> ans;
    if (m == 0) return ans;
 
    vector<vector<bool> > seen(m, vector<bool>(n, false));
    int dr[] = { 0, 1, 0, -1 };
    int dc[] = { 1, 0, -1, 0 };
 
    int x = 0, y = 0, di = 0;
 
    // Iterate from 0 to m * n - 1
    for (int i = 0; i < m * n; i++) {
        ans.push_back(matrix[x][y]);

        seen[x][y] = true;
        int newX = x + dr[di];
        int newY = y + dc[di];
 
        if (0<=newX && newX<m && 0<=newY && newY<n && !seen[newX][newY]) {
            x = newX;
            y = newY;
        }
        else { //after performing clockwise turn
            di = (di + 1) % 4;
            x += dr[di];
            y += dc[di];
        }
    }
    return ans;
}