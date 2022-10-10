42. Trapping Rain Water
==========================
Given n non-negative integers representing an elevation map where the 
width of each bar is 1, compute how much water it can trap after raining.


Example 1:  
                              
                              |
                      |       | |   |
				  |   | |   | | | | | |
				-----------------------
				0 1 0 2 1 0 1 3 2 1 2 1

Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
Explanation: The above elevation map (black section) is represented by 
array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water 
(blue section) are being trapped.

Example 2:

Input: height = [4,2,0,3,2,5]
Output: 9

int trap(vector<int>& height) {
        int n=height.size();
        vector<int> left(n),right(n);
        left[0]=height[0];
        for(int i=1;i<n;++i){
            left[i]=max(left[i-1],height[i]);
        }
        right[n-1]=height[n-1];
        for(int i=n-2;i>=0;--i){
            right[i]=max(right[i+1],height[i]);
        }
        int area=0;
        for(int i=0;i<n;++i){
            area+=min(left[i],right[i])-height[i];
        }
        return area;
    }


// faster
int trap(vector<int>& A) {
        int n=A.size();
        int left=0; int right=n-1;
        int res=0;
        int maxleft=0, maxright=0;
        while(left<=right){
            if(A[left]<=A[right]){
                if(A[left]>=maxleft) maxleft=A[left];
                else res+=maxleft-A[left];
                left++;
            }
            else{
                if(A[right]>=maxright) maxright= A[right];
                else res+=maxright-A[right];
                right--;
            }
        }
        return res;
    }


407. Trapping Rain Water II
===============================
// Given an m x n integer matrix heightMap representing the height of each 
// unit cell in a 2D elevation map, return the volume of water it can trap 
// after raining.

 

// Example 1:

// Input: heightMap = [[1,4,3,1,3,2],[3,2,1,3,2,4],[2,3,3,2,3,1]]
// Output: 4
// Explanation: After the rain, water is trapped between the blocks.
// We have two small ponds 1 and 3 units trapped.
// The total volume of water trapped is 4.

// Example 2:

// Input: heightMap = [[3,3,3,3,3],[3,2,2,2,3],[3,2,1,2,3],[3,2,2,2,3],[3,3,3,3,3]]
// Output: 10

// Imagine the pool is surrounded by many bars. The water can only go out 
// from the lowest bar. So we always start from the lowest boundary and keep 
// pushing the bar from boundary towards inside. It works as if we are replacing 
// the old bars with a bar higher than it.
// See the following simple example:
// 4 4 4 4
// 4 0 1 2
// 4 4 4 4
// it looks like we push the bar of 2 towards left and record the difference. 
// Then you can use the same procedure with the following figure
// 4 4 4 4
// 4 0 2 2
// 4 4 4 4

int trapRainWater(vector<vector<int>>& heightMap) {
        typedef pair<int,int> cell;
        priority_queue<cell, vector<cell>, greater<cell>> q;
        int m = heightMap.size();
        if (m == 0) return 0;
        int n = heightMap[0].size();
        vector<int> visited(m*n, false);
        
        for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            if (i == 0 || i == m-1 || j == 0  || j == n-1) {
                if (!visited[i*n+j])
                    q.push(cell(heightMap[i][j], i*n+j));
                visited[i*n+j] = true;
            }
        }
        
        int dir[4][2] = {{0,1}, {0, -1}, {1, 0}, {-1, 0}};
        int ans = 0;
        while(!q.empty()) {
            cell c = q.top();
            q.pop();
            int i = c.second/n, j = c.second%n;
            
            for (int r = 0; r < 4; ++r) {
                int ii = i+dir[r][0], jj = j+dir[r][1];
                if (ii < 0 || ii >= m || jj < 0 || jj >= n || visited[ii*n+jj])
                    continue;
                ans += max(0, c.first - heightMap[ii][jj]);
                q.push(cell(max(c.first, heightMap[ii][jj]), ii*n+jj));
                visited[ii*n+jj] = true;
            }
        }
        return ans;
    }