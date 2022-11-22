//Pattern - 2-D version of Largest Area Histogram (LAH)

//Approach - using LAH concept
vector<int> heights(c,0)
1. for 1st row  heights[i]=1 (if m[0][i]==1) -> area+=LAH(heights)
2. from 2nd row we do height[i]+=1 (if m[i][j]==1 else) height[i]=0 
      + area+=LAH(heights) for each row


//Approach - DP ---> all values can be derived from previous + current rows

1. compute heights //current number of countinous '1' in column j 
   int cur_left=0, cur_right=n; 
// left[i]: the index of leftmost '1' in the current group (containing element i ) in this row.
// right[i]: the index of rightmost '1' plus one in the current group (containing element i ) in this row.
// height[i]: the depth of consecutive ones in this column.
// max(left) and min(r) uses the data from the previous row. uarantee that whatever index left[i] points to has the same height as height[i]

2. compute left //left bound index --> 1st '1' while traversing left to right
    //initalized with 0 i.e no rectangle
    left[j]=max(left[j],cur_left) //m[i][j]==1
    reset to inital value & cur_left=j+1 //otherwise
3. compute right //right bound index --> 1st '1' while traversing right to left
    //initalized with n i.e no rectangle
    right[j]=min(right[j],cur_right) //m[i][j]==1
    reset to inital value & cur_right=j

4. ///from any side
   maxA = max(maxA,(right[j]-left[j])*height[j]);


85. Maximal Rectangle
==========================
Given a rows x cols binary matrix filled with 0''s and 1''s, 
find the largest rectangle containing only 1''s and return its area.

 

Example 1:

Input: matrix = [["1","0","1","0","0"],
	             ["1","0","1","1","1"],
	             ["1","1","1","1","1"],
	             ["1","0","0","1","0"]]
Output: 6
Explanation: The maximal rectangle is shown in the above picture.
heights= [4 0 0 3 0 ]

Example 2:

Input: matrix = [["0"]]
Output: 0
heights=[0]

Example 3:

Input: matrix = [["1"]]
Output: 1
heights=[1]

Example 4:
Input: matrix =[["0","1"],["1","0"]]
Output: 1
heights=[ 1 0 ] 

//75% faster , 61% less memory , TC : n*n(matrix)+n(LAH),
//SC : n(heights)+n(LAH,stack) ---> LAH : Largest Area Histogram

int maximalRectangle(vector<vector<char>>& matrix) {
        int r=matrix.size(),c=matrix[0].size();
        vector<int> heights(c,0);
        for(int j=0;j<c;++j)
            if(matrix[0][j]=='1')
                heights[j]=1;
        int area=largestRectangleArea(heights);
        for(int i=1;i<r;++i){
            for(int j=0;j<c;++j){
                if(matrix[i][j]=='1'){
                    heights[j]+=1;
                }
                else heights[j]=0;
            }
            area=max(area,largestRectangleArea(heights));
        }
        // cout<<endl;
        // for(int i=0;i<c;++i)
        //     cout<<heights[i]<<" ";
        // cout<<endl;
        return area;
    }

//DP solution , 95% faster,98% less memory

// left[i]: the index of leftmost '1' in the current group (containing element i ) in this row.
// right[i]: the index of rightmost '1' plus one in the current group (containing element i ) in this row.
// height[i]: the depth of consecutive ones in this column.
// max(left) and min(r) uses the data from the previous row to guarantee that whatever index left[i] points to has the same height as height[i]




    row 0: 0 0 0 1 0 0 0

height: 0 0 0 1 0 0 0
left: 0 0 0 3 0 0 0
right 7 7 7 4 7 7 7

    row 1: 0 0 1 1 1 0 0

height: 0 0 1 2 1 0 0 
left: 0 0 2 3 2 0 0
right: 7 7 5 4 5 7 7

    row 2: 0 1 1 1 1 1 0

height: 0 1 2 3 2 1 0
left: 0 1 2 3 2 1 0
right: 7 6 5 4 5 6 7


int maximalRectangle(vector<vector<char> > &matrix) {
    if(matrix.empty()) return 0;
    const int m = matrix.size();
    const int n = matrix[0].size();
    int left[n], right[n], height[n]; 
    //left & right bound index : 1st left index on left,1st right index on right
    fill_n(left,n,0); fill_n(right,n,n); fill_n(height,n,0);
    int maxA = 0;
    for(int i=0; i<m; i++) {
        int cur_left=0, cur_right=n; 
        for(int j=0; j<n; j++) { // compute height (can do this from either side)
            if(matrix[i][j]=='1') height[j]++; 
            else height[j]=0;
        }
        for(int j=0; j<n; j++) { // compute left (from left to right)
            if(matrix[i][j]=='1') left[j]=max(left[j],cur_left);
            else {left[j]=0; cur_left=j+1;}
        }
        // compute right (from right to left)
        for(int j=n-1; j>=0; j--) {
            if(matrix[i][j]=='1') right[j]=min(right[j],cur_right);
            else {right[j]=n; cur_right=j;}    
        }
        // compute the area of rectangle (can do this from either side)
        for(int j=0; j<n; j++)
            maxA = max(maxA,(right[j]-left[j])*height[j]);
    }
    return maxA;
}
