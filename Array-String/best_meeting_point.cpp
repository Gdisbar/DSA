Best Meeting Point/Minimum Travel
====================================
// 1. A group of two or more people wants to meet and minimize the total 
// travel distance.
// 2. You are given a 2D grid of values 0 or 1, where each 1 marks the 
// home of someone in the group. 
// 3. Return min distance where distance is calculated using 
// 'Manhattan Distance', where distance(p1, p2) = |p2.x - p1.x| + |p2.y - p1.y|.

// Input Format

// [
//     [1,0,|0|,0,1],
//     [0,0, 0, 0,0],
//     [0,0, 1, 0,0]
// ]

// Output Format

// 6

// Explanation:
// The point (0,2) is an ideal meeting point, as the total travel distance 
// of 2 + 2 + 2 = 6 is minimal. So return 6.

// Constraints

// 1. Distance should me minimum

// Sample Input

// 3 5
// 1 0 0 0 1
// 0 0 0 0 0
// 0 0 1 0 0

// Sample Output

// 6

// Median is the best meeting point , so we make x=[] & y=[] containing the row &
// column index of '1' , find median & calculate the distance of median from x=[] & y=[]

//M-1 , finding median traverse row & column & store value & then sort them 
//--> middle element of sorted array in median
//but if matrix has all ones the sorting time r*c*log(r*c)

//M-2 , we don't sort x=[] & y=[], rather than we insert them in sorted order

int bestMeetingPoint(vector<vector<int>> &grid){
    int r=grid.size(),c=grid[0].size();
    vector<int> x,y;
    //traverse row-wise for sorted insertion of x co-ordinate
    for(int i=0;i<r;i++){
      for(int j=0;j<c;j++){
        if(grid[i][j])
          x.push_back(i);
      }
    }
    //traverse column-wise for sorted insertion of y co-ordinate
    for(int j=0;j<c;j++){
      for(int i=0;i<r;i++){
        if(grid[i][j])
          y.push_back(j);
      }
    }
    
    int xm=x[x.size()/2],ym=y[y.size()/2],ans=0;
    for(int i=0;i<x.size();i++){
      ans+=abs(x[i]-xm)+abs(y[i]-ym);
    }
    return ans;
}

