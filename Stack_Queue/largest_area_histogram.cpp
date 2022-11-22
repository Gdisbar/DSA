84. Largest Rectangle in Histogram
=======================================
// Given an array of integers heights representing the histogram''s bar height 
// where the width of each bar is 1, return the area of the largest rectangle 
// in the histogram.

//                   | 
//                 | |
//                 | |
//                 | |   |
//             |   | | | |
//             | | | | | |
//             _ _ _ _ _ _

// Example 1:

// Input: a = [2,1,5,6,2,3]
// Output: 10
// Explanation: The above is a histogram where width of each bar is 1.
// The largest rectangle is shown in the red area, which has an 
// area = 10 units(min(5,6)).


// a[i=0]=2,s={<0,2>} 
// a[i=1]=1,s={<0,2>} -> idx=0,h=2,s={},mx=(1-0)*2=2,st=0 -> s={<0,1>}
// a[i=2]=5,s={<0,1>} -> s={<0,1>,<2,5>}
// a[i=3]=6,s={<0,1>,<2,5>} -> s={<0,1>,<2,5>,<3,6>}
// a[i=4]=2,s={<0,1>,<2,5>,<3,6>} -> idx=3,h=6,s={<0,1>,<2,5>},mx=(4-3)*6=6,st=3
//         -> idx=2,h=5,s={<0,1>},mx=(4-2)*5=10,st=2 
// a[i=5]=3,s={<0,1>,<2,2>} -> s={<0,1>,<2,2>,<5,3>},st=5,mx=10



// Example 2:

// Input: heights = [2,4]
// Output: 4 (4,2+2)


int largestRectangleArea(vector<int>& heights) {
        stack<pair<int,int>> s; //index,height
        int n=heights.size(),start=0,area=0;
        for(int i=0;i<n;++i){
            start=i;
            while(!s.empty()&&s.top().second>heights[i]){ //found a smaller height
                int idx=s.top().first,hei=s.top().second;
                s.pop();
                area=max(area,(i-idx)*hei);
                start=idx; //keepting track of s.top() index,as we compare each of previous width
            }
            s.push(make_pair(start,heights[i]));
        }
        while(!s.empty()){
            area=max(area,s.top().second*(n-s.top().first));
            s.pop();
        }
        return area;
    }