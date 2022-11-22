1691. Maximum Height by Stacking Cuboids 
=============================================
// Given n cuboids where the dimensions of the ith cuboid is 
// cuboids[i] = [widthi, lengthi, heighti] (0-indexed). Choose a subset of cuboids 
// and place them on each other.

// You can place cuboid i on cuboid j if widthi <= widthj and lengthi <= lengthj and 
// heighti <= heightj. You can rearrange any cuboid''s dimensions by rotating it to put 
// it on another cuboid.

// Return the maximum height of the stacked cuboids.

 

// Example 1:

// Input: cuboids = [[50,45,20],[95,37,53],[45,23,12]]
// Output: 190
// Explanation:
// Cuboid 1 is placed on the bottom with the 53x37 side facing down with height 95.
// Cuboid 0 is placed next with the 45x20 side facing down with height 50.
// Cuboid 2 is placed next with the 23x12 side facing down with height 45.
// The total height is 95 + 50 + 45 = 190.

// Example 2:

// Input: cuboids = [[38,25,45],[76,35,3]]
// Output: 76
// Explanation:
// You can''t place any of the cuboids on the other.
// We choose cuboid 1 and rotate it so that the 35x3 side is facing down and its 
// height is 76.

// Example 3:

// Input: cuboids = [[7,11,17],[7,17,11],[11,7,17],[11,17,7],[17,7,11],[17,11,7]]
// Output: 102
// Explanation:
// After rearranging the cuboids, you can see that all cuboids have the same dimension.
// You can place the 11x7 side down on all cuboids so their heights are 17.
// The maximum height of stacked cuboids is 6 * 17 = 102.

// DP - n*n

// If the question is:
// "You can place cuboid i on cuboid j if width[i] <= width[j] and 
// length[i] <= length[j]"
// that's will be difficult.

// But it's
// "You can place cuboid i on cuboid j if width[i] <= width[j] and 
// length[i] <= length[j] and height[i] <= height[j]"
// That's much easier.

cuboids = [[50,45,20],[95,37,53],[45,23,12]] //[width, length, height]
c [ [ 20 45 50 ] [ 37 53 95 ] [ 12 23 45 ] [ 0 0 0 ] ] //1st sort
c [ [ 0 0 0 ] [ 12 23 45 ] [ 20 45 50 ] [ 37 53 95 ] ] //2nd sort



int maxHeight(vector<vector<int>>& c) {
        for(auto &x : c)
            sort(x.begin(),x.end());      //[width, length, height]
        c.push_back({0,0,0});
        sort(c.begin(),c.end());
        int n = c.size(),mx=0;
        vector<int> dp(n,0);
        for(int j=1;j<n;++j){
            for(int i=0;i<j;++i){ //comapare upto last place cuboid
                if(c[i][0]<=c[j][0]&&c[i][1]<=c[j][1]&&c[i][2]<=c[j][2]) //(width,length) = (c[i][0],c[i][1]) , height = c[i][2]
                    dp[j]=max(dp[j],dp[i]+c[j][2]);
                mx=max(mx,dp[j]);
            }
        }
        return mx;
    }


    // def maxHeight(self, A):
    //     A = [[0, 0, 0]] + sorted(map(sorted, A))
    //     dp = [0] * len(A)
    //     for j in xrange(1, len(A)):
    //         for i in xrange(j):
    //             if all(A[i][k] <= A[j][k] for k in xrange(3)):
    //                 dp[j] = max(dp[j], dp[i] + A[j][2])
    //     return max(dp)