1039. Minimum Score Triangulation of Polygon
=============================================
You have a convex n-sided polygon where each vertex has an integer value. 
You are given an integer array values where values[i] is the value of the 
ith vertex (i.e., clockwise order).

You will triangulate the polygon into n - 2 triangles. For each triangle, 
the value of that triangle is the product of the values of its vertices, and the 
total score of the triangulation is the sum of these values over all n - 2 
triangles in the triangulation.

Return the smallest possible total score that you can achieve with some 
triangulation of the polygon.

 

Example 1:
								   1
								 /   \
							  2 /-----3

Input: values = [1,2,3]
Output: 6
Explanation: The polygon is already triangulated, and the score of the only 
triangle is 6.

Example 2:
									3------7
									|   /  |
									|  /   |
								    5------4

								    3------7
									|  \   |
									|   \  |
								    5------4
Input: values = [3,7,4,5]
Output: 144
Explanation: There are two triangulations, with possible 
scores: 3*7*5 + 4*5*7 = 245, or 3*4*5 + 3*4*7 = 144.
The minimum score is 144.

Example 3: 
                                             4
									1\-------/
                           		  /	| \     / 
                           		 /	|  \   /   
							    /	|   \ /    
							    3   |    / 1
							     \  |   / \
							      \ |  /   \
							       \| /     \
                                    1/-------5
Input: values = [1,3,1,4,1,5]
Output: 13
Explanation: The minimum score triangulation has 
score 1*1*3 + 1*1*4 + 1*1*5 + 1*1*1 = 13.


class Solution {
private:
    vector<vector<int>> dp;
    int helper(vector<int>& a,int i,int j){
        if(j-i<2) return 0;
        if(dp[i][j]!=0) return dp[i][j];
        int mn=INT_MAX;
        for(int k=i+1;k<j;++k){
            int score=a[i]*a[k]*a[j]+helper(a,i,k)+helper(a,k,j);
            mn=min(mn,score);
        }
        return dp[i][j]=mn;
    }
public:
    int minScoreTriangulation(vector<int>& values) {
        int n=values.size();
        dp.resize(n,vector<int>(n,0));
        return helper(values,0,n-1);
    }
};

//dp

int minScoreTriangulation(vector<int>& a) {
        int n=a.size();
        vector<vector<int>> dp(n,vector<int>(n,0));
        for(int i=n-1;i>=0;--i){
            for(int j=i+1;j<n;++j){
                for(int k=i+1;k<j;++k){
                    int score = a[i]*a[k]*a[j]+dp[i][k]+dp[k][j];
                    if(dp[i][j]==0)
                        dp[i][j]=score;
                    else
                        dp[i][j]=min(dp[i][j],score);   
                }
            }
        }
        return dp[0][n-1];
    }