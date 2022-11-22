Sorted matrix 
===============
Given an NxN matrix Mat. Sort all elements of the matrix.

Example 1:

Input:
N=4
Mat=[[10,20,30,40],
[15,25,35,45] 
[27,29,37,48] 
[32,33,39,50]]
Output:
10 15 20 25 
27 29 30 32
33 35 37 39
40 45 48 50
Explanation:
Sorting the matrix gives this result.

//TC : r*c*log(r*c) , SC : r*c

vector<vector<int>> sortedMatrix(vector<vector<int>> Mat,int r,int c) {
        vector<int> tmp(r*c);
        for(int i = 0;i<r;i++)for(int j=0;j<c;j++) tmp[i*c+j]=Mat[i][j];
        sort(tmp.begin(),tmp.end());
        for(int i = 0;i<r*c;i++)
           Mat[i/c][i%c]=tmp[i];
        return Mat;
    }


//sort a 2-D matrix by second column
sort(Mat.begin(), Mat.end(),[] sortcol(const vector<int>& v1, const vector<int>& v2){
    return v1[1] < v2[1];
});

{3, 5, 1}, 
{4, 8, 6}, 
{7, 2, 9};

After sorting this matrix by the second column, we get

{7, 2, 9} // Row with smallest value in second column 
{3, 5, 1} // Row with smallest value in second column 
{4, 8, 6}


