48. Rotate Image
=====================
Given a square matrix, turn it by 90 degrees in a clockwise direction 
without using any extra space.

Examples: 

Input:
1 2 3 
4 5 6
7 8 9  
Output:
7 4 1 
8 5 2
9 6 3

Input:
1 2
3 4
Output:
3 1
4 2 

//TC : n*n (square matrix)

Consider a 3 x 3 matrix having indices (i, j) as follows. 

00 01 02 
10 11 12 
20 21 22

After rotating the matrix by 90 degrees in clockwise direction, indices 
transform into
20 10 00  current_row_index = 0, i = 2, 1, 0 
21 11 01 current_row_index = 1, i = 2, 1, 0 
22 12 02  current_row_index = 2, i = 2, 1, 0

Observation: In any row, for every decreasing row index i, there exists a constant 
column index j, such that j = current_row_index. 

void rotate90Clockwise(int arr[N][N])
{
    // printing the matrix on the basis of
    // observations made on indices.
    for (int j = 0; j < N; j++)
    {
        for (int i = N - 1; i >= 0; i--)
            cout << arr[i][j] << " ";
        cout << '\n';
    }
}

//M-2

The only thing that is different is to print the elements of the cycle in a 
clockwise direction i.e. An N x N matrix will have floor(N/2) square cycles

Let size of row and column be 3. 
During first iteration – 
a[i][j] = Element at first index (leftmost corner top)= 1.
a[j][n-1-i]= Rightmost corner top Element = 3.
a[n-1-i][n-1-j] = Rightmost corner bottom element = 9.
a[n-1-j][i] = Leftmost corner bottom element = 7.
Move these elements in the clockwise direction. 
During second iteration – 
a[i][j] = 2.
a[j][n-1-i] = 6.
a[n-1-i][n-1-j] = 8.
a[n-1-j][i] = 4. 
Similarly, move these elements in the clockwise direction. 


void rotate(vector<vector<int>>& matrix) {
        int n = matrix.size();
        // Traverse each cycle
        for(int i = 0;i<n/2;i++){
            for(int j = i;j<n-i-1;j++){
                // Swap elements of each cycle
            // in clockwise direction ------> 1 - 3 - 9 - 7 - 1
                int tmp=matrix[i][j]; //1
                matrix[i][j]=matrix[n-j-1][i]; //7
                matrix[n-j-1][i]=matrix[n-i-1][n-j-1]; //9
                matrix[n-i-1][n-j-1]=matrix[j][n-i-1]; //3
                matrix[j][n-i-1]=tmp;
            }
        }
    }