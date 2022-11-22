Common elements in all rows of a given matrix
===============================================
Given an r x c matrix, find all common elements present in all rows in O(mn) 
time and one traversal of matrix.

Example: 

Input:
mat[4][5] = {{1, 2, 1, 4, 8},
             {3, 7, 8, 5, 1},
             {8, 7, 7, 3, 1},
             {8, 1, 2, 7, 9},
            };

Output: 
1 8 or 8 1
8 and 1 are present in all rows.

//https://www.geeksforgeeks.org/common-elements-in-all-rows-of-a-given-matrix/

//M-1 , subproblem lets do it for single element
sort all rows in r*c*log(c) & find common element in r*c using binary search ,but
this can be improved by using merge sort like approach 

Returns common element in all rows of mat[M][N]. If there is no
common element, then -1 is returned

Input: mat[4][5] = { {1, 2, 3, 4, 5},
                    {2, 4, 5, 8, 10},
                    {3, 5, 7, 9, 11},
                    {1, 3, 5, 7, 9},
                  };
Output: 5

column = {4, 4, 4, 4}  --> {4, 3, 3, 3}, min_row=0 --> {4, 2, 2, 2}, min_row=0
         --> {4, 2, 1, 2},min_row=0
         
//TC : r*c

int findCommon(vector<vector<int>> &mat,int r,int c){
    // An array to store indexes of current last column
    // Initialize current last element of all rows
    vector<int> column(r,c-1);
    int min_row=0; // To store index of row whose current
    // last element is minimum,Initialize min_row as first row
 
    // Keep finding min_row in current last column, till either
    // all elements of last column become same or we hit first column.
    while (column[min_row] >= 0) { //initally column[0]=c-1
        // Find minimum in current last column
        for (int i = 0; i < r; i++) {
            if (mat[i][column[i]] < mat[min_row][column[min_row]]) 
                min_row = i;
        }
 
        // eq_count is count of elements equal to minimum in current last
        // column.
        int eq_count = 0;
 
        // Traverse current last column elements again to update it
        for (i = 0; i < r; i++) {
            // Decrease last column index of a row whose value is more
            // than minimum.
            if (mat[i][column[i]] > mat[min_row][column[min_row]]) {
                if (column[i] == 0) // reached 1st column min element doesn''t exist
                    return -1;
 
                column[i] -= 1; // Reduce last column index by 1
            }
            else
                eq_count++;
        }
 
        // If equal count becomes r, return the value
        if (eq_count == r)
            return mat[min_row][column[min_row]];
    }
    return -1;
}

 Optimizing the above code,this concept will be used to find all common element

// Step1:  Create a Hash Table with all key as distinct elements 
//         of row1. Value for all these will be 0.

// Step2:  
// For i = 1 to M-1
//  For j = 0 to N-1
//   If (mat[i][j] is already present in Hash Table)
//    If (And this is not a repetition in current row.
//       This can be checked by comparing HashTable value with
//       row number)
//          Update the value of this key in HashTable with current 
//          row number

// Step3: Iterate over HashTable and print all those keys for 
//        which value = M

 Returns common element in all rows of mat[M][N]. If there is no
 common element, then -1 is returned

//TC : r*c

int findCommon(vector<vector<int>> &mat,int r,int c)
{
    // A hash map to store count of elements
    unordered_map<int, int> cnt;
 
    int i, j;
 
    for (i = 0; i < r; i++) {
 
        // Increment the count of first
        // element of the row
        cnt[mat[i][0]]++; //we're starting from j=1,so just defining edge case of j=0
 
        // Starting from the second element/column
        // of the current row
        for (j = 1; j < c; j++) {
 
            // If current element is different from
            // the previous element i.e. it is appearing
            // for the first time in the current row , we avoid duplicate count
            //if we consider duplicate & that duplicate is common element then
            //cnt[duplicate_element] > r
            if (mat[i][j] != mat[i][j - 1]) //rows are not sorted
                cnt[mat[i][j]]++;
        }
    }
 
    // Find element having count equal to number of rows
    for (auto ele : cnt) {
        if (ele.second == r)
            return ele.first;
    }
 
    // No such element found
    return -1;
}

//Now extending above idea for finding all common elements, TC : r*c

void printCommonElements(vector<vector<int>> &mat)
{
	int r = mat.size(),c=mat[0].size();
    unordered_map<int, int> mp;
 
    // initialize 1st row elements with value 1
    for (int j = 0; j < c; j++)
        mp[mat[0][j]] = 1;
 
    // traverse the matrix
    for (int i = 1; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            // If element is present in the map and
            // is not duplicated in current row.

            if (mp[mat[i][j]] == i)
            {
               // we increment count of the element in map by 1 so that
               // if we get this element in next row it meets the condition mp[mat[i][j]] == i
               //as it increments once for 1st occurrence of the common element in that 
               //row mp[mat[i][j]] == i condition holds true & avoid counting duplicates
                mp[mat[i][j]] = i + 1; 
 
                // If this is last row
                if (i==r-1 && mp[mat[i][j]]==r)
                  cout << mat[i][j] << " ";
            }
        }
    }
}

