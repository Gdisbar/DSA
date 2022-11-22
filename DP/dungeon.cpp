174. Dungeon Game
==================
// Note that any room can contain threats or power-ups, even the first room 
// the knight enters and the bottom-right room where the princess is imprisoned.

//only movement down or right, need to take optimal path from (0,0) to (m-1,n-1)
//values on grid represent the -afffect on health , if health <= 0 knight dies

// Example 1:

// Input: dungeon = [[-2,-3,3],[-5,-10,1],[10,30,-5]]
// Output: 7
// Explanation: The initial health of the knight must be at least 7 if he 
// follows the optimal path: RIGHT-> RIGHT -> DOWN -> DOWN.

// Example 2:

// Input: dungeon = [[0]]
// Output: 1


int getVal(vector<vector<int>> &mat,vector<vector<int>> &dp,int i=0,int j=0){
        int n = mat.size();
        int m = mat[0].size();    
        //outside of matrix for current traversal- invalid position
        if(i == n || j == m)    return 1e9; 
        //reached destination - using one path , explore other paths
        if(i == n-1 and j == m-1)
            return (mat[i][j] <= 0) ? -mat[i][j] + 1 : 1;
        // overlapping value,already computed -  return those values 
        if( dp[i][j] != 1e9)
            return dp[i][j];
        
        int IfWeGoRight = getVal(mat , dp , i , j+1);
        int IfWeGoDown = getVal(mat , dp , i+1 , j);
        //we need the minimum health so we take min(IfWeGoRight , IfWeGoDown)
        int minHealthRequired =  min(IfWeGoRight , IfWeGoDown)-mat[i][j];

        //to survive health must be at least 1
        dp[i][j] = ( minHealthRequired <= 0 ) ? 1 : minHealthRequired;      
        return dp[i][j];
    }
    
    int calculateMinimumHP(vector<vector<int>>& dungeon) {
        
        int n = dungeon.size();
        int m = dungeon[0].size();
        
        vector<vector<int>> dp(n , vector<int>(m , 1e9));
        
        return getVal(dungeon, dp);     
    }


// dp

int calculateMinimumHP(vector<vector<int> > &dungeon) {

        int n = dungeon.size();
        int m = dungeon[0].size();

        vector<vector<int> > dp(n + 1, vector<int>(m + 1, 1e9));
        // we need min health 1 to rescue 
        dp[n][m - 1] = 1; // n-1,m-1 <-- n,m-1 , move right
        dp[n - 1][m] = 1; //n-1,m-1 
        					// ^
        					// |             ,move down
        				 //   n-1,m
        
        for (int i = n - 1; i >= 0; i--) 
        {
            for (int j = m - 1; j >= 0; j--) 
            {
            	//we need the minimum health so we take min(down,right)
                int need = min(dp[i + 1][j], dp[i][j + 1]) - dungeon[i][j];                
                // store this value , to survive health must be at least 1
                dp[i][j] = need <= 0 ? 1 : need;
            }
        }
        return dp[0][0];
    }
