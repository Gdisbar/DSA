Optimal Strategy for a Game
==============================
// Consider a row of n coins of values v1 . . . vn, where n is even. We play a 
// game against an opponent by alternating turns. In each turn, a player selects 
// either the first or last coin from the row, removes it from the row permanently, 
// and receives the value of the coin. Determine the maximum possible amount of 
// money we can definitely win if we move first.
// Note: The opponent is as clever as the user.

// Let us understand the problem with few examples:  

//     5, 3, 7, 10 : The user collects maximum value as 15(10 + 5)
//     8, 15, 3, 7 : The user collects maximum value as 22(7 + 15)

// Does choosing the best at each move gives an optimal solution? No. 
// In the second example, this is how the game can be finished:

//     …….User chooses 8. 
//     …….Opponent chooses 15. 
//     …….User chooses 7. 
//     …….Opponent chooses 3. 
//     Total value collected by user is 15(8 + 7)
//     …….User chooses 7. 
//     …….Opponent chooses 8. 
//     …….User chooses 15. 
//     …….Opponent chooses 3. 
//     Total value collected by user is 22(7 + 15)

// So if the user follows the second game state, the maximum value can be collected 
// although the first move is not the best. 


// If we denote the coins collected by us as a positive score of an equivalent 
// amount, whereas the coins removed by our opponent with a negative score of 
// an equivalent amount, then the problem transforms to maximizing our score 
// if we go first. 

// dp[i][j]  maximum score a player can get in the subarray [i...j]
// dp[i][j] = max(arr[i]-dp[i+1][j], arr[j]-dp[i][j-1])

// user : v[i] , opponent options : v[(i+1)...j]
// user : v[j] , opponent options : v[i...(j-1)]

// final answer will be in dp[0][n-1] but 

// This can be solved using the simple Dynamic Programming relation given above. 
// The final answer would be contained in dp[0][n-1].

// However, we still need to account for the impact of introducing the negative score. 

// say, dp[0][n-1] = VAL,sum(all the scores)= SUM, 
// total score of opponent = OPP

// Then according to the original problem we are supposed to calculate 
// abs(OPP) + VAL since our opponent does not have any negative impact on our 
// final answer according to the original problem statement. 
// This value can be easily calculated as,
// VAL + abs(OPP) = SUM - abs(OPP) => abs(OPP) = (SUM - VAL)/2
// (OPP removed by our opponent implies that we had gained OPP amount as well, 
// hence the 2*abs(OPP))=> VAL + abs(OPP) = (SUM + VAL)/2

// Function to find the maximum possible
// amount of money we can win.
long long maximumAmount(int arr[], int n)
{
    long long sum = 0;
    vector<vector<long long> > dp(n,vector<long long>(n, 0));
    for (int i = (n - 1); i >= 0; i--) {
         
        // Calculating the sum of all the elements
        sum += arr[i];
        for (int j = i; j < n; j++) {
            if (i == j) { 
            // If there is only one element then we can only get arr[i] score
                dp[i][j] = arr[i];
            }
            else {
            // Calculating the dp states using the relation
                dp[i][j] = max(arr[i] - dp[i + 1][j],arr[j] - dp[i][j - 1]);
            }
        }
    }
    // Equating and returning the final answer as per the relation
    return (sum + dp[0][n - 1]) / 2;
}