Egg Dropping
=================
// The following is a description of the instance of this famous puzzle involving 
// 2 eggs and a building with 100 floors. 

// Suppose that we wish to know which stories in a 100-storey building are safe 
// to drop eggs from, and which will cause the eggs to break on landing. 
// What strategy should be used to drop eggs such that a total number of drops 
// in the worst case is minimized and we find the required floor 

// We may make a few assumptions: 

//     An egg that survives a fall can be used again.
//     A broken egg must be discarded.
//     The effect of a fall is the same for all eggs.
//     If an egg breaks when dropped, then it would break if dropped from a higher 
//     floor.
//     If an egg survives a fall then it would survive a shorter fall.

// The problem is not actually to find the critical floor, but merely to 
// decide floors from which eggs should be dropped so that the total number of 
// trials is minimized. 

 Method-1
=============
// If we use Binary Search Method to find the floor and we start from the 50’th 
// floor, then we end up doing 50 comparisons in the worst case. The worst-case 
// happens when the required floor is 49’th floor.

Method-2
============
// Let us make our first attempt on x'th floor.

// If it breaks, we try remaining (x-1) floors one by one. 
// So in worst case, we make x trials.

// If it doesn't break, we jump (x-1) floors (Because we have
// already made one attempt and we don't want to go beyond 
// x attempts.  Therefore (x-1) attempts are available),
//     Next floor we try is floor x + (x-1)

// Similarly, if this drop does not break, next need to jump 
// up to floor x + (x-1) + (x-2), then x + (x-1) + (x-2) + (x-3)
// and so on.

// Since the last floor to be tried is 100'th floor, sum of
// series should be 100 for optimal value of x.

//  x + (x-1) + (x-2) + (x-3) + .... + 1  = 100

//  x(x+1)/2  = 100
//          x = 13.651

// Therefore, we start trying from 14'th floor. If Egg breaks on 14th floor
// we one by one try remaining 13 floors, starting from 1st floor.  If egg doesn't break
// we go to 27th floor.
// If egg breaks on 27'th floor, we try floors form 15 to 26.
// If egg doesn't break on 27'th floor, we go to 39'th floor.

// An so on...

// The optimal number of trials is 14 in the worst case. 

// k ==> Number of floors 
// n ==> Number of Eggs 
// eggDrop(n, k) ==> Minimum number of trials needed to find the critical 
// floor in worst case.
// eggDrop(n, k) = 1 + min{max(eggDrop(n – 1, x – 1), eggDrop(n, k – x)), 
// where x is in {1, 2, …, k}}
// Concept of worst case: 

// For example : 

// Let there be ‘2’ eggs and ‘2’ floors then-:
// If we try throwing from ‘1st’ floor: 
// Number of tries in worst case= 1+max(0, 1) 
// 0=>If the egg breaks from first floor then it is threshold floor 
// (best case possibility). 
// 1=>If the egg does not break from first floor we will now have ‘2’ eggs and 1 
// floor to test which will give answer as ‘1’.(worst case possibility) 
// We take the worst case possibility in account, so 1+max(0, 1)=2

// If we try throwing from ‘2nd’ floor: 
// Number of tries in worst case= 1+max(1, 0) 
// 1=>If the egg breaks from second floor then we will have 1 egg and 1 floor to 
// find threshold floor.(Worst Case) 
// 0=>If egg does not break from second floor then it is threshold floor.(Best Case) 
// We take worst case possibility for surety, so 1+max(1, 0)=2.
// The final answer is min(1st, 2nd, 3rd….., kth floor) 
// So answer here is ‘2’. 

int eggDrop(int n, int k)
{
    // If there are no floors,then no trials needed.
    // OR if there is one floor, one trial needed.
    if (k == 1 || k == 0)
        return k;
 
    // We need k trials for one egg and k floors
    if (n == 1)
        return k;
 
    int min = INT_MAX, x, res;
 
    // Consider all droppings from 1st floor to kth floor and
    // return the minimum of these values plus 1.
    for (x = 1; x <= k; x++) {
        res = max(eggDrop(n - 1, x - 1),eggDrop(n, k - x));
        if (res < min)
            min = res;
    }
 
    return min + 1;
}

// n = 2, k = 10; 
// --> Minimum number of trials in worst case with 2 eggs and 10 floors is 4

 						E(2, 4)
                           |                      
          ------------------------------------- 
          |             |           |         |   
          |             |           |         |       
      x=1/          x=2/      x=3/     x=4/ 
        /             /         ....      ....
       /             /    
 E(1, 0)  E(2, 3)     E(1, 1)  E(2, 2)
          /  /...         /  
      x=1/                 .....
        /    
     E(1, 0)  E(2, 2)
            /   
            ......

// Partial recursion tree for 2 eggs and 4 floors.

int eggDrop(int n, int k)
{
    // eggFloor[i][j] =min number of trials needed for i eggs and j floors.
    int eggFloor[n + 1][k + 1];
    int res;
    int i, j, x;
 
    // We need one trial for one floor and 0 trials for 0 floors
    for (i = 1; i <= n; i++) {
        eggFloor[i][1] = 1;
        eggFloor[i][0] = 0;
    }
 
    // We always need j trials for one egg and j floors.
    for (j = 1; j <= k; j++)
        eggFloor[1][j] = j;
 
    // Fill rest of the entries in table using
    // optimal substructure property
    for (i = 2; i <= n; i++) {
        for (j = 2; j <= k; j++) {
            eggFloor[i][j] = INT_MAX;
            for (x = 1; x <= j; x++) {
                res = 1 + max(eggFloor[i - 1][x - 1],eggFloor[i][j - x]);
                if (res < eggFloor[i][j])
                    eggFloor[i][j] = res;
            }
        }
    }
 
    // eggFloor[n][k] holds the result
    return eggFloor[n][k];
}

// Time Complexity: O(n*k^2). 
// =============================
// Where ‘n’ is the number of eggs and ‘k’ is the number of floors, 
// as we use a nested for loop ‘k^2’ times for each egg

// Auxiliary Space: O(n*k). 
// ===========================
// As a 2-D array of size ‘n*k’ is used for storing elements.

// The approach with O(n * k^2) has been discussed before, where 
// dp[n][k] = 1 + max(dp[n – 1][i – 1], dp[n][k – i]) for i in 1…k. 
// You checked all the possibilities in that approach.

// Consider the problem in a different way:

// dp[m][x] means that, given x eggs and m moves,
// what is the maximum number of floors that can be checked

// The dp equation is: dp[m][x] = 1 + dp[m - 1][x - 1] + dp[m - 1][x],
// which means we take 1 move to a floor.
// If egg breaks, then we can check dp[m - 1][x - 1] floors.
// If egg doesn't break, then we can check dp[m - 1][x] floors.

int minTrials(int n, int k)
{
   // Initialize 2D of size (k+1) * (n+1).
   vector<vector<int> > dp(k + 1, vector<int>(n + 1, 0));
   int i = 0; // Number of moves
   while (dp[i][n] < k) {
       i++;
       for (int x = 1; x <= n; x++) {
           dp[i][x] = 1 + dp[i - 1][x - 1] + dp[i - 1][x];
       }
   }
   return i;
}

// space optimized

int minTrials(int n, int k)
{
   // Initialize array of size (n+1) and m as moves.
   int dp[n + 1] = { 0 }, m;
   for (m = 0; dp[n] < k; m++) {
       for (int x = n; x > 0; x--) {
           dp[x] += 1 + dp[x - 1];
       }
   }
   return m;
}

    // Time Complexity: O(n * log k)
    // Auxiliary Space: O(n)