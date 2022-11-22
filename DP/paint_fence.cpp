514 · Paint Fence
====================
// There is a fence with n posts, each post can be painted with one of the k colors.
// You have to paint all the posts such that no more than two adjacent fence posts 
// have the same color.
// Return the total number of ways you can paint the fence.
// n and k are non-negative integers.

// Input: n=3, k=2  

// Output: 6

// Explanation:

//           post 1,   post 2, post 3

//     way1    0         0       1 

//     way2    0         1       0

//     way3    0         1       1

//     way4    1         0       0

//     way5    1         0       1

//     way6    1         1       0


// Input: n=2, k=2  

// Output: 4

// Explanation:

//           post 1,   post 2

//     way1    0         0       

//     way2    0         1            

//     way3    1         0          

//     way4    1         1       


int numWays(int n, int k)
{
    int same = 0, diff = k;
    for (int i = 2; i <= n; i++){
        int temp = same; // for same we've k choice & different k*(k-1)
        same = diff;     // same=diff(we can add any color that will make 2 same color at max) 
        diff = (diff + temp) * (k - 1); //diff=total * (k-1) , if we take total & add a color 2 color will be same at max
    }
    return same + diff;
}