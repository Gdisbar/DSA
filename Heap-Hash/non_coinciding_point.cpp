Pairs of Non Coinciding Points
=====================================
// In a given cartesian plane, there are N points. We need to find the 
// Number of Pairs of  points(A, B) such that

//  Point A and Point B do not coincide.
//  Manhattan Distance and the Euclidean Distance between the points should be equal.

// Note: Pair of 2 points(A,B) is considered different from  Pair of 2 points(B ,A).
// Manhattan Distance = |x2-x1|+|y2-y1|
// Euclidean Distance   = ((x2-x1)^2 + (y2-y1)^2)^0.5, where points are (x1,y1) 
// and (x2,y2).

 

// Example 1:

// Input:
// N = 2
// X = {1, 7}
// Y = {1, 5}
// Output:
// 0
// Explanation:
// None of the pairs of points have
// equal Manhatten and Euclidean distance.

// Example 2:

// Input:
// N = 3
// X = {1, 2, 1}
// Y = {2, 3, 3}
// Output:
// 2
// Explanation:
// The pairs {(1,2), (1,3)}, and {(1,3), (2,3)}
// have equal Manhatten and Euclidean distance.

int numOfPairs(int X[], int Y[], int N) {
    int c1=0,c2=0,c3=0;
    map<int,int> mp1,mp2;
    map<pair<int,int>,int> mp3;
    for(int i=0;i<N;++i){
        mp1[X[i]]++;
        mp2[Y[i]]++;
        mp3[{X[i],Y[i]}]++;
        
        c1+=mp1[X[i]];
        c2+=mp2[Y[i]];
        c3+=mp3[{X[i],Y[i]}];
    }
    return c1+c2-2*c3;
}