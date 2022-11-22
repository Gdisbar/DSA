K Centers Problem 
=====================
Given n cities and distances between every pair of cities, select k 
cities to place warehouses (or ATMs or Cloud Server) such that the maximum 
distance of a city to a warehouse (or ATM or Cloud Server) is minimized. 

For example consider the following four cities, 0, 1, 2, and 3, and the 
distances between them, how to place 2 ATMs among these 4 cities so that 
the maximum distance of a city to an ATM is minimized. 

			10
		1---------0
		| \5   /7 |
	8	|  \  /   |  6
		|  / \    |
		2---------3
			12

==> if we place ATM in-between 2 & 3 distance from 1 is 5 & from 0 is 6
    so the max is 6


There is no polynomial-time solution available for this problem as 
the problem is a known NP-Hard problem. There is a polynomial-time 
Greedy approximate algorithm, the greedy algorithm provides a solution 
that is never worse than twice the optimal solution. The greedy solution 
works only if the distances between cities follow Triangular Inequality 
(The distance between two points is always smaller than the sum of distances 
	through a third point). 

The 2-Approximate Greedy Algorithm: 
-----------------------------------------
Choose the first center arbitrarily. 
Choose remaining k-1 centers using the following criteria. 
    Let c1, c2, c3, … ci be the already chosen centers. Choose 
    (i+1)’th center by picking the city which is farthest from already 
    selected centers, i.e, the point p which has following value as maximum 
    Min[dist(p, c1), dist(p, c2), dist(p, c3), …. dist(p, ci)] 

Example (k = 3 in the above-shown Graph):

Let the first arbitrarily picked vertex be 0. 
The next vertex is 1 because 1 is the farthest vertex from 0. 
Remaining cities are 2 and 3. Calculate their distances from already 
selected centers (0 and 1). The greedy algorithm basically calculates the 
following values. 
    Minimum of all distanced from 2 to already considered centers 
    Min[dist(2, 0), dist(2, 1)] = Min[7, 8] = 7 
    Minimum of all distanced from 3 to already considered centers 
    Min[dist(3, 0), dist(3, 1)] = Min[6, 5] = 5 
    After computing the above values, city 2 is picked as the value 
    corresponding to 2 is maximum. 

Note that the greedy algorithm doesn’t give the best solution for k = 2 as 
this is just an approximate algorithm with a bound as twice optimal. 

Proof that the above greedy algorithm is 2 approximate. 
----------------------------------------------------------
Let OPT be the maximum distance of a city from a center in the Optimal 
solution. We need to show that the maximum distance obtained from the 
Greedy algorithm is 2*OPT. 
The proof can be done using contradiction. 

    Assume that the distance from the furthest point to all centers 
    is > 2·OPT. 
    This means that distances between all centers are also > 2·OPT. 
    We have k + 1 points with distances > 2·OPT between every pair. 
    Each point has a center of the optimal solution with 
    distance <= OPT to it. 

There exists a pair of points with the same center X in the optimal 
solution (pigeonhole principle: k optimal centers, k+1 points) 
The distance between them is at most 2·OPT (triangle inequality) which is 
a contradiction. 



 
int maxindex(int* dist, int n)
{
    int mi = 0;
    for (int i = 0; i < n; i++) {
        if (dist[i] > dist[mi])
            mi = i;
    }
    return mi;
}
 
void selectKcities(int n, int weights[4][4], int k)
{
    int* dist = new int[n];
    vector<int> centers;
    for (int i = 0; i < n; i++) {
        dist[i] = INT_MAX;
    }
 
    // index of city having the
    // maximum distance to it's
    // closest center
    int max = 0;
    for (int i = 0; i < k; i++) {
        centers.push_back(max);
        for (int j = 0; j < n; j++) {
 
            // updating the distance
            // of the cities to their
            // closest centers
            dist[j] = min(dist[j], weights[max][j]);
        }
 
        // updating the index of the
        // city with the maximum
        // distance to it's closest center
        max = maxindex(dist, n);
    }
 
    // Printing the maximum distance
    // of a city to a center
    // that is our answer
    cout << endl << dist[max] << endl;
 
    // Printing the cities that
    // were chosen to be made
    // centers
    for (int i = 0; i < centers.size(); i++) {
        cout << centers[i] << " ";
    }
    cout << endl;
}
 
// Driver Code
int main()
{
    int n = 4;
    int weights[4][4] = { { 0, 4, 8, 5 },
                          { 4, 0, 10, 7 },
                          { 8, 10, 0, 9 },
                          { 5, 7, 9, 0 } };
    int k = 2;
 
    // Function Call
    selectKcities(n, weights, k);
}