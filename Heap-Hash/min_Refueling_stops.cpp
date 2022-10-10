871. Minimum Number of Refueling Stops
=======================================
// A car travels from a starting position to a destination which is target miles 
// east of the starting position.

// There are gas stations along the way. The gas stations are represented as 
// an array stations where stations[i] = [positioni, fueli] indicates that 
// the ith gas station is positioni miles east of the starting position and has 
// fueli liters of gas.

// The car starts with an infinite tank of gas, which initially has startFuel 
// liters of fuel in it. It uses one liter of gas per one mile that it drives. 
// When the car reaches a gas station, it may stop and refuel, transferring all 
// the gas from the station into the car.

// Return the minimum number of refueling stops the car must make in order to 
// reach its destination. If it cannot reach the destination, return -1.

// Note that if the car reaches a gas station with 0 fuel left, the car can still 
// refuel there. If the car reaches the destination with 0 fuel left, it is 
// still considered to have arrived.

 

// Example 1:

// Input: target = 1, startFuel = 1, stations = []
// Output: 0
// Explanation: We can reach the target without refueling.

// Example 2:

// Input: target = 100, startFuel = 1, stations = [[10,100]]
// Output: -1
// Explanation: We can not reach the target (or even the first gas station).

// Example 3:

// Input: target = 100, startFuel = 10, stations = [[10,60],[20,30],[30,30],
//      [60,40]]
// Output: 2
// Explanation: We start with 10 liters of fuel.
// We drive to position 10, expending 10 liters of fuel.  We refuel from 0 liters 
// to 60 liters of gas.
// Then, we drive from position 10 to position 60 (expending 50 liters of fuel),
// and refuel from 10 liters to 50 liters of gas.  We then drive to and reach the 
// target.We made 2 refueling stops along the way, so we return 2.



//  TC : O(Nlog⁡N), where N is the length of stations.

//  SC : O(N), the space used by pq. 

// BFS template

int minRefuelStops(int target, int startFuel, vector<vector<int>>& stations) {
        int stationIndex = 0, fuelStock = 0;
        int n = (int) stations.size();
        std::priority_queue<int> pq;
        pq.push(startFuel);
        for (int stationCount = 0; !pq.empty(); stationCount++) {
            int current = pq.top();
            pq.pop();
            fuelStock += current;
            if (fuelStock >= target) {
                return stationCount;
            }
            while (stationIndex < n && stations[stationIndex][0] <= fuelStock) {
                pq.push(stations[stationIndex][1]);
                stationIndex++;
            }
        }
        return -1;
}


// DP : n*n

// dp[t] means the furthest distance that we can get with t times of refueling.

// So for every station s[i],
// if the current distance dp[t] >= s[i][0],i.e we can reach this station, 
//     then we can refuel: dp[t + 1] = max(dp[t + 1], dp[t] + s[i][1])

// In the end, we''ll return the first t with dp[t] >= target,
// otherwise we''ll return -1.


int minRefuelStops(int target, int startFuel, vector<vector<int>> s) {
    long dp[501] = {startFuel};
    for (int i = 0; i < s.size(); ++i)
        for (int t = i; t >= 0 && dp[t] >= s[i][0]; --t)
            dp[t + 1] = max(dp[t + 1], dp[t] + s[i][1]);
    for (int t = 0; t <= s.size(); ++t)
        if (dp[t] >= target) return t;
    return -1;
}
