Maximum trains for which stoppage can be provided
=====================================================
// We are given n-platform and two main running railway track for both direction. 
// Trains which needs to stop at your station must occupy one platform for their 
// stoppage and the trains which need not to stop at your station will run away 
// through either of main track without stopping. Now, each train has three value first 
// arrival time, second departure time and third required platform number. We are 
// given m such trains you have to tell maximum number of train for which you can 
// provide stoppage at your station.

// Examples:

// Input : n = 3, m = 6 
// Train no.|  Arrival Time |Dept. Time | Platform No.
//     1    |   10:00       |  10:30    |    1
//     2    |   10:10       |  10:30    |    1
//     3    |   10:00       |  10:20    |    2
//     4    |   10:30       |  12:30    |    2
//     5    |   12:00       |  12:30    |    3
//     6    |   09:00       |  10:05    |    1
// Output : Maximum Stopped Trains = 5
// Explanation : If train no. 1 will left 
// to go without stoppage then 2 and 6 can 
// easily be accommodated on platform 1. 
// And 3 and 4 on platform 2 and 5 on platform 3.

// Input : n = 1, m = 3
// Train no.|Arrival Time|Dept. Time | Platform No.
//     1    | 10:00      |  10:30    |    1
//     2    | 11:10      |  11:30    |    1
//     3    | 12:00      |  12:20    |    1
           
// Output : Maximum Stopped Trains = 3
// Explanation : All three trains can be easily
// stopped at platform 1.

// If we start with a single platform only then we have 1 platform and some trains 
// with their arrival time and departure time and we have to maximize the number of 
// trains on that platform. This task is similar as Activity Selection Problem. So, 
// for n platforms we will simply make n-vectors and put the respective trains in those 
// vectors according to platform number. After that by applying greedy approach we 
// easily solve this problem.
// Note : We will take input in form of 4-digit integer for arrival and departure 
// time as 1030 will represent 10:30 so that we may handle the data type easily.
// Also, we will choose a 2-D array for input as arr[m][3] where arr[i][0] denotes 
// arrival time, arr[i][1] denotes departure time and arr[i][2] denotes the platform 
// for ith train.


// number of platforms and trains
#define n 2
#define m 5
  
// function to calculate maximum trains stoppage
int maxStop(int arr[][3])
{
    // declaring vector of pairs for platform
    vector<pair<int, int> > vect[n + 1]; // arrival,departure,platform
  
    // Entering values in vector of pairs as per platform number
    // make departure time first element  of pair
    for (int i = 0; i < m; i++)
        vect[arr[i][2]].push_back(make_pair(arr[i][1], arr[i][0]));
  
    // sort trains for each platform as per dept. time
    for (int i = 0; i <= n; i++)
        sort(vect[i].begin(), vect[i].end());
      
    // perform activity selection approach
    int count = 0;
    for (int i = 0; i <= n; i++) {
        if (vect[i].size() == 0) continue;
        // first train for each platform will also be selected
        int x = 0;
        count++;
        for (int j = 1; j < vect[i].size(); j++) {
            // next arrival > current departure 
            if (vect[i][j].second >=vect[i][x].first) { 
                x = j;
                count++;
            }
        }
    }
    return count;
}
  
// driver function
int main()
{
    int arr[m][3] = { 1000, 1030, 1,
                      1010, 1020, 1,
                      1025, 1040, 1,
                      1130, 1145, 2,
                      1130, 1140, 2 };
    cout << "Maximum Stopped Trains = "<< maxStop(arr);
    return 0;
}

// Similar kind of problem

//https://leetcode.com/discuss/interview-question/124552/minimum-number-of-train-stops

// There are an infinite number of train stops starting from station number 0.

// There are an infinite number of trains. The nth train stops at all of the 
// k * 2^(n - 1) stops where k is between 0 and infinity.

// When n = 1, the first train stops at stops 0, 1, 2, 3, 4, 5, 6, etc.

// When n = 2, the second train stops at stops 0, 2, 4, 6, 8, etc.

// When n = 3, the third train stops at stops 0, 4, 8, 12, etc.

// Given a start station number and end station number, return the minimum number 
// of stops between them. You can use any of the trains to get from one stop to 
// another stop.

// For example, the minimum number of stops between start = 1 and end = 4 is 3 
// because we can get from 1 to 2 to 4.





// Here is my simple recursive with DP solution. With detailed explanation.
// I believe it is O((end - start) * log (end)) time and O(end - start) space

// The idea is simple:
// 1- we need to keep track of the least stops to reach each station from start to 
// end and hence the dp array is of size [end - start + 1]

// 2- we start by calling the recursive function findMinStops with the end station

// 3- in this function we need to know all trains that can pass through this station. 
// we can know that by noticing that trains move with steps of multiples of two. 
// like train 1 moves 1 step and train 2 moves 2 steps and 3 moves 3 steps and 
// so on... . So we loop starting from 1 and keep multiplying by 2 to check all 
// trains until the step of the train exceeds our station which means that this 
// train and the following trains first stop is beyond our station and no need 
// to check them.

// 4- for each of the previous trains we need to know if it actually stops at our 
// station. we can do that by checking that the station is divisible by the train 
// step station % trainStep == 0.

// 5- if the train actually passes through our station. then we know it can reach 
// our station in one stop from its previous station. And its previous station 
// is (station - trainStep). and hence the recursive call 
// findMinStops(station - trainStepSize) + 1 gives us the cost to reach our station 
// using that train from the previous station

// 6- now if we keep track of the minimum of all those costs of trains able to 
// reach our station and select the minimum we will end up with the required solution.

// 7- since the problem can have overlapping problems we use a dp array to store 
// the min stops to reach a station once it is found once.

    private static int[] dp;
    private static int startStation;
    
    public static int findMinTrainStops(int start, int end){
        if(end <= start) return 0;
        
        startStation = start;
        
        dp = new int[end - start + 1];
        
        return findMinStops(end);
    }
    
    private static int findMinStops(int station){
        if(station == startStation) return 1;
        if(station < startStation || station < 0) return 2000000000;
        // we're away from end by station - startStation from current station
        if(dp[station - startStation] != 0) return dp[station - startStation];
        
        int minStops = Integer.MAX_VALUE;
        int trainStepSize = 1;
        // this train and the following trains first stop is beyond our station 
        // and no need to check them , so check until end
        for(;trainStepSize <= station;trainStepSize *= 2){ 
            // need to know if train stops at our station using previous station
            if(station % trainStepSize == 0){
                // cost to reach current from previous station is +1
                minStops = Math.min(minStops, findMinStops(station - trainStepSize) + 1);
            }
        }
        
        dp[station - startStation] = minStops;
        return minStops;
    }



// Minimum Platforms 
// ====================
// https://practice.geeksforgeeks.org/problems/minimum-platforms-1587115620/1