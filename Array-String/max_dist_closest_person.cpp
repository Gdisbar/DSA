849. Maximize Distance to Closest Person
==========================================
// You are given an array representing a row of seats where seats[i] = 1 
// represents a person sitting in the ith seat, and seats[i] = 0 represents that 
// the ith seat is empty (0-indexed).

// There is at least one empty seat, and at least one person sitting.

// Alex wants to sit in the seat such that the distance between him and the closest 
// person to him is maximized. 

// Return that maximum distance to the closest person.

 

// Example 1:

// Input: seats = [1,0,0,0,1,0,1]
// Output: 2
// Explanation: 
// If Alex sits in the second open seat (i.e. seats[2]), then the closest person 
// has distance 2.
// If Alex sits in any other open seat, the closest person has distance 1.
// Thus, the maximum distance to the closest person is 2.

// Example 2:

// Input: seats = [1,0,0,0]
// Output: 3
// Explanation: 
// If Alex sits in the last seat (i.e. seats[3]), the closest person is 3 seats away.
// This is the maximum distance possible, so the answer is 3.

// Example 3:

// Input: seats = [0,1]
// Output: 1

// Example 4:
// Input: seats = [0,0,1,0,1,0,0,0]
// Output: 3

def maxDistToClosest(self, seats: List[int]) -> int:
    l,mx,n=-1,0,len(seats)
    for i in range(n):
        if seats[i]==1:
            if l==-1:
                l=i
                mx=i
            else:
                mx=max(mx,(i-l)//2)
                l=i
        elif i==n-1 and seats[n-1]==0:
            mx=max(mx,n-1-l)
            l=n-1
        
    return mx

    // def maxDistToClosest(self, seats):
    //     res, last, n = 0, -1, len(seats)
    //     for i in range(n):
    //         if seats[i]:
    //             res = max(res, i if last < 0 else (i - last) // 2)
    //             last = i
    //     return max(res, n - last - 1)