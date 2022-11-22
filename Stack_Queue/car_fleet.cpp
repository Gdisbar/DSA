853. Car Fleet
=================
// There are n cars going to the same destination along a one-lane road. 
// The destination is target miles away.

// You are given two integer array position and speed, both of length n, 
// where position[i] is the position of the ith car and speed[i] is the speed 
// of the ith car (in miles per hour).

// A car can never pass another car ahead of it, but it can catch up to it and 
// drive bumper to bumper at the same speed. The faster car will slow down to 
// match the slower car''s speed. The distance between these two cars is 
// ignored (i.e., they are assumed to have the same position).

// A car fleet is some non-empty set of cars driving at the same position and 
// same speed. Note that a single car is also a car fleet.

// If a car catches up to a car fleet right at the destination point, it will 
// still be considered as one car fleet.

// Return the number of car fleets that will arrive at the destination.

 

// Example 1:

// Input: target = 12, position = [10,8,0,5,3], speed = [2,4,1,1,3]
// Output: 3
// Explanation:
// The cars starting at 10 (speed 2) and 8 (speed 4) become a fleet, meeting each 
// other at 12.
// The car starting at 0 does not catch up to any other car, so it is a fleet by 
// itself.
// The cars starting at 5 (speed 1) and 3 (speed 3) become a fleet, meeting each 
// other at 6. The fleet moves at speed 1 until it reaches target.
// Note that no other cars meet these fleets before the destination, so the 
// answer is 3.

// Example 2:

// Input: target = 10, position = [3], speed = [3]
// Output: 1
// Explanation: There is only one car, hence there is only one fleet.

// Example 3:

// Input: target = 100, position = [0,2,4], speed = [4,2,1]
// Output: 1
// Explanation:
// The cars starting at 0 (speed 4) and 2 (speed 2) become a fleet, meeting each 
// other at 4. The fleet moves at speed 2.
// Then, the fleet (speed 2) and the car starting at 4 (speed 1) become one fleet, 
// meeting each other at 6. The fleet moves at speed 1 until it reaches target.


// Solution: O(NlogN +N) Time, Sort + Mono Stack

// Let''s look at some examples first.

// Ex:
// Target: 12
// Pos:  [10,8,0,5,3]
// Speed:[ 2,4,1,1,3]

// First we sort the cars by there position.

// Ex:
// Target: 12
// Pos:  [0,3,5,8,10]
// Speed:[1,3,1,4, 2]

// Then we calculate the time = (target-pos)/speed it need to reach target

// Ex:
// Target: 12
// Pos:  [0, 3, 5, 8,10]
// Speed:[1, 3, 1, 4, 2]
// Time: [12,3, 7, 1, 1]

// How to calculate the time?
// Easy, (target-pos)/time

// Alright, I think the previous procedure is sort of intuitive if 
// we want to observe the pattern of the problem.

// Now let''s observe the pattern of time.

// For car 0, it is really slow, it takes 12s to reach target. 
// Thus it is itself a fleet.
// For car 3, it is fast, but it will be blocked by car 5.
// For car 8, it is fast, but it will be blocked by car 10.

// When does the fleet occur?

//     When one car is blocked by the next car.
//     Car 0, 5, 10 are the ones who block the previous.

// And it is actually a monotonic decreasing stack! Why?

// From the perspective of numbers:
//     Let's pick up 0, 5, 10 and there time needed: [12,7,1]
//     12,7,1 is monotonic decreasing
// From the perspective of reasoning:
//     If a car is slower then previous, all the previous faster car will bump into 
//     this car and become a fleet.
//     Ex:
//         Time [1,2,3,5] ==> [5]
//             5 is too slow, that 1,2,3 bump into 5
//         Time [1,2,3,5,3,4] ==> [5,4]
//             5 is too slow, and 4 is slow too.
//             123 bump into 5, 3 bump into 4.
//             4 is still faster than 5. Thus they won't bump together.
//         Time [1,2,3,5,3,4,8] ==> [8]
//             8 is really too slow all the previous bump into it.

// Finally, the answer will be the size of the stack.

// Since the stack is recording the cars that is blocking others. Which causes a fleet.

class Car{
public:
    Car(int pos, int speed){
        this->pos = pos;
        this->speed= speed;
    }
    int pos;
    int speed;
};

class Solution {
public:
    int carFleet(int target, vector<int>& position, vector<int>& speed) {
        vector<Car> cars;
        int N = position.size();
        for(int i = 0; i<N; i++){
            cars.emplace_back(position.at(i), speed.at(i));
        }
        
        sort(cars.begin(), cars.end(), [](const Car& a, const Car& b){
            return a.pos<b.pos;
        });
        
        stack<float> mono;
        for(int i = 0; i<N; i++){
            float time = (target-cars.at(i).pos)/(float)cars.at(i).speed;
            while(!mono.empty() && time >= mono.top()){
                mono.pop();
            }
            mono.push(time);
        }
        return mono.size();
    }
};

// Time Complexity:

//     O(NlogN +N)
//     NlogN for sorting
//     N for iterate through all cars to form a mono stack.


int carFleet(int target, vector<int>& pos, vector<int>& speed) {
        map<int, double> m;
        for (int i = 0; i < pos.size(); i++) 
            m[-pos[i]] = (double)(target - pos[i]) / speed[i];
        int res = 0; double cur = 0;
        for (auto it : m) 
            if (it.second > cur) 
                cur = it.second, res++;
        return res;
    }



    def carFleet(self, target, pos, speed):
        time = [float(target - p) / s for p, s in sorted(zip(pos, speed))]
        res = cur = 0
        for t in time[::-1]:
            if t > cur:
                res += 1
                cur = t
        return res
