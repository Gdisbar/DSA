403. Frog Jump
===============
// A frog is crossing a river. The river is divided into some number of units, 
// and at each unit, there may or may not exist a stone. The frog can jump on a 
// stone, but it must not jump into the water.

// Given a list of stones' positions (in units) in sorted ascending order, 
// determine if the frog can cross the river by landing on the last stone. 
// Initially, the frog is on the first stone and assumes the first jump must 
// be 1 unit.

// If the frog's last jump was k units, its next jump must be either k - 1, k, 
// or k + 1 units. The frog can only jump in the forward direction.

 

// Example 1:

// Input: stones = [0,1,3,5,6,8,12,17]
// Output: true
// Explanation: The frog can jump to the last stone by jumping 1 unit to the 
// 2nd stone, then 2 units to the 3rd stone, then 2 units to the 4th stone, 
// then 3 units to the 6th stone, 4 units to the 7th stone, and 5 units to the 
// 8th stone.

// Example 2:

// Input: stones = [0,1,2,3,4,8,9,11]
// Output: false
// Explanation: There is no way to jump to the last stone as the gap between the 
// 5th and 6th stone (4 & 8) is too large.


// Search for the last stone in a depth-first way, prune those exceeding 
// the [k-1,k+1] range.  

bool canCross(vector<int>& stones, int pos = 0, int k = 0) {
    for (int i = pos + 1; i < stones.size(); i++) {
        int gap = stones[i] - stones[pos];
        if (gap < k - 1) continue;
        if (gap > k + 1) return false;
        if (canCross(stones, i, gap)) return true;
    }
    return pos == stones.size() - 1;
}



unordered_map<int, bool> dp;

bool canCross(vector<int>& stones, int pos = 0, int k = 0) {
	//left shift k by 11 ,combine pos and k into a key for the hashtable
    int key = pos | k << 11; 

    if (dp.count(key) > 0)
        return dp[key];

    for (int i = pos + 1; i < stones.size(); i++) {
        int gap = stones[i] - stones[pos];
        if (gap < k - 1)
            continue;
        if (gap > k + 1)
            return dp[key] = false;
        if (canCross(stones, i, gap))
            return dp[key] = true;
    }

    return dp[key] = (pos == stones.size() - 1);
}

// The number of stones is less than 1100 so pos will always be less 
// than 2^11 (2048).
// Stone positions could be theoretically up to 2^31 but k is practically 
// not possible to be that big for the parameter as the steps must start from 
// 0 and 1 and at the 1100th step the greatest valid k would be 1100. So 
// combining pos and k is safe here.