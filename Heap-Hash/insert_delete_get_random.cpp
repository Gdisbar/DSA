380. Insert Delete GetRandom O(1)
===================================
// Implement the RandomizedSet class:

// RandomizedSet() Initializes the RandomizedSet object.
// bool insert(int val) Inserts an item val into the set if not present. 
// Returns true if the item was not present, false otherwise.
// bool remove(int val) Removes an item val from the set if present. 
// Returns true if the item was present, false otherwise.
// int getRandom() Returns a random element from the current set of elements 
// (it's guaranteed that at least one element exists when this method is called). 
// Each element must have the same probability of being returned.

// You must implement the functions of the class such that each function works 
// in average O(1) time complexity.

 

// Example 1:

// Input
// ["RandomizedSet", "insert", "remove", "insert", "getRandom", "remove", 
// "insert", "getRandom"]
// [[], [1], [2], [2], [], [1], [2], []]
// Output
// [null, true, false, true, 2, true, false, 2]

// Explanation
// RandomizedSet randomizedSet = new RandomizedSet();
// randomizedSet.insert(1); // Inserts 1 to the set. Returns true as 1 was 
// inserted successfully.
// randomizedSet.remove(2); // Returns false as 2 does not exist in the set.
// randomizedSet.insert(2); // Inserts 2 to the set, returns true. Set now 
// contains [1,2].
// randomizedSet.getRandom(); // getRandom() should return either 1 or 2 randomly.
// randomizedSet.remove(1); // Removes 1 from the set, returns true. Set now 
// contains [2].
// randomizedSet.insert(2); // 2 was already in the set, so return false.
// randomizedSet.getRandom(); // Since 2 is the only number in the set, 
// getRandom() will always return 2.



// removing element (order doesn't matter)
// 0 1 2 3 4 5 6 -- remove(3) --> 0 1 2 6 4 5

// Here is the pseudo-code

// If the element you are trying to remove is the last element in the vector, 
// remove it, done, ELSE,
// Read the last element of the vector and write it over the element-to-be-removed. 
// (swap is O(1))
// Now remove the last element of the vector. (C++ pop_back() in a vector is O(1))


class RandomizedSet {

private:
    vector<int> nums;
    unordered_map<int, int> m;
public:
    /** Initialize your data structure here. */
    RandomizedSet() {
        
    }
    
    /** Inserts a value to the set. Returns true if the set did not already 
        contain the specified element. **/
    bool insert(int val) {
        if (m.find(val) != m.end()) return false;
        nums.emplace_back(val);
        m[val] = nums.size() - 1;
        return true;
    }
    
    /** Removes a value from the set. Returns true if the set contained the 
        specified element. **/
    bool remove(int val) {
        if (m.find(val) == m.end()) return false;
        int last = nums.back();
        m[last] = m[val];
        nums[m[val]] = last;
        nums.pop_back();
        m.erase(val);
        return true;
    }
    
    /** Get a random element from the set. */
    int getRandom() {
        return nums[rand() % nums.size()];
    }

};

381. Insert Delete GetRandom O(1) - Duplicates allowed
========================================================
// Example 1:

// Input
// ["RandomizedCollection", "insert", "insert", "insert", "getRandom", "remove", "getRandom"]
// [[], [1], [1], [2], [], [1], []]
// Output
// [null, true, false, true, 2, true, 1]

// Explanation
// RandomizedCollection randomizedCollection = new RandomizedCollection();
// randomizedCollection.insert(1);   
// // return true since the collection does not contain 1.
// // Inserts 1 into the collection.
// randomizedCollection.insert(1);   
// // return false since the collection contains 1.
// // Inserts another 1 into the collection. Collection now contains [1,1].
// randomizedCollection.insert(2);   
// // return true since the collection does not contain 2.
// // Inserts 2 into the collection. Collection now contains [1,1,2].
// randomizedCollection.getRandom(); 
// // getRandom should:
// // - return 1 with probability 2/3, or
// // - return 2 with probability 1/3.
// randomizedCollection.remove(1);   
// // return true since the collection contains 1.
// // Removes 1 from the collection. Collection now contains [1,2].
// randomizedCollection.getRandom(); 
// // getRandom should return 1 or 2, both equally likely.

// Like in the previous problem, Insert Delete GetRandom O(1), the solution 
// is to maintain a vector with all elements to get the random number in O(1).
// With duplicates allowed, instead of just one index, we now need to store 
// indexes of all elements of the same value in our vector. The remove method 
// becomes a bit more complicated therefore, as we need to:

// Remove any index of the element being removed
// Swap the last element in the vector with the element being removed 
// (same as in the previous problem)
// Remove old and add new index for the swapped (last) element



 vector<int> v;
  unordered_map<int, unordered_set<int>> m;
  bool insert(int val) {
    v.push_back(val);
    m[val].insert(v.size() - 1);
    return m[val].size() == 1;
  }
  bool remove(int val) {
    auto it = m.find(val);
    if (it != end(m)) {
      auto free_pos = *it->second.begin();
      it->second.erase(it->second.begin());
      v[free_pos] = v.back();
      m[v.back()].insert(free_pos);
      m[v.back()].erase(v.size() - 1);
      v.pop_back();
      if (it->second.size() == 0) m.erase(it);
      return true;
    }
    return false;
  }
  int getRandom() { return v[rand() % v.size()]; }
