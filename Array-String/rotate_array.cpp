189. Rotate Array
===================
Given an array, rotate the array to the right by k steps, where k is non-negative.

 

Example 1:

Input: nums = [1,2,3,4,5,6,7], k = 3
Output: [5,6,7,1,2,3,4]
Explanation:
rotate 1 steps to the right: [7,1,2,3,4,5,6]
rotate 2 steps to the right: [6,7,1,2,3,4,5]
rotate 3 steps to the right: [5,6,7,1,2,3,4]

Example 2:

Input: nums = [-1,-100,3,99], k = 2
Output: [3,99,-1,-100]
Explanation: 
rotate 1 steps to the right: [99,-1,-100,3]
rotate 2 steps to the right: [3,99,-1,-100]

//TLE

void rotate(vector<int>& nums, int k) {
        while(k--){
            int tmp =nums[nums.size()-1];
            for(int i = nums.size()-1;i>0;i--){
                 nums[i]=nums[i-1];
            }
            nums[0]=tmp;
        }
    }

//TC : n , SC : n
void rotate(vector<int>& nums, int k) {
        if ((nums.size() == 0) || (k <= 0)){
            return;
        }
        vector<int> cpy(nums);
        for(int i = 0;i<nums.size();i++)
            nums[(i+k)%nums.size()]=cpy[i]; //rotate right by k
    }

//TC : n , SC : 1


nums = [1,2,3,4,5,6,7] (0,4)(1,5) --> [ 5 2 3 4 1 6 7 ] (1,5)(2,6) --> [ 5 6 3 4 1 2 7 ] (2,6)(3,7) --> [ 5 6 7 4 1 2 3 ] --> next=4(7),first=3

nums = [ 5 6 7 4 1 2 3 ] (3,4)(4,1) --> [ 5 6 7 1 4 2 3 ] --> middle=5(4),first=4

nums = [ 5 6 7 1 4 2 3 ] (4,5)(4,2) --> [ 5 6 7 1 2 4 3 ] --> middle=6(5),first=5

nums = [ 5 6 7 1 2 4 3 ] (5,6)(4,3) --> [5,6,7,1,2,3,4] --> next=6(7),first=6

//when next reaches last we''ve placed all rotated elements in the beginning ,
//now next is at n-1  so we set it to n-k & first is at n-k-1 
//when first reaches middle is at n-k+1 so set it to n-k

void rotate(vector<int>& nums, int k) {
        int n = nums.size();
        if (k%n == 0) return;
        int first = 0, middle = n-k%n, last = n;
        int next = middle; //n-k
        while(first != next) {
            swap (nums[first++], nums[next++]);
            if (next == last) next = middle;  //set next from n-1 to n-k,[5,6,7] swapping
            else if (first == middle) middle = next; //set middle from n-k+1 to n-k,[4 1 2 3] swapping
        }
    }    

//TC : n , SC : 1 , but reverse operation actually takes ~ 2*TC
/***
reverse(a,l,h){
    while(l<h){
        swap(a[l],a[h]);
        l++;
        h--;
    }
}
**/
void rotate(vector<int>& nums, int k) {
        int n = nums.size();
        k = k%n;
        // Reverse the first n - k numbers.
        // Index i (0 <= i < n - k) becomes n - k - i.
        reverse(nums.begin(), nums.end() - k); //[4,3,2,1]
        // Reverse tha last k numbers.
        // Index n - k + i (0 <= i < k) becomes n - i.
        reverse(nums.end() - k, nums.end()); //[4,3,2,1]+[7,6,5] = [4,3,2,1,7,6,5]
        // Reverse all the numbers.
        // Index i (0 <= i < n - k) becomes n - (n - k - i) = i + k.
        // Index n - k + i (0 <= i < k) becomes n - (n - i) = i.
        reverse(nums.begin(), nums.end()); //[4,3,2,1,7,6,5] --> [5,6,7,1,2,3,4]
    }


