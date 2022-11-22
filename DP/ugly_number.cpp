Ugly Numbers
=============
// Ugly numbers are numbers whose only prime factors are 2, 3 or 5. 
// The sequence 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, … shows the first 11 ugly 
// numbers. By convention, 1 is included. 
// Given a number n, the task is to find n’th Ugly number.

// Examples:  

// Input  : n = 7
// Output : 8

// Input  : n = 10
// Output : 12

// Input  : n = 15
// Output : 24

// Input  : n = 150
// Output : 5832

// Method 1 : loop until count of ugly no < n 

// Method 2 (Use Dynamic Programming) 
// Here is a time efficient solution with O(n) extra space. 
// The ugly-number sequence is 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, … 

//      (1) 1×2, 2×2, 3×2, 4×2, 5×2, … 
//      (2) 1×3, 2×3, 3×3, 4×3, 5×3, … 
//      (3) 1×5, 2×5, 3×5, 4×5, 5×5, …

// initialize
//    ugly[] =  | 1 |
//    i2 =  i3 = i5 = 0;

// First iteration
//    ugly[1] = Min(ugly[i2]*2, ugly[i3]*3, ugly[i5]*5)
//             = Min(2, 3, 5)
//             = 2
//    ugly[] =  | 1 | 2 |
//    i2 = 1,  i3 = i5 = 0  (i2 got incremented ) 

// Second iteration
//     ugly[2] = Min(ugly[i2]*2, ugly[i3]*3, ugly[i5]*5)
//              = Min(4, 3, 5)
//              = 3
//     ugly[] =  | 1 | 2 | 3 |
//     i2 = 1,  i3 =  1, i5 = 0  (i3 got incremented ) 

// Third iteration
//     ugly[3] = Min(ugly[i2]*2, ugly[i3]*3, ugly[i5]*5)
//              = Min(4, 6, 5)
//              = 4
//     ugly[] =  | 1 | 2 | 3 |  4 |
//     i2 = 2,  i3 =  1, i5 = 0  (i2 got incremented )

// Fourth iteration
//     ugly[4] = Min(ugly[i2]*2, ugly[i3]*3, ugly[i5]*5)
//               = Min(6, 6, 5)
//               = 5
//     ugly[] =  | 1 | 2 | 3 |  4 | 5 |
//     i2 = 2,  i3 =  1, i5 = 1  (i5 got incremented )

// Fifth iteration
//     ugly[4] = Min(ugly[i2]*2, ugly[i3]*3, ugly[i5]*5)
//               = Min(6, 6, 10)
//               = 6
//     ugly[] =  | 1 | 2 | 3 |  4 | 5 | 6 |
//     i2 = 3,  i3 =  2, i5 = 1  (i2 and i3 got incremented )

// Will continue same way till I < 150

// Function to get the nth ugly number
unsigned getNthUglyNo(unsigned n)
{
    // To store ugly numbers
    unsigned ugly[n]; 
    unsigned i2 = 0, i3 = 0, i5 = 0;
    unsigned next_multiple_of_2 = 2;
    unsigned next_multiple_of_3 = 3;
    unsigned next_multiple_of_5 = 5;
    unsigned next_ugly_no = 1;
  
    ugly[0] = 1;
    for (int i = 1; i < n; i++) {
        next_ugly_no = min(next_multiple_of_2,
            min(next_multiple_of_3, next_multiple_of_5));
        ugly[i] = next_ugly_no;
        if (next_ugly_no == next_multiple_of_2) {
            i2 = i2 + 1;
            next_multiple_of_2 = ugly[i2] * 2;
        }
        if (next_ugly_no == next_multiple_of_3) {
            i3 = i3 + 1;
            next_multiple_of_3 = ugly[i3] * 3;
        }
        if (next_ugly_no == next_multiple_of_5) {
            i5 = i5 + 1;
            next_multiple_of_5 = ugly[i5] * 5;
        }
    }  
    
    // End of for loop (i=1; i<n; i++)
    return next_ugly_no;
}


// using set

int nthUglyNumber(int n)
{
    // Base cases...
    if (n == 1 or n == 2 or n == 3 or n == 4 or n == 5)
        return n;
  
    set<long long int> s;
    s.insert(1);
    n--;
  
    while (n) {
        auto it = s.begin();
  
        // Get the beginning element of the set
        long long int x = *it;
  
        // Deleting the ith element
        s.erase(it);
  
        // Inserting all the other options
        s.insert(x * 2);
        s.insert(x * 3);
        s.insert(x * 5);
        n--;
    }
  
    // The top of the set represents the nth ugly number
    return *s.begin();
}

// TC : nlogn , SC : n


// using binary search


int nthUglyNumber(int n)
{
  
  int pow[40] = { 1 };
  
  // stored powers of 2 from 
  // pow(2,0) to pow(2,30)
  for (int i = 1; i <= 30; ++i)
    pow[i] = pow[i - 1] * 2;
  
  // Initialized low and high
  int l = 1, r = 2147483647;
  
  int ans = -1;
  
  // Applying Binary Search
  while (l <= r) {
  
    // Found mid
    int mid = l + ((r - l) / 2); 
  
    // cnt stores total numbers of ugly
    // number less than mid
    int cnt = 0; 
  
    // Iterate from 1 to mid 
    for (long long i = 1; i <= mid; i *= 5)
  
    {
      // Possible powers of i less than mid is i
      for (long long j = 1; j * i <= mid; j *= 3)
  
      {
        // possible powers of 3 and 5 such that
        // their product is less than mid
  
        // using the power array of 2 (pow) we are
        // trying to find the max power of 2 such
        // that i*j*power of 2 is less than mid
  
        cnt += upper_bound(pow, pow + 31,mid / (i * j)) - pow;
      }
    }
  
    // If total numbers of ugly number 
    // less than equal
    // to mid is less than n we update l
    if (cnt < n)
      l = mid + 1;
  
    // If total numbers of ugly number 
    // less than equal to
    // mid is greater than n we update
    // r and ans simultaneously.
    else
      r = mid - 1, ans = mid;
  }
  
  return ans;
}

// TC : nlogn , SC : 1

Super Ugly Number (Number whose prime factors are in given set)
================================================================
// Super ugly numbers are positive numbers whose all prime factors are in the 
// given prime list. Given a number n, the task is to find the nth Super Ugly number.
// It may be assumed that a given set of primes is sorted. Also, the first 
// Super Ugly number is 1 by convention.

// Examples:  

// Input  : primes[] = [2, 5]
//          n = 5
// Output : 8
// Super Ugly numbers with given prime factors 
// are 1, 2, 4, 5, 8, ...
// Fifth Super Ugly number is 8

// Input  : primes[] = [2, 3, 5]
//          n = 50
// Output : 243

// Input : primes[] = [3, 5, 7, 11, 13]
//         n = 9
// Output: 21 


// Function to get the nth super ugly number
// primes[]       --> given list of primes f size k
// ugly           --> set which holds all super ugly numbers from 1 to n                 
// k              --> Size of prime[]
int superUgly(int n, int primes[], int k)
{
    // nextMultiple holds multiples of given primes
    vector<int> nextMultiple(primes, primes+k);
 
    // To store iterators of all primes
    int multiple_Of[k];
    memset(multiple_Of, 0, sizeof(multiple_Of));
 
    // Create a set to store super ugly numbers and
    // store first Super ugly number
    set<int> ugly;
    ugly.insert(1);
 
    // loop until there are total n Super ugly numbers
    // in set
    while (ugly.size() != n)
    {
        // Find minimum element among all current
        // multiples of given prime
        int next_ugly_no = *min_element(nextMultiple.begin(),nextMultiple.end());
 
        // insert this super ugly number in set
        ugly.insert(next_ugly_no);
 
        // loop to find current minimum is multiple
        // of which prime
        for (int j=0; j<k; j++)
        {
            if (next_ugly_no == nextMultiple[j])
            {
                // increase iterator by one for next multiple
                // of current prime
                multiple_Of[j]++;
 
                // this loop is similar to find  dp[++index[j]]
                // it -->  dp[++index[j]]
                set<int>::iterator it = ugly.begin();
                for (int i=1; i<=multiple_Of[j]; i++)
                    it++;
 
                nextMultiple[j] = primes[j] * (*it);
                break;
            }
        }
    }
 
    // n'th super ugly number
    set<int>::iterator it = ugly.end();
    it--;
    return *it;
}

// using priority queue


// Assuming a[] = {2, 3, 5}, 
// so at first iteration 1 is top so 1 is popped and 1 * 2, 1 * 3, 1 * 5 is pushed. 
// At second iteration min is 2, so it is popped and 2 * 2, 2 * 3, 2 * 5 is 

int ugly(int a[], int size, int n){
     
    // n cannot be negative hence
    // return -1 if n is 0 or -ve
    if(n <= 0)
        return -1;
  
    if(n == 1)
        return 1;
     
    // Declare a min heap priority queue
    priority_queue<int, vector<int>, greater<int>> pq;
     
    // Push all the array elements to priority queue
    for(int i = 0; i < size; i++){
        pq.push(a[i]);
    }
     
    // once count = n we return no
    int count = 1, no;
     
    while(count < n){
        // Get the minimum value from priority_queue
        no = pq.top();
        pq.pop();
         
        // If top of pq is no then don't increment count.
        // This to avoid duplicate counting of same no.
        if(no != pq.top())
        {
            count++;
         
            // Push all the multiples of no. to priority_queue
            for(int i = 0; i < size; i++){
                pq.push(no * a[i]);
            // cnt+=1;
        }
        }
    }
    // Return nth super ugly number
    return no;
}

// TC : n*size*logn , SC : n
