Consecutive 1''s not allowed
===============================
// Given a positive integer N, count all possible distinct binary strings of 
// length N such that there are no consecutive 1’s. Output your answer modulo 10^9 + 7.

// Example 1:

// Input:
// N = 3
// Output: 5
// Explanation: 5 strings are (000,
// 001, 010, 100, 101).

// Example 2:

// Input:
// N = 2
// Output: 3
// Explanation: 3 strings are
// (00,01,10).


// We'll use recursion first and if the last digit was '0' we have 2 options 
// -> append '0' to it or append '1' to it
// else if the last digit is '1' we can only append '0' to it since consequtive 1's 
// 	are not allowed

int solve(int prev=0,int n)
	{
	   if(n==0) return 0;
	   if(n==1){
	       if(prev==0) return 2;
	       if(prev==1) return 1;
	   }
	   if(prev==0) return solve(0,n-1)+solve(1,n-1);
	   if(prev==1) return solve(0,n-1);
	}

// print the string

void countStrings(int n, string out, int last_digit)
{
    // if the number becomes n–digit, print it
    if (n == 0)
    {
        cout << out << endl;
        return;
    }
 
    // append 0 to the result and recur with one less digit
    countStrings(n - 1, out + "0", 0);
 
    // append 1 to the result and recur with one less digit
    // only if the last digit is 0
    if (last_digit == 0) {
        countStrings(n - 1, out + "1", 1);
    }
}


// Let a[i] be the number of binary strings of length i which do not contain any two 
// consecutive 1’s and which end in 0.
// Similarly, let b[i] be the number of such strings which end in 1. We can append 
// either 0 or 1 to a string ending in 0, but we can only append 0 to a string ending 
// in 1.

// This yields the recurrence relation:

// a[i] = a[i - 1] + b[i - 1]
// b[i] = a[i - 1]


//     a[i] means no of substrings of length i ending with '0'
//     b[i] means no of substrings of length i ending with '1'

int countStrings(int n)
{
    int a[n+1], b[n+1];
    a[0] = b[0] = 0;
	a[1]=b[1]=1;
    for (int i = 2; i < n; i++)
    {
        a[i] = a[i-1] + b[i-1];
        b[i] = a[i-1];
    }
    return a[n-1] + b[n-1];
}

// n = 1, count = 2  = fib(3)
// n = 2, count = 3  = fib(4)
// n = 3, count = 5  = fib(5)
// n = 4, count = 8  = fib(6)
// n = 5, count = 13 = fib(7)

// This probelm just reduced to finding the nth fibonacci term i.e fib[n+1]



Count possible ways to construct buildings
=============================================
// Given an input number of sections and each section has 2 plots on either sides of 
// the road. Find all possible ways to construct buildings 
// in the plots such that there is a space between any 2 buildings.


// N = 1
// Output = 4
// Place a building on one side.
// Place a building on other side
// Do not place any building.
// Place a building on both sides.

// N = 3 
// Output = 25
// 3 sections, which means possible ways for one side are 
// BSS, BSB, SSS, SBS, SSB where B represents a building 
// and S represents an empty space
// Total possible ways are 25, because a way to place on 
// one side can correspond to any of 5 ways on other side.

// N = 4 
// Output = 64



// We can simplify the problem to first calculate for one side only. If we know 
// the result for one side, we can always do the square of the result and get the 
// result for two sides.

// A new building can be placed on a section if section just before it has space. 
// A space can be placed anywhere (it doesn’t matter whether the previous section 
// has a building or not).

// Base case N==1 ,res=4 ,  2 for one side and 4 for two sides

// Let countB(i) be count of possible ways with i sections
//               ending with a building.
//     countS(i) be count of possible ways with i sections
//               ending with a space.

// initally : countB = 1, countS = 1 , countB(N-1) & countS(N-1) use variable
// for 2 to N
// A space can be added after a building or after a space.
countS(N) = countB(N-1) + countS(N-1)
// A building can only be added after a space.
countB[N] = countS(N-1)
// Result for one side is sum of the above two counts.
result1(N) = countS(N) + countB(N)
// Result for two sides is square of result1(N)
result2(N) = result1(N) * result1(N) 



1 => Represents building has been made on the ith plot
0 => Represents building has not been made on the ith plot


Example :

N = 3 


000 (No building) (Possible)
001 (Building on 3rd plot) (Possible)
010 (Building on 2nd plot) (Possible)
011 (Building on 2nd and 3rd plot) (Not Possible as there are consecutive buildings)
100 (Building on 1st plot) (Possible)
101 (Building on 1st and 3rd plot) (Possible)
110 (Building on 1st and 2nd plot) (Not Possible as there are consecutive buildings)
111 (Building on 1st, 2nd, 3rd plot) (Not Possible as there are consecutive buildings)

Total Possible Ways = 5 
Answer = Total Possible Ways * Total Possible Ways = 25  



 result(N) = fib(N+2)*fib(N+2)
  
  fib(N) is a function that returns N''th Fibonacci Number. 

 O(LogN) implementation of Fibonacci Numbers to find the number of ways 
 in O(logN) time
