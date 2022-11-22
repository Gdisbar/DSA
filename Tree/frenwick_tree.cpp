FENTREE - Fenwick Trees
============================
// Mr. Fenwick has an array a with many integers, and his children love to do 
// operations on the
// array with their father. The operations can be a query or an update.


// For each query the children say two indices l and r , and their father 
// answers back with the sum
// of the elements from indices l to r (both included).


// When there is an update, the children say an index i and a value x , and 
// Fenwick will add x to
// ai (so the new value of ai  is ai + x ).


// Because indexing the array from zero is too obscure for children, all 
// indices start from 1.
// Fenwick is now too busy to play games, so he needs your help with a program 
// that plays with his
// children for him, and he gave you an input/output specification.

// Input

// The first line of the input contains N (1 ≤ N ≤ 106 ) . The second line 
// contains N integers
// ai (− 109 ≤ ai ≤ 109 ) , the initial values of the array. The third line 
// contains Q (1 ≤ Q ≤ 3 × 105 ) ,
// the number of operations that will be made. Each of the next Q lines 
// contains an operation.
// Query operations are of the form “q l r ” ( 1 ≤ l ≤ r ≤ N ) , while 
// update operations are of the form
// “u i x ” ( 1 ≤ i ≤ N , − 109 ≤ x ≤ 109 ) .
// Output

// You have to print the answer for every query in a different
// line, in the same order of the input.
// Example

// Input:
// 10
// 3 2 4 0 42 33 -1 -2 4 4
// 6
// q 3 5
// q 1 10
// u 5 -2
// q 3 5
// u 6 7
// q 4 7

// Output:
// 46
// 89
// 44
// 79

#include <iostream>
using namespace std;

long long BIT[1000000+1] = { 0 };
long long A[1000000+1];
int N;

long long getSum(int idx)
{
	long long sum = 0;
	idx ++;
	while(idx > 0)
	{
		sum += BIT[idx];
		idx -= idx & (-idx);
	}
	return sum;
}

void update(int idx, int v)
{
	idx ++;
	while(idx <= N)
	{
		BIT[idx] += v;
		idx += idx & (-idx);
	}
}

int main() {
	// your code goes here
	cin >> N;
	for(int i = 0; i < N; i ++)
	{
		cin >> A[i];
	}
	for(int i = 1; i <= N; i ++)
	{
		BIT[i] = 0;
	}
	for(int i = 0; i < N; i ++)
	{
		update(i,A[i]);
	}
	int Q;
	cin >> Q;
	while(Q--)
	{
		char op;
		cin >> op;
		int u, v;
		cin >> u >> v;
		if(op=='q')
		{
			cout << getSum(v-1) - getSum(u-2) << endl;
		}
		else
		{
			update(u-1,v);
		}
	}
	return 0;
}
Footer
