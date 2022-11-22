RMQSQ - Range Minimum Query
==============================

// You are given a list of N numbers and Q queries. Each query is specified 
// by two numbers i and j; the answer to each query is the minimum number between 
// the range [i, j] (inclusive).

// Note: the query ranges are specified using 0-based indexing.
// Input

// The first line contains N, the number of integers in our list (N <= 100,000). 
// The next line holds N numbers that are guaranteed to fit inside an integer. 
// Following the list is a number Q (Q <= 10,000). The next Q lines each contain 
// two numbers i and j which specify a query you must answer (0 <= i, j <= N-1).
// Output

// For each query, output the answer to that query on its own line in the order 
// the queries were made.


// Example

// Input:
// 3
// 1 4 1
// 2
// 1 1
// 1 2

// Output:
// 4
// 1



#include <iostream>
#include <cstdio>
#include <algorithm>
#define MAX_TREE_SIZE 270000
#define SIZE 100000

using namespace std;

int arr[SIZE], segmentTree[MAX_TREE_SIZE];

int prepareTreeUtil(int x, int y, int index){
	if(x == y){
		segmentTree[index] = arr[x];
		//cout << x << " " << y << " " << index << " " << segmentTree[index] << endl;
		return segmentTree[index];
	}
	int m = (x + y) / 2;
	int leftChild = 2 * index + 1;
	int rightChild = 2 * index + 2;
	int leftMin = prepareTreeUtil(x, m, leftChild);
	int rightMin = prepareTreeUtil(m + 1, y, rightChild);
	segmentTree[index] = min(leftMin, rightMin);
	//cout << x << " " << y << " " << index << " " << segmentTree[index] << endl;
	return segmentTree[index];
}

void prepareTree(int n){
	prepareTreeUtil(0, n-1, 0);
}

int queryTreeUtil(int x, int y, int left, int right, int index){
	if(x == left && y == right)
		return segmentTree[index];
	int mid = (left + right) / 2;
	int leftChild = 2 * index + 1;
	int rightChild = 2 * index + 2;
	if(y <= mid)
		return queryTreeUtil(x, y, left, mid, leftChild);
	if(x > mid)
		return queryTreeUtil(x, y, mid + 1, right, rightChild);
	int leftMin = queryTreeUtil(x, mid, left, mid, leftChild);
	int rightMin = queryTreeUtil(mid + 1, y, mid + 1, right, rightChild);
	return min(leftMin, rightMin);
}

int queryTree(int x, int y, int n){
	return queryTreeUtil(x, y, 0, n-1, 0);
}

int main(){

	int n;
	scanf("%d", &n);
	for(int i=0;i<n;i++)
		scanf("%d", &arr[i]);
	prepareTree(n);
	int q, x, y;
	scanf("%d", &q);
	while(q--){
		scanf("%d%d", &x, &y);
		if(x > y)
			swap(x, y);
		printf("%d\n", queryTree(x, y, n));
	}
	return 0;
}
