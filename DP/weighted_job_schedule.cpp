Weighted Job Scheduling
========================
// Given N jobs where every job is represented by following three elements of it.
// 1. Start Time 
// 2. Finish Time 
// 3. Profit or Value Associated
// Find the maximum profit subset of jobs such that no two jobs in the subset overlap.

// Input:  
// Number of Jobs n = 4
// Job Details {Start Time, Finish Time, Profit}
// Job 1: {1, 2, 50}
// Job 2: {3, 5, 20}
// Job 3: {6, 19, 100}
// Job 4: {2, 100, 200}

// Output:  
// Job 1: {1, 2, 50}
// Job 4: {2, 100, 200}

// Explanation: We can get the maximum profit by 
// scheduling jobs 1 and 4 and maximum profit is 250.

// DP - n*n

// A job has start time, finish time and profit.
struct Job
{
    int start, finish, profit;
};
 
// Utility function to calculate sum of all vector
// elements
int findSum(vector<Job> arr)
{
    int sum = 0;
    for (int i = 0; i < arr.size(); i++)
        sum +=  arr[i].profit;
    return sum;
}
 
// comparator function for sort function
int compare(Job x, Job y)
{
    return x.start < y.start;
}
 
// The main function that finds the maximum possible
// profit from given array of jobs
void findMaxProfit(vector<Job> &arr)
{
    // Sort arr[] by start time.
    sort(arr.begin(), arr.end(), compare);
 
    // L[i] stores Weighted Job Scheduling of
    // job[0..i] that ends with job[i]
    vector<vector<Job>> L(arr.size());
 
    // L[0] is equal to arr[0]
    L[0].push_back(arr[0]);
 
    // start from index 1
    for (int i = 1; i < arr.size(); i++)
    {
        // for every j less than i
        for (int j = 0; j < i; j++)
        {
            // L[i] = {MaxSum(L[j])} + arr[i] where j < i
            // and arr[j].finish <= arr[i].start
            if ((arr[j].finish <= arr[i].start) && (findSum(L[j]) > findSum(L[i])))
                L[i] = L[j];
        }
        L[i].push_back(arr[i]);
    }
 
    vector<Job> maxChain;
 
    // find one with max profit
    for (int i = 0; i < L.size(); i++)
        if (findSum(L[i]) > findSum(maxChain))
            maxChain = L[i];
 
    for (int i = 0; i < maxChain.size(); i++)
        cout << "(" <<  maxChain[i].start << ", " <<
             maxChain[i].finish << ", "
             <<  maxChain[i].profit << ") ";
}

Job a[] = { {3, 10, 20}, {1, 2, 50}, {6, 19, 100},{2, 100, 200} };
int n = sizeof(a) / sizeof(a[0]);
vector<Job> arr(a, a + n);

// The main function that finds the maximum possible 
// profit from given array of jobs
void findMaxProfit(vector<Job> arr)
{
	// L[i] stores stores Weighted Job Scheduling of
	// job[0..i] that ends with job[i]
    vector<vector<Job>> L(arr.size());
	vector<int> sum(arr.size(), 0);
	
	// L[0] is equal to arr[0]
    L[0].push_back(arr[0]);
    sum[0] = arr[0].profit;

	// start from index 1
    for(int i = 1; i < arr.size(); i++)
	{
		// for every j less than i
		for(int j = 0; j < i; j++)
		{
			// L[i] = {MaxSum(L[j])} + arr[i] where j < i 
			// and arr[j].finish <= arr[i].start
			if((arr[j].finish <= arr[i].start) && (sum[j] > sum[i]))
			{
				L[i] = L[j];
				sum[i] = arr[j].profit;
			}
		}
		L[i].push_back(arr[i]);
		sum[i] += arr[i].profit;
    }

    vector<Job> maxChain;
    int max = 0;
	// find one with max profit
    for(int i = 0; i < L.size(); i++)
		if(sum[i] > max)
		{
			max = sum[i];
			maxChain = L[i];
		}

    for(int i = 0; i < maxChain.size(); i++)
		cout << "(" <<  maxChain[i].start << ", " <<
            maxChain[i].finish << ", " <<  maxChain[i].profit << ") ";
}

