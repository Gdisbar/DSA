Job Sequencing Problem
=======================
// Given an array of jobs where every job has a deadline and associated 
// profit if the job is finished before the deadline. It is also given that 
// every job takes a single unit of time, so the minimum possible deadline 
// for any job is 1. Maximize the total profit if only one job can be scheduled 
// at a time.

// Examples: 

//     Input: Four Jobs with following deadlines and profits

//     JobID  Deadline  Profit

//       a           4          20   
//       b           1          10
//       c           1          40  
//       d          1          30

//     Output: Following is maximum profit sequence of jobs: c, a   

//     Input:  Five Jobs with following deadlines and profits

//     JobID   Deadline  Profit

//       a            2          100
//       b            1          19
//       c            2          27
//      d            1          25
//      e            3          15

//     Output: Following is maximum profit sequence of jobs: c, a, e



// A structure to represent a job
struct Job {

	char id; // Job Id
	int dead; // Deadline of job
	int profit; // Profit earned if job is completed before
				// deadline
};

// Custom sorting helper struct which is used for sorting
// all jobs according to profit
struct jobProfit {
	bool operator()(Job const& a, Job const& b)
	{
		return (a.profit < b.profit);
	}
};

// Returns maximum profit from jobs
void printJobScheduling(Job arr[], int n)
{
	vector<Job> result;
	sort(arr, arr + n,
		[](Job a, Job b) { return a.dead < b.dead; });

	// set a custom priority queue
	priority_queue<Job, vector<Job>, jobProfit> pq;

	for (int i = n - 1; i >= 0; i--) {
		int slot_available;
	
		// we count the slots available between two jobs
		if (i == 0) {
			slot_available = arr[i].dead;
		}
		else {
			slot_available = arr[i].dead - arr[i - 1].dead;
		}
	
		// include the profit of job(as priority),
		// deadline and job_id in maxHeap
		pq.push(arr[i]);
	
		while (slot_available > 0 && pq.size() > 0) {
		
			// get the job with the most profit
			Job job = pq.top();
			pq.pop();
		
			// reduce the slots
			slot_available--;
		
			// add it to the answer
			result.push_back(job);
		}
	}

	// sort the result based on the deadline
	sort(result.begin(), result.end(),
		[&](Job a, Job b) { return a.dead < b.dead; });

	// print the result
	for (int i = 0; i < result.size(); i++)
		cout << result[i].id << ' ';
	cout << endl;
}

// Driver's code
int main()
{
	Job arr[] = { { 'a', 2, 100 },
				{ 'b', 1, 19 },
				{ 'c', 2, 27 },
				{ 'd', 1, 25 },
				{ 'e', 3, 15 } };

	int n = sizeof(arr) / sizeof(arr[0]);
	cout << "Following is maximum profit sequence of jobs "
			"\n";

	// Function call
	printJobScheduling(arr, n);
	return 0;
}

// This code is contributed By Reetu Raj Dubey

# Python3 code for the above approach

# function to schedule the jobs take 2
# arguments array and no of jobs to schedule


def printJobScheduling(arr, t):

	# length of array
	n = len(arr)

	# Sort all jobs according to
	# decreasing order of profit
	for i in range(n):
		for j in range(n - 1 - i):
			if arr[j][2] < arr[j + 1][2]:
				arr[j], arr[j + 1] = arr[j + 1], arr[j]

	# To keep track of free time slots
	result = [False] * t

	# To store result (Sequence of jobs)
	job = ['-1'] * t

	# Iterate through all given jobs
	for i in range(len(arr)):

		# Find a free slot for this job
		# (Note that we start from the
		# last possible slot)
		for j in range(min(t - 1, arr[i][1] - 1), -1, -1):

			# Free slot found
			if result[j] is False:
				result[j] = True
				job[j] = arr[i][0]
				break

	# print the sequence
	print(job)


# Driver's Code
if __name__ == '__main__':
	arr = [['a', 2, 100], # Job Array
			['b', 1, 19],
			['c', 2, 27],
			['d', 1, 25],
			['e', 3, 15]]


	print("Following is maximum profit sequence of jobs")

	# Function Call
	printJobScheduling(arr, 3)

# This code is contributed
# by Anubhav Raj Singh
