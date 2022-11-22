1235. Maximum Profit in Job Scheduling
============================================
// We have n jobs, where every job is scheduled to be done from 
// startTime[i] to endTime[i], obtaining a profit of profit[i].

// You''re given the startTime, endTime and profit arrays, return the maximum 
// profit you can take such that there are no two jobs in the subset with 
// overlapping time range.

// If you choose a job that ends at time X you will be able to start another job 
// that starts at time X.

 

// Example 1:

// Input: startTime = [1,2,3,3], endTime = [3,4,5,6], profit = [50,10,40,70]
// Output: 120
// Explanation: The subset chosen is the first and fourth job. 
// Time range [1-3]+[3-6] , we get profit of 120 = 50 + 70.



#define f first
#define s second

class Solution {
public:
    static bool comp(pair<pair<int,int>,int> p1, pair<pair<int,int>,int> p2){
        return p1.f.s<p2.f.s;
    }
    int lastNOC(vector<pair<pair<int,int>,int>> &a, int k){
        for(int i=k-1;i>=0;i--){
            if(a[i].f.s<=a[k].f.f) return i; //end[i] < start[k], we can do this job
        }
        return -1;
    }
    int jobScheduling(vector<int>& start,vector<int>& end,vector<int>& profit){
        if(start.size()==0) return 0;
        int n=start.size();
        vector<pair<pair<int,int>,int>> a(n); //start,end,profit
        for(int i=0;i<n;i++) a[i]={{start[i],end[i]},profit[i]};
        // sort by endtime increasing order
        sort(a.begin(),a.end(),[](pair<pair<int,int>,int> p1, pair<pair<int,int>,int> p2){
          return p1.f.s<p2.f.s;
        });
        vector<int> dp(n,0); // store profit
        dp[0]=a[0].s;
        for(int i=1;i<n;i++){
            int incP=a[i].s;
            int j=lastNOC(a,i);
            if(j!=-1) incP+=dp[j];
            dp[i]=max(incP,dp[i-1]);
        }
        return dp[n-1];
    }
};

Job Sequencing Problem
=============================
// Given a set of N jobs where each job[i] has a deadline and profit associated 
// with it.

// Each job takes 1 unit of time to complete and only one job can be scheduled at a 
// time. We earn the profit associated with job if and only if the job is completed 
// by its deadline.

// Find the number of jobs done and the maximum profit.

// Note: Jobs will be given in the form (Jobid, Deadline, Profit) associated with 
// that Job.


// Example 1:

// Input:
// N = 4
// Jobs = {(1,4,20),(2,1,10),(3,1,40),(4,1,30)}
// Output:
// 2 60
// Explanation:
// Job1 and Job3 can be done with
// maximum profit of 60 (20+40).

// Example 2:

// Input:
// N = 5
// Jobs = {(1,2,100),(2,1,19),(3,2,27),
//         (4,1,25),(5,1,15)}
// Output:
// 2 127
// Explanation:
// 2 jobs can be done with
// maximum profit of 127 (100+27).

// TC : n^2 , SC : 2*n

// A structure to represent various attributes of a Job
struct Job
{
    // Each job has id, deadline and profit
    char id;
    int deadLine, profit;
};
vector<int> JobScheduling(Job arr[], int n) { 
        sort(arr,arr+n,[](Job a,Job b){return a.profit>b.profit;});
        vector<int> res(n);
        vector<bool> slot(n,false);
        
        int prof=0,job=0;
        for(int i=0;i<n;++i){
            for(int j=min(arr[i].dead,n)-1;j>=0;--j){
                if(!slot[j]){
                    res[j]=i;
                    //job++;
                    //prof+=arr[j].profit;
                    slot[j]=true;
                    break;
                }
            }
        }
        for(int i=0;i<n;++i){
            if(slot[i]){
                job++;
                prof+=arr[res[i]].profit;
            }
        }
        return vector<int> {job,prof};
    } 

int main()
{
    Job arr[] =  { { 'a', 2, 100 }, { 'b', 1, 19 },
                   { 'c', 2, 27 },  { 'd', 1, 25 },
                   { 'e', 3, 15 } };
    int n = sizeof(arr) / sizeof(arr[0]);
    cout << "Following jobs need to be "<< "executed for maximum profit\n";
    printJobScheduling(arr, n);
    return 0;
}

// TC : nlog(n) , SC : n

// sort jobs based on deadline
// for priority_queue compare parameter is profit
// slots_available btw 2 jobs = a[i].dead - a[i-1].dead & each step push 
// the job a[i] in priority_queue
// if slot_availabe > 0 & pq.size() > 0 --> we assign a slot to a & push it in res
// sort res based on deadline & then calculate profit,job count

// class Solution 
// {
//     private:
//      // Custom sorting helper struct which is used for sorting
//     // all jobs according to profit (ascending order)
//      struct jobProfit {
//         bool operator()(Job const& a, Job const& b){
//             return (a.profit < b.profit);
//         }
//       };
//     public:
//     //Function to find the maximum profit and the number of jobs done.
//     vector<int> JobScheduling(Job arr[], int n) 
//     { 
//         sort(arr, arr + n,[](Job a,Job b){return a.dead < b.dead;});
//         vector<Job> res;

        
//         int prof=0,job=0;
//         priority_queue<Job,vector<Job>,jobProfit> pq;
//         for (int i=n - 1;i >=0;--i) {
//             int slot_available;
//             // we count the slots available between two jobs
//             if(i == 0) {
//                 slot_available = arr[i].dead;
//             }
//             else{
//                 slot_available = arr[i].dead - arr[i - 1].dead;
//             }
//             // include the profit of job(as priority),
//             // deadline and job_id in maxHeap
//             pq.push(arr[i]);
//             while(slot_available>0&&pq.size()>0) {
//                 Job job = pq.top();
//                 pq.pop();
//                 // get the job with the most profit
//                 slot_available--;
//                 // add it to the answer
//                 res.push_back(job);
//             }
//         }
//         sort(res.begin(),res.end(),[&](Job a, Job b){return a.dead<b.dead;});
//         for(int i=0;i<res.size();++i){
//                 job++;
//                 prof+=res[i].profit;
//         }
//         return vector<int> {job,prof};
//     } 

// Same Complexity as above

// A Simple Disjoint Set Data Structure
struct DisjointSet
{
    int *parent;
 
    // Constructor
    DisjointSet(int n)
    {
        parent = new int[n+1];
 
        // Every node is a parent of itself
        for (int i = 0; i <= n; i++)
            parent[i] = i;
    }
 
    // Path Compression
    int find(int s)
    {
        /* Make the parent of the nodes in the path
           from u--> parent[u] point to parent[u] */
        if (s == parent[s])
            return s;
        return parent[s] = find(parent[s]);
    }
 
    // Makes u as parent of v.
    void merge(int u, int v)
    {
        //update the greatest available
        //free slot to u
        parent[v] = u;
    }
};
 
// Used to sort in descending order on the basis
// of profit for each job
bool cmp(Job a, Job b)
{
    return (a.profit > b.profit);
}
 
// Functions returns the maximum deadline from the set
// of jobs
int findMaxDeadline(struct Job arr[], int n)
{
    int ans = INT_MIN;
    for (int i = 0; i < n; i++)
        ans = max(ans, arr[i].deadLine);
    return ans;
}
 
int printJobScheduling(Job arr[], int n)
{
    // Sort Jobs in descending order on the basis
    // of their profit
    sort(arr, arr + n, cmp);
 
    // Find the maximum deadline among all jobs and
    // create a disjoint set data structure with
    // maxDeadline disjoint sets initially.
    int maxDeadline = findMaxDeadline(arr, n);
    DisjointSet ds(maxDeadline);
 
    // Traverse through all the jobs
    for (int i = 0; i < n; i++)
    {
        // Find the maximum available free slot for
        // this job (corresponding to its deadline)
        int availableSlot = ds.find(arr[i].deadLine);
 
        // If maximum available free slot is greater
        // than 0, then free slot available
        if (availableSlot > 0)
        {
            // This slot is taken by this job 'i'
            // so we need to update the greatest
            // free slot. Note that, in merge, we
            // make first parameter as parent of
            // second parameter. So future queries
            // for availableSlot will return maximum
            // available slot in set of
            // "availableSlot - 1"
            ds.merge(ds.find(availableSlot - 1),availableSlot);
 
            cout << arr[i].id << " ";
        }
    }
}