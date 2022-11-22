Maximum Bipartite Matching
============================
// There are M job applicants and N jobs.  Each applicant has a subset of 
// jobs that he/she is interseted in. Each job opening can only accept one 
// applicant and a job applicant can be appointed for only one job. Given a 
// matrix G where G(i,j) denotes ith applicant is interested in jth job. Find 
// an assignment of jobs to applicant in such that as many applicants as 
// possible get jobs.
 

// Example 1:

// Input: G = {{1,1,0,1,1},{0,1,0,0,1},
// {1,1,0,1,1}}
// Output: 3
// Explanation: There is one of the possible
// assignment-
// First applicant gets the 1st job.
// Second applicant gets the 2nd job.
// Third applicant gets the 3rd job.

// Example 2:

// Input: G = {{1,1},{0,1},{0,1},{0,1},
// {0,1},{1,0}}
// Output: 2
// Explanation: There is one of the possible
// assignment-
// First applicant gets the 1st job.
// Second applicant gets the 2nd job.


// Maximum Bipartite Matching (MBP) problem can be solved by converting it into a 
// flow network 

class Solution {
public:

    bool find_mapping(int applicant, vector<bool> &job_visited, 
    	              vector<int> &mapping, vector<vector<int>> &g) {
        for (int i = 0; i < g[applicant].size(); i++) {
            if (g[applicant][i] and !job_visited[i]) {
                job_visited[i] = true;
                if (mapping[i] == -1 or 
                	find_mapping(mapping[i], job_visited, mapping, g)) {
                    mapping[i] = applicant;
                    return true;
                }
            }
        }
        return false;
    }

	int maximumMatch(vector<vector<int>>&G){
	    int applicants = G.size(), jobs = G[0].size(), result = 0;
	    // Mapping which will give the applicant corresponding to a job.
	    vector<int> mapping(jobs, -1);
	    for (int i = 0; i < applicants; i++) {
	        // This vector will tell us whether the job has been visited in the DFS call or not.
	        vector<bool> job_visited(jobs, false);
	        // If we are able to map the applicant to a job then mapping will inc. by 1.
	        if (find_mapping(i, job_visited, mapping, G))
	            result += 1;
	    }
	    return result;
	}

};