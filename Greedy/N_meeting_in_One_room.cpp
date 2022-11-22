N meetings in one room
==========================
There is one meeting room in a firm. There are N meetings in the form of 
(start[i], end[i]) where start[i] is start time of meeting i and end[i] is 
finish time of meeting i.
What is the maximum number of meetings that can be accommodated in the meeting 
room when only one meeting can be held in the meeting room at a particular time?

Note: Start time of one chosen meeting can''t be equal to the end time of the 
other chosen meeting.


Example 1:

Input:
N = 6
start[] = {1,3,0,5,8,5}
end[] =  {2,4,6,7,9,9}
Output: 
4
Explanation:
Maximum four meetings can be held with
given start and end timings.
The meetings are - (1, 2),(3, 4), (5,7) and (8,9)

Example 2:

Input:
N = 3
start[] = {10, 12, 20}
end[] = {20, 25, 30}
Output: 
1
Explanation:
Only one meetings can be held
with given start and end timings.


//TC : n*log(n) , SC : n

int maxMeetings(int start[], int end[], int n){
        // Your code here
        vector<pair<int,int>> a(n);
        vector<int> m; //final selected meetings.
        for(int i=0;i<n;++i) a[i]={end[i],i};
        sort(a.begin(),a.end());
        int cnt=1,tle=a[0].first; //check if new meeting can be arranged or not
        m.push_back(a[0].second+1); //select 1st meeting,0-based indexing so add 1
        for(int i=1;i<n;++i){
            if(start[a[i].second]>tle){
                tle=a[i].first;
                m.push_back(a[i].second+1);
                cnt++;
            }
        }
        return cnt;
    }