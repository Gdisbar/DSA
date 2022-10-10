850 · Employee Free Time
=========================
// We are given a list schedule of employees, which represents the working 
// time for each employee.

// Each employee has a list of non-overlapping Intervals, and these intervals 
// are in sorted order.

// Return the list of finite intervals representing common, positive-length 
// free time for all employees, also in sorted order.

// The Intervals is an 1d-array. Each two numbers shows an interval. 
// For example, [1,2,8,10] represents that the employee works in [1,2] and [8,10].

// Also, we wouldn''t include intervals like [5, 5] in our answer, as they 
// have zero length.


// Input：schedule = [[1,2,5,6],[1,3],[4,10]]

// Output：[(3,4)]

// Explanation:

// There are a total of three employees, and all common

// free time intervals would be [-inf, 1], [3, 4], [10, inf].

// We discard any intervals that contain inf as they aren''t finite.


// Input：schedule = [[1,3,6,7],[2,4],[2,5,9,12]]

// Output：[(5,6),(7,9)]

// Explanation：

// There are a total of three employees, and all common

// free time intervals would be [-inf, 1], [5, 6], [7, 9],[12,inf].

// We discard any intervals that contain inf as they aren''t finite.


/**
 * Definition for an interval.
 * public class Interval {
 *     int start;
 *     int end;
 *     Interval() { start = 0; end = 0; }
 *     Interval(int s, int e) { start = s; end = e; }
 * }
 */
class Solution {
    public List<Interval> employeeFreeTime(List<List<Interval>> avails) {
        List<Interval> result = new ArrayList<>();
        List<Interval> timeLine = new ArrayList<>();
        avails.forEach(e -> timeLine.addAll(e));
        Collections.sort(timeLine, ((a, b) -> a.start - b.start));

        Interval temp = timeLine.get(0);
        for (Interval each : timeLine) {
            if (temp.end < each.start) {
                result.add(new Interval(temp.end, each.start));
                temp = each;
            } else {
                temp = temp.end < each.end ? each : temp;
            }
        }
        return result;
    }
}