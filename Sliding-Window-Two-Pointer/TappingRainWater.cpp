// Given n non-negative integers representing an elevation map where the 
//width of each bar is 1, compute how much water it can trap after raining.

 

// Example 1:

// Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
// Output: 6
// Explanation: The above elevation map (black section) is 
// represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 
// 6 units of rain water (blue section) are being trapped.

// Example 2:

// Input: height = [4,2,0,3,2,5]
// Output: 9


//===========================================================================================

int trap(vector<int>& height) {
        
        int n = height.size();
        vector<int> left(n);
        vector<int> right(n);
        int ans = 0;
        
        left[0] = height[0];
        for(int i = 1;i<n;i++)
            left[i]=max(left[i-1],height[i]);
        
        right[n-1] = height[n-1];
        for(int i = n-2;i>=0;i--)
            right[i]=max(right[i+1],height[i]);
        
        for(int i = 0;i<n;i++)
            ans += min(left[i],right[i])-height[i];
        
        return ans;
    }

//===========================================================================================
// speed slower,space slightly better
    
int trap(vector<int>& height) {
        
        int n = height.size();
        int left_max = 0, right_max = 0;
        int ans = 0,l = 0,h = n - 1;
        
        while(l<=h){
            if(height[l]<height[h]){
                if(height[l]>left_max)
                    left_max = height[l];
                else
                    ans += left_max - height[l];
                l++;
            }else{
                if(height[h]>right_max)
                    right_max = height[h];
                else
                    ans += right_max - height[h];
                h--;
            }
        }
        
        return ans;
    }

//===========================================================================================


//===========================================================================================
