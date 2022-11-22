# How to find the smallest number with given digit sum s and number of 
# digits d? 
# Examples : 
 

# Input  : s = 9, d = 2
# Output : 18
# There are many other possible numbers 
# like 45, 54, 90, etc with sum of digits
# as 9 and number of digits as 2. The 
# smallest of them is 18.

# Input  : s = 20, d = 3
# Output : 299


# Prints the smallest possible
# number with digit sum 's'
# and 'm' number of digits.
 
def findSmallest(m,s):
 
    # If sum of digits is 0,
    # then a number is possible
    # only if number of digits is 1.
    if (s == 0):
         
        if(m == 1) :
              print("Smallest number is 0")
        else :
              print("Not possible")
        return
  
    # Sum greater than the
    # maximum possible sum.
    if (s > 9*m):
     
        print("Not possible")
        return
  
    # Create an array to
    # store digits of result
    res=[0 for i in range(m+1)]
  
    # deduct sum by one to
    # account for cases later
    # (There must be 1 left
    # for the most significant
    #  digit)
    s -= 1
  
    # Fill last m-1 digits
    # (from right to left)
    for i in range(m-1,0,-1):
     
        # If sum is still greater than 9,
        # digit must be 9.
        if (s > 9):
         
            res[i] = 9
            s -= 9
     
        else:
         
            res[i] = s
            s = 0
  
    # Whatever is left should
    # be the most significant
    # digit.
    # The initially subtracted 1 is
    # incorporated here.
    res[0] = s + 1
                    
  
    print("Smallest number is ",end="")
    for i in range(m):
        print(res[i],end="")
 
 
s = 9
m = 2
findSmallest(m, s)