#questions taken from leetcode. I specify the problem number in the docstring, and a shortened description
#No unittests, since leetcode environment allows you to easily set up tests to run against your code

from itertools import permutations
from collections import counter
 class TreeNode:
     def __init__(self, x):
         self.val = x
         self.left = None
         self.right = None
#provided by leetcode

#problem: 654
def constructMaximumBinaryTree(self, nums: List[int]) -> TreeNode:
    """problem: 654
       Given an integer array with no duplicates build max tree
    """
    if not nums:
        return
    max_index=nums.index(max(nums))
    root=TreeNode(nums[max_index])
    #set root to be largest value of nums
    root.left=self.constructMaximumBinaryTree(nums[:max_index])
    root.right=self.constructMaximumBinaryTree(nums[max_index+1:])
    #recurse on left and right halves
    return root

#problem: 46
def permute(self, nums, answer=[], sofar=None):
       """
       problem: 46
        generate all permutations of a number
       """
       #backtracing method
       if sofar is None:
           sofar=[]
       if len(sofar)== len(nums):
           answer.append(sofar)
           return
       for num in nums:
           if num not in sofar:
               copy=sofar+[num]
               self.permute(nums, answer, copy)
       return answer
def permute_lazy(self, nums):
       """
       problem: 46, the lazy way
       """
        return list(itertools.permutations(nums))

# problem 451
def frequencySort(s: str) -> str:
    """
    problem: 451
    Given a string, sort it in decreasing order based on the frequency of characters.
    """
    counts=Counter(s).most_common()
    #returns in decreasing order
    answer=[letter*count for letter,count in counts]
    return ''.join(answer)

# problem 287
def findDuplicate(self, nums: List[int]) -> int:
    """
    problem 287
    find duplicate number in nums
    """
    for num in nums:
        if nums[abs(num)-1] < 0:
            return abs(num)
        else:
            nums[abs(num)-1]*=-1
    #could simply use a hashmap or a counter, but this uses O(1) space
    #we encode whether a number has been visited before using that numbers sign

#problem 98
def isValidBST(self, root):
    """
    problem 98
    check if a binary search tree is valid
    """
    if root is None:
        return True
    return self.issorted(self.inorder(root))

def inorder(self, bst, L=None):
    """do an in order traversal of bst"""
    if L is None: L=[]
    if bst.left is not None:
        self.inorder(bst.left, L)
    L.append(bst.val)
    if bst.right is not None:
        self.inorder(bst.right, L)
    return L

def issorted(self, L):
    """check if bst is sorted"""
    for i in range(len(L)-1):
        if L[i] >= L[i+1]:
            return False
    return True

#problem 62

def uniquePaths(self, m, n):
    """ how many unique ways are there to go from (0,0) to (m,n)
    in grid going only down or right"""
    g=Grid(m,n)
    return g.countpaths()

class Grid:
    def __init__(self, m,n):
        self._grid=[[1 for i in range(m)] for j in range(n)]
        self.m=m
        self.n=n
    def __getitem__(self,tup):
        m,n=tup[0], tup[1]
        return self._grid[n][m]
    def countpaths(self):
        #dp to count paths
        for y in range(1, self.n):
            for x in range(1, self.m):
                self._grid[y][x] = self._grid[y-1][x]+ self._grid[y][x-1]
        return self._grid[self.n-1][self.m-1]
    def __str__(self):
        return str(self._grid)

#problem 33
def search(self, nums: List[int], target: int, L=0, R=None) -> int:
    """ problem 33
    search for target in rotated sorted list nums
    """
    #trivial solution is O(N), but we can use the sorted property to reduce to O(log(n))
    if R is None:
        R=len(nums)-1
    if R < L:
        return -1
    first, middle, last=nums[L], nums[(L+R)//2], nums[R]
    mid=(L+R)//2
    if target==middle:
        return mid
    if first > middle:
        #we are on right side of breakpoint. all stuff to right is sorted
        if target in range(middle, last+1):
            return self.search(nums, target, mid+1, R)
        else:
            return self.search(nums, target, L, mid-1)
    else:
        #we are on left side of breakpoint. All stuff to left is sorted.
        if target in range(first, middle+1):
            return self.search(nums, target, L, mid-1)
        else:
            return self.search(nums, target, mid+1, R)

#problem 39
   def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        """
        problem 39. given candidate list of ints
        find all unique combinations of candidates that sums to target
        """
        #build up shorter lists that we append to the big list.
        #backtracking style problem
        candidates.sort()
        answer=[]
        self._combinationSum(candidates, 0,target, answer, [])
        return answer

    def _combinationSum(self, candidates, index, target, answer, path):
        if target==0:
            answer.append(path)
            return
        #we've foud a valid solution
        if target < 0:
            return
        #its hopeless, give up and backtrack
        for x in range(index, len(candidates)):
            if candidates[x] > target:
                break
            self._combinationSum(candidates, x, target-candidates[x], answer, path+[candidates[x]])

#problem 19
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        """problem 19
        remove nth-from-end node of linked list
        """
        #rather than doing an initial traversal of the LL to find its length,
        #we can send a fast runner out n links ahead so we know which position to emove in one pass
        current=runner= head
        while(n > 0):
            runner=runner.next
            n-=1
        #runner is now n ahead of current
        while(runner and runner.next):
            current=current.next
            runner=runner.next
        if runner is None:
            head=head.next
        if current.next is None:
            return None
        current.next=current.next.next
        return head
#problem 15

def threeSum(self, nums: List[int]) -> List[List[int]]:
    """Given an array nums of n integers, are there elements a, b, c in nums such that a + b + c = 0?
    Find all unique triplets in the array which gives the sum of zero."""
    #two pointers method
    answer=[]
    nums.sort()
    i=0
    for i in range(len(nums)-2):
        if (i > 0 and nums[i-1]==nums[i]):
            continue
        #if there are repeat numbers, skip-- we want all unique triplets.
        L, R= i+1, len(nums)-1
        while(L < R):
            numlist= (nums[i], nums[L], nums[R])
            total=sum(numlist)
            if total == 0:
                answer.append(numlist)
                while(L < R and nums[L+1]==nums[L]):
                    L+=1
                while(L< R and nums[R-1]==nums[R]):
                    R-=1
                #skip over repeated elements for our left and right pointers
                R-=1
                L+=1
            elif total > 0:
                R-=1
                #move right pointer in
            else:
                L+=1
                #move left pointer in
    return answer

#problem 16

def threeSumClosest(self, nums: List[int], target: int) -> int:
    """given an array nums of n integers and an integer target,
 find three integers in nums such that the sum is closest to target"""
    nums.sort()
    dist= float("inf")
    #initial distance
    answer=[]
    #will hold our three integers which are closest
    i=0
    while(i < len(nums)-2):
        while(i > 0 and i < len(nums)-2 and nums[i] == nums[i-1]):
            i+=1
        L,R= i+1, len(nums)-1
        while(L < R):
            numlist= (nums[i], nums[L], nums[R])
            if abs(target-sum(numlist)) < dist:
                #if this is the closest we've seen so far
                dist = abs(target-sum(numlist))
                answer=numlist
            if sum(numlist) > target:
                R-=1
            elif(sum(numlist) < target):
                L+=1
            else:
                #if our list exactly equals target, we can exit early.
                return sum(answer)
        i+=1
    return sum(answer)

#problem 11
def maxArea(self, height: List[int]) -> int:
    """the worlds most convoluted problem-- check leetcode for full description"""
    maxarea, L, R= 0, 0, len(height)-1
    #two pointers method
    while(L != R):
        A=min(height[L], height[R])*(R-L)
        maxarea= max(maxarea, A)
        if(height[L]< height[R]):
            L+=1
        else:
            R-=1
    return maxarea

#problem 2
def addTwoNumbers(self, l1: ListNode, l2: ListNode, c=0) -> ListNode:
"""You are given two non-empty linked lists representing two non-negative integers.
 The digits are stored in reverse order and each of their nodes contain a single digit.
Add the two numbers and return it as a linked list."""
    total=l1.val+l2.val+c
    c= total // 10
    ret= ListNode(total % 10)
    if l1.next or l2.next or c:
        if l1.next is None and l2.next is None:
            l1.next=ListNode(0)
            l2.next=ListNode(0)
        if l1.next is None:
            l1.next= ListNode(0)
        elif l2.next is None:
            l2.next=ListNode(0)
        ret.next=self.addTwoNumbers(l1.next, l2.next, c)
    return ret
