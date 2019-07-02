def permute(self, nums, answer=[], sofar=None):
       """
       :type nums: List[int]
       :rtype: List[List[int]]
       problem: 46
       """
       #realistically i'd import it from itertools. But lets get mildly clevered.
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
