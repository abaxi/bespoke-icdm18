import copy, random

class topN:
    def __init__(self, N):
        self.max_size = N
        self.ordered_list = []

    def insert_update(self, key, value):
        pair = (key, value)
        if len(self.ordered_list) == 0:
            self.ordered_list.append(pair)
            return
        elif value <= self.list_min() and len(self.ordered_list)==self.max_size:
            return
        pos = self.find_pos(value)
        self.ordered_list.insert(pos, pair)
        if len(self.ordered_list) > self.max_size:
            self.ordered_list.pop() #pops last element

    def find_pos(self, value):
        start = 0
        end = len(self.ordered_list)
        while start<end:
            mid = (start+end)/2
            if mid == start:
                break
            mid_val = self.ordered_list[mid][1]
            if mid_val < value:
                end = mid
            elif mid_val > value:
                start = mid
            else:
                break
        mid_val = self.ordered_list[mid][1]
        if value <= mid_val:
            insert_at = mid +1
        else:
            insert_at = max(0,mid-1)
        return insert_at

    def pop(self,index=0):
        return self.ordered_list.pop(index)

    def insert_last(self, pair):
        self.ordered_list.append(pair)
        
    def list_min(self):
        return self.ordered_list[-1][1]

    def __repr__(self):
        return str(self.ordered_list)

    def __str__(self):
        return str(self.ordered_list)

    def  __len__(self):
        return len(self.ordered_list)
