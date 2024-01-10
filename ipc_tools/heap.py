# This code has been extensively copied from the heapq module

# self.heap.pop -> removes last item
# self.heap.append -> appends to the right

class Heap:

    def __init__(self):
        self.heap = []

    def push(self, item):
        """Push item onto heap, maintaining the heap invariant."""
        self.heap.append(item)
        self._siftdown(0, len(self.heap)-1)

    def pop(self):
        """Pop the smallest item off the heap, maintaining the heap invariant."""
        lastelt = self.heap.pop()    # raises appropriate IndexError if heap is empty
        if self.heap:
            returnitem = self.heap[0]
            self.heap[0] = lastelt
            self._siftup(0)
            return returnitem
        return lastelt

    def replace(self, item):
        """Pop and return the current smallest value, and add the new item.

        This is more efficient than heappop() followed by heappush(), and can be
        more appropriate when using a fixed-size heap.  Note that the value
        returned may be larger than item!  That constrains reasonable uses of
        this routine unless written as part of a conditional replacement:

            if item > heap[0]:
                item = heapreplace(heap, item)
        """
        returnitem = self.heap[0]    # raises appropriate IndexError if heap is empty
        self.heap[0] = item
        self._siftup(0)
        return returnitem

    def pushpop(self, item):
        """Fast version of a heappush followed by a heappop."""
        if self.heap and self.heap[0] < item:
            item, self.heap[0] = self.heap[0], item
            self._siftup(0)
        return item

    def _pop_max(self):
        """Maxheap version of a heappop."""
        lastelt = self.heap.pop()    # raises appropriate IndexError if heap is empty
        if self.heap:
            returnitem = self.heap[0]
            self.heap[0] = lastelt
            self._siftup_max(0)
            return returnitem
        return lastelt

    def _replace_max(self, item):
        """Maxheap version of a heappop followed by a heappush."""
        returnitem = self.heap[0]    # raises appropriate IndexError if heap is empty
        self.heap[0] = item
        self._siftup_max(0)
        return returnitem

    # 'heap' is a heap at all indices >= startpos, except possibly for pos.  pos
    # is the index of a leaf with a possibly out-of-order value.  Restore the
    # heap invariant.
    def _siftdown(self, startpos, pos):
        newitem = self.heap[pos]
        # Follow the path to the root, moving parents down until finding a place
        # newitem fits.
        while pos > startpos:
            parentpos = (pos - 1) >> 1
            parent = self.heap[parentpos]
            if newitem < parent:
                self.heap[pos] = parent
                pos = parentpos
                continue
            break
        self.heap[pos] = newitem

    def _siftup(self, pos):
        endpos = len(self.heap)
        startpos = pos
        newitem = self.heap[pos]
        # Bubble up the smaller child until hitting a leaf.
        childpos = 2*pos + 1    # leftmost child position
        while childpos < endpos:
            # Set childpos to index of smaller child.
            rightpos = childpos + 1
            if rightpos < endpos and not self.heap[childpos] < self.heap[rightpos]:
                childpos = rightpos
            # Move the smaller child up.
            self.heap[pos] = self.heap[childpos]
            pos = childpos
            childpos = 2*pos + 1
        # The leaf at pos is empty now.  Put newitem there, and bubble it up
        # to its final resting place (by sifting its parents down).
        self.heap[pos] = newitem
        self._siftdown(startpos, pos)

    def _siftdown_max(self, startpos, pos):
        'Maxheap variant of _siftdown'
        newitem = self.heap[pos]
        # Follow the path to the root, moving parents down until finding a place
        # newitem fits.
        while pos > startpos:
            parentpos = (pos - 1) >> 1
            parent = self.heap[parentpos]
            if parent < newitem:
                self.heap[pos] = parent
                pos = parentpos
                continue
            break
        self.heap[pos] = newitem

    def _siftup_max(self, pos):
        'Maxheap variant of _siftup'
        endpos = len(self.heap)
        startpos = pos
        newitem = self.heap[pos]
        # Bubble up the larger child until hitting a leaf.
        childpos = 2*pos + 1    # leftmost child position
        while childpos < endpos:
            # Set childpos to index of larger child.
            rightpos = childpos + 1
            if rightpos < endpos and not self.heap[rightpos] < self.heap[childpos]:
                childpos = rightpos
            # Move the larger child up.
            self.heap[pos] = self.heap[childpos]
            pos = childpos
            childpos = 2*pos + 1
        # The leaf at pos is empty now.  Put newitem there, and bubble it up
        # to its final resting place (by sifting its parents down).
        self.heap[pos] = newitem
        self._siftdown_max(startpos, pos)

    def __str__(self):
        return str(self.heap)


from multiprocessing import RawArray
from enum import Enum

class HeapType(Enum):
    MAXHEAP = -1,
    MINHEAP = 1,   

    # this is useful for argparse
    def __str__(self):
        return self.name

class Heap2:

    def __init__(self, heapsize: int, heaptype: HeapType = HeapType.MINHEAP) -> None:
        self.heapsize = heapsize
        self.numel = 0
        self.heap = RawArray('d', self.heapsize)
        self.heaptype = heaptype
        if heaptype == HeapType.MINHEAP:
            self.push = self._push_min
            self.pop = self._pop_min
            self.poppush = self._poppush_min
            self.pushpop = self._pushpop_min
        elif heaptype == HeapType.MAXHEAP:
            self.push = self._push_max
            self.pop =  self._pop_max
            self.poppush = self._poppush_max
            self.pushpop = self._pushpop_max
        else:
            raise ValueError('Unknown heap type')

    def _push_min(self, item):
        """Push item onto heap, maintaining the heap invariant."""
        self.heap[self.numel] = item
        self._siftdown(0, self.numel)
        self.numel += 1

    def _push_max(self, item):
        """Push item onto heap, maintaining the heap invariant."""
        self.heap[self.numel] = item
        self._siftdown_max(0, self.numel)
        self.numel += 1

    def _siftdown(self, startpos, pos):
        newitem = self.heap[pos]
        # Follow the path to the root, moving parents down until finding a place
        # newitem fits.
        while pos > startpos:
            parentpos = (pos - 1) >> 1
            parent = self.heap[parentpos]
            if newitem < parent:
                self.heap[pos] = parent
                pos = parentpos
                continue
            break
        self.heap[pos] = newitem

    def _pop_min(self):
        """Pop the smallest item off the heap, maintaining the heap invariant."""
        self.numel -= 1
        if self.numel < 0:
            raise IndexError('pop from empty heap') 
        lastelt = self.heap[self.numel]    
        if self.numel > 0:
            returnitem = self.heap[0]
            self.heap[0] = lastelt
            self._siftup(0)
            return returnitem
        return lastelt

    def _siftup(self, pos):
        endpos = self.numel
        startpos = pos
        newitem = self.heap[pos]
        # Bubble up the smaller child until hitting a leaf.
        childpos = 2*pos + 1    # leftmost child position
        while childpos < endpos:
            # Set childpos to index of smaller child.
            rightpos = childpos + 1
            if rightpos < endpos and not self.heap[childpos] < self.heap[rightpos]:
                childpos = rightpos
            # Move the smaller child up.
            self.heap[pos] = self.heap[childpos]
            pos = childpos
            childpos = 2*pos + 1
        # The leaf at pos is empty now.  Put newitem there, and bubble it up
        # to its final resting place (by sifting its parents down).
        self.heap[pos] = newitem
        self._siftdown(startpos, pos)

    def _poppush_min(self, item):
        """Pop and return the current smallest value, and add the new item.

        This is more efficient than heappop() followed by heappush(), and can be
        more appropriate when using a fixed-size heap.  Note that the value
        returned may be larger than item!  That constrains reasonable uses of
        this routine unless written as part of a conditional replacement:

            if item > heap[0]:
                item = heapreplace(heap, item)
        """
        if self.numel == 0:
            raise IndexError('replace from empty heap') 
        returnitem = self.heap[0]   
        self.heap[0] = item
        self._siftup(0)
        return returnitem

    def _pushpop_min(self, item):
        """Fast version of a heappush followed by a heappop."""
        if self.numel > 0 and self.heap[0] < item:
            item, self.heap[0] = self.heap[0], item
            self._siftup(0)
        return item
    
    def _pushpop_max(self, item): # not sure that works, please test 
        """Fast version of a heappush followed by a heappop."""
        if self.numel > 0 and self.heap[0] < item:
            item, self.heap[0] = self.heap[0], item
            self._siftup_max(0)
        return item
    
    def _pop_max(self):
        """Maxheap version of a heappop."""
        self.numel -= 1
        if self.numel < 0:
            raise IndexError('pop from empty heap') 
        lastelt = self.heap[self.numel]    
        if self.numel > 0:
            returnitem = self.heap[0]
            self.heap[0] = lastelt
            self._siftup_max(0)
            return returnitem
        return lastelt

    def _poppush_max(self, item):
        """Maxheap version of a heappop followed by a heappush."""
        if self.numel == 0:
            raise IndexError('replace from empty heap') 
        returnitem = self.heap[0]    
        self.heap[0] = item
        self._siftup_max(0)
        return returnitem

    def _siftdown_max(self, startpos, pos):
        'Maxheap variant of _siftdown'
        newitem = self.heap[pos]
        # Follow the path to the root, moving parents down until finding a place
        # newitem fits.
        while pos > startpos:
            parentpos = (pos - 1) >> 1
            parent = self.heap[parentpos]
            if parent < newitem:
                self.heap[pos] = parent
                pos = parentpos
                continue
            break
        self.heap[pos] = newitem

    def _siftup_max(self, pos):
        'Maxheap variant of _siftup'
        endpos = self.numel
        startpos = pos
        newitem = self.heap[pos]
        # Bubble up the larger child until hitting a leaf.
        childpos = 2*pos + 1    # leftmost child position
        while childpos < endpos:
            # Set childpos to index of larger child.
            rightpos = childpos + 1
            if rightpos < endpos and not self.heap[rightpos] < self.heap[childpos]:
                childpos = rightpos
            # Move the larger child up.
            self.heap[pos] = self.heap[childpos]
            pos = childpos
            childpos = 2*pos + 1
        # The leaf at pos is empty now.  Put newitem there, and bubble it up
        # to its final resting place (by sifting its parents down).
        self.heap[pos] = newitem
        self._siftdown_max(startpos, pos)

    def __str__(self):
        return str(list(self.heap[:self.numel]))