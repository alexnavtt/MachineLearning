import math

class PriorityQueueItem:
    def __init__(self, value, priority):
        self.value = value
        self.priority = priority

class PriorityQueue:
    def __init__(self, maxsize=math.inf):
        self.data = []
        self.maxsize = maxsize
        self.datalen = 0

    def __getitem__(self, i):
        return self.data[i].value

    def __len__(self):
        return len(self.data)
    
    def insert(self, item, priority):
        append = False

        # If we start with an empty array, just add the item
        if self.datalen == 0:
            self.data.append(PriorityQueueItem(item, priority))
            self.datalen += 1
            return True

        # If we have a not full array, insert a new bogus element to be replaced
        elif self.datalen < self.maxsize:
            # Add on an extra element
            append = True

        # If the priority value is too high, don't bother
        elif priority > self.data[-1].priority:
            return False

        if append:
            self.data.append(0)
            self.datalen += 1

        # Shift all elements with less priority upwards in the queue
        for i in range(self.datalen-1, 0, -1):
            next = self.data[i-1]
            if priority <= next.priority:
                self.data[i] = next
            else:
                self.data[i] = PriorityQueueItem(item, priority)
                return True

        self.data[0] = PriorityQueueItem(item, priority)
        return True

if __name__ == "__main__":
    q = PriorityQueue(maxsize=2)
    q.insert("four", 4)
    q.insert("three", 3)
    q.insert("two", 2)
    q.insert("another", 1)

    print("--------")
    for item in q.data:
        print(item.value)
        print(item.priority)
    





