""" This module defines basic data structures classes """

from common_header import *

class Node(object):
    """ Basic unit class that can be used in stack or queue """
    def __init__(self, data=None):
        self.data = data
        self.next = None

class Queue(object):
    """ FIFO Queue """

    def __init__(self):
        self.head = None
        self.tail = None
        self.length = 0
    
    def is_empty(self):
        return self.length == 0

    def enqueue(self, new_node):
        if self.is_empty():
            self.head = self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node
        
        self.length += 1
    
    def dequeue(self):

        if self.is_empty():
            print("ERROR: Cannot dequeue from empty Queue")
            return None
    
        removed_node = self.head
        self.head = self.head.next
        removed_node.next = None

        self.length -= 1

        return removed_node


