import unittest
from optionloop import optionloop
from collections import OrderedDict
"""
Simple unit tests for optionloop, requires unittest
"""

class TestOptionLoop(unittest.TestCase):
    def test_empty(self):
        d = {}
        op = optionloop(d)
        i = None
        for i in op:
            pass
        self.assertTrue(i is None)

    def test_string1(self):
        d = {'a' : 'a'}
        op = optionloop(d)
        for i in op:
            self.assertTrue(i['a'] == 'a')

    def test_string2(self):
        d = {'a' : 'abc'}
        op = optionloop(d)
        for i in op:
            self.assertTrue(i['a'] == 'abc')

    def test_multiple_values1(self):
        d = {'a' : [False, True]}
        op = optionloop(d)
        for i, state in enumerate(op):
            self.assertTrue(state['a'] == i)

    def test_multiple_values2(self):
        d = OrderedDict()
        d['a'] = [False, True]
        d['b'] = [False]
        op = optionloop(d)
        for i, state in enumerate(op):
            self.assertTrue(state['a'] == i % 2)
            self.assertTrue(state['b'] == False)

    def test_multiple_values3(self):
        d = OrderedDict()
        d['a'] = [False, True]
        d['b'] = [False]
        d['c'] = [1, 2, 3]
        op = optionloop(d)
        for i, state in enumerate(op):
            self.assertTrue(state['a'] == i % 2)
            self.assertTrue(state['b'] == False)
            self.assertTrue(state['c'] == (i / 2) + 1)

    def test_no_len(self):
        d = {'a' : None}
        op = optionloop(d)
        for i in op:
            self.assertTrue(i['a'] is None)

    def test_add_oploops(self):
        d = {'a' : [False, True]}
        op1 = optionloop(d)
        d = {'a' : [1, 2]}
        op2 = optionloop(d)
        op = op1 + op2
        for i, state in enumerate(op):
            if i < 2:
                self.assertTrue(state['a'] == i % 2)
            else:
                self.assertTrue(state['a'] == i - 1)

    def test_add_oploops2(self):
        d = {'a' : [False]}
        op1 = optionloop(d)
        d = {'a' : [True]}
        op2 = optionloop(d)
        op = op1 + op2
        d = {'a' : [1, 2]}
        op3 = optionloop(d)
        op = op + op3
        for i, state in enumerate(op):
            if i < 2:
                self.assertTrue(state['a'] == i % 2)
            else:
                self.assertTrue(state['a'] == i - 1)

    def test_add_oploops2(self):
        d = {'a' : [False]}
        op1 = optionloop(d)
        d = {'a' : [True]}
        op2 = optionloop(d)
        op = op1 + op2
        d = {'a' : [1, 2]}
        op3 = optionloop(d)
        op = op3 + op
        for i, state in enumerate(op):
            if i >= 2:
                self.assertTrue(state['a'] == i % 2)
            else:
                self.assertTrue(state['a'] == i + 1)


    def test_add_oploops_bad(self):
        d = {'a' : [False, True]}
        op1 = optionloop(d)
        for i, state in enumerate(op1):
            op2 = optionloop(d)
            try:
                op = op1 + op2
                self.assertTrue(False)
            except Exception:
                self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()