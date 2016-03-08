# optionLoop
Simple loop structure to iterate over all combinations of an initializing dictionary

No longer will you need a million nested for-loops...

The optionloop works as follows:

First, initialize a dictionary with various keys and values, e.g.:

```python
d = {'doThingX' : [True, False], 'doThingY' : False,
		'thingZValue' : ['a', 'b', 1]}
```

Next create the option loop:

```python
oploop = optionloop(d)
```

Finally iterate and get your values:

```python
for state in oploop:
	doX = state['doThingX']
	doY = state['doThingY']
	zVal = state['thingZValue']

	f(doX, doY, zVal)
```

This is intended to replace an equivalent looping structure of:

```python
for doX in doThingX:
	for doY in doThingY:
		for zVal in thingZValue:
			f(doX, doY, zVal)
```

which quickly becomes cumbersome.

Also, option loops can be added to create even more complex looping structures, e.g.:

```python
d1 = {'lang' : ['c'], 'doThingX' : [True, False]}
d2 = {'lang' : ['fortran'], 'doThingX' : [True, False], 'doThingY' : [True, False]}

oploop1 = optionloop(d1)
oploop2 = optionloop(d2)
oploop = oploop1 + oploop2

for state in oploop:
	...
```

is equivalent to:

```python
langs = ['c', 'fortran']
doThingX = [True, False]
doThingY = [True, False]

for lang in langs:
	if lang == 'c':
		for doX in doThingX:
			f(lang, doX)
	elif lang == 'fortran':
		for doX in doThingX:
			for doY in doThingY:
				f(lang, doX, doY)
```

Note, if the order of iteration matters an ordered dict can be used, e.g.:

```python
d = OrderedDict()
d['a'] = [False, True]
d['b'] = [False]
d['c'] = [1, 2, 3]
oploop = optionloop(d)
for state in oploop:
	...
```

is equivalent to:

```python
for a in [False, True]:
	for b in [False]:
		for c in [1, 2, 3]:
			....
```
