import itertools
import operator

#list (list(objs)) to map (key -> list(objs grouped by key))
def group_by_key(data, key_group_by):
    list_grouped = []
    data = sorted(data, key=operator.itemgetter(key_group_by))
    for key, items in itertools.groupby(data, operator.itemgetter(key_group_by)):
        list_grouped.append(list(items))
    return {v[0][key_group_by]:v for v in list_grouped}


import re

#https://stackoverflow.com/questions/2669059/how-to-sort-alpha-numeric-set-in-python
def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)
