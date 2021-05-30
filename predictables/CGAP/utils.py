import math

def checknan(val):
    if isinstance(val,str):
        return val
    if math.isnan(val):
        return -1
    else:
        return val