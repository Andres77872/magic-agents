"""Decode complete JSON without repairing provider output or discarding fields."""
import json
import math


def _pairs(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError()
        result[key]=value
    return result


def _integer(value):
    if value=='-0':raise ValueError()
    return int(value)


def _float(value):
    result=float(value)
    if not math.isfinite(result) or result==0 and math.copysign(1,result)<0:raise ValueError()
    return result


def _constant(value):
    raise ValueError()


def strict_json(value):
    try:
        return json.loads(value,object_pairs_hook=_pairs,parse_int=_integer,
            parse_float=_float,parse_constant=_constant)
    except (ValueError,TypeError,RecursionError,UnicodeError):
        pass
    raise ValueError('Invalid JSON output')
