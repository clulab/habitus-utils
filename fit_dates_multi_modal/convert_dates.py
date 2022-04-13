import time
import numpy

_DATE_RANGE = ("2000-05-12", "2020-05-12")
_DATE_FORMAT = "%Y-%m-%d"
_EMPIRICAL_SCALE_RATIO = 0.15
_DISTRIBUTION_SIZE = 1000


time_range = tuple(time.mktime(time.strptime(d, _DATE_FORMAT)) for d in _DATE_RANGE)

distribution = numpy.random.normal(
    loc=(time_range[0] + time_range[1]) * 0.5,
    scale=(time_range[1] - time_range[0]) * _EMPIRICAL_SCALE_RATIO,
    size=_DISTRIBUTION_SIZE,
)
date_range = tuple(
    time.strftime(_DATE_FORMAT, time.localtime(t)) for t in numpy.sort(distribution)
)
print(date_range)
