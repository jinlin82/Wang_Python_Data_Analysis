import math
math.sin(3)
""""""

import pkgutil
import numpy

for importer, modname, ispkg in pkgutil.iter_modules(numpy.__path__, prefix="numpy."):
    print(modname)

""""""

class Point:
    """Represents a point in 2-D space."""
Point()
""""""

>>> Point
<class '__main__.Point'>

>>> blank = Point()
>>> blank
<__main__.Point object at 0xb7e9d3ac>

>>> blank.x = 3.0
>>> blank.y = 4.0

>>> blank.y
4.0
>>> x = blank.x
>>> x
3.0

a=[1,2,3]
import numpy as np
a=np.array(a)
a+1
""""""
