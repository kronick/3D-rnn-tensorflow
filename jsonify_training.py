import numpy as np
from RandomWalker import RandomWalker

walker = RandomWalker("bunny2.ply")
points = walker.walk(10000, smooth = 0.8)

with open("walk.js", "w+") as f:
    f.write("pathPoints = [\n")
    last_point = np.array([0,0,0,0])
    for i, p in enumerate(points):
        _p = last_point + p
        f.write("\t[{}, {}, {}]{}\n".format(_p[0], _p[1], _p[2], ',' if i < len(points) else ''))
        last_point = _p

    f.write("];")
