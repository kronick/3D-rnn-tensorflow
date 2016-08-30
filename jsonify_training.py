import numpy as np
from RandomWalker import RandomWalker

walker = RandomWalker("bunny2.ply")
points = walker.walk(10000, relative = False, smooth = 0.8)

SCALE = 100

with open("visualize/walk.js", "w+") as f:
    f.write("pathPoints = [\n")

    points[:,0:3] *= SCALE
    for i, p in enumerate(points):
        f.write("\t[{}, {}, {}, {}, {}, {}]{}\n".format(p[0], p[1], p[2], p[4], p[5], p[6], ',' if i < len(points) else ''))

        norm = np.array((p[4],p[5],p[6]))
        mag = np.sqrt(norm.dot(norm))
        #print mag

    f.write("];")
