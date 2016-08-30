import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, default=None,
                     help='Input filename')
parser.add_argument('-o', type=str, default=None,
                     help='Output filename')

args = parser.parse_args()

with open(args.i, "r") as infile:
    with open(args.o or "walk.js", "w+") as outfile:
        outfile.write("pathPoints = [\n")

        points = []
        for line in infile:
            point = line.split(" ")
            points.append(point)
                
        for i, p in enumerate(points):
            outfile.write("\t[{}, {}, {}]{}\n".format(p[0], p[1], p[2], ',' if i < len(points) else ''))

        outfile.write("];")