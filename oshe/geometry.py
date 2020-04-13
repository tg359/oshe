import numpy as np


class RadianceGeometry(object):
    def __init__(self, rad_files: str):

        self.vertices = None
        self.materials = None
        self.data = {}

        # Construct framework
        materials = []
        vertices = []

        for rad_file in rad_files:
            # Open RAD file containing geometry
            with open(rad_file, "r") as f:
                opq = [i.strip() for i in f.readlines()]

            # For each RAD geometry object, get the vertices describing the polygon and material name
            for geo in chunk(opq[3:], n=4):
                materials.append(geo[0].split(" ")[0])
                vertices.append(np.array(chunk([float(i) for i in geo[-1].split(" ")[1:]], n=3))[:, :2].tolist())

        # Create list of unique materials
        self.materials = list(np.unique(materials))

        # Sort the vertices and materials to populate the dataset
        for ii in self.materials:
            self.data[ii] = []
        for v, m in list(zip(*[vertices, materials])):
            self.data[m].append(list(v))


def chunk(enumerable, n=None):
    enumerable = np.array(enumerable)
    return [list(enumerable[i:i + n]) for i in range(0, enumerable.shape[0], n)]
