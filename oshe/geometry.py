import io
import typing

from eppy.modeleditor import IDF, IDDAlreadySetError
from honeybee.hbsurface import HBSurface
from ladybug_geometry.geometry2d.mesh import Mesh2D
from ladybug_geometry.geometry2d.polygon import Polygon2D
from ladybug_geometry.geometry3d.face import Face3D
from ladybug_geometry.geometry3d.pointvector import Point3D

from oshe.helpers import random_id
from . import material as mat


class Face3D(Face3D):
    def to_hb(self, rad_properties, name, surface_type):
        return HBSurface(sorted_points=self.vertices, name=name, rad_properties=rad_properties,
                         surface_type=surface_type)


class Ground(object):
    def __init__(self, material: mat.MaterialBase, xy: float = 10, depth: float = 1.5, subsurface_size: float = 2):
        self.surface_material = material
        self.underground_material = mat.material_dict["INTERFACE"]
        self.xy = xy
        self.subsurface_size = subsurface_size
        self.depth = depth

        self.generate_faces()

    def generate_faces(self):
        polygon = Polygon2D.from_dict({"type": "Polygon2D",
                                       "vertices": [(-self.xy / 2, -self.xy / 2), (self.xy / 2, -self.xy / 2),
                                                    (self.xy / 2, self.xy / 2), (-self.xy / 2, self.xy / 2)]})
        mesh = Mesh2D.from_polygon_grid(polygon, x_dim=self.subsurface_size, y_dim=self.subsurface_size)
        self.faces_top = [Face3D([Point3D(mesh.vertices[vertex].x, mesh.vertices[vertex].y, 0) for vertex in face]) for
                          face in mesh.faces]

        pt0 = Point3D(-self.xy / 2, -self.xy / 2, 0)
        pt1 = Point3D(self.xy / 2, -self.xy / 2, 0)
        pt2 = Point3D(self.xy / 2, self.xy / 2, 0)
        pt3 = Point3D(-self.xy / 2, self.xy / 2, 0)
        pt4 = Point3D(-self.xy / 2, -self.xy / 2, -self.depth)
        pt5 = Point3D(self.xy / 2, -self.xy / 2, -self.depth)
        pt6 = Point3D(self.xy / 2, self.xy / 2, -self.depth)
        pt7 = Point3D(-self.xy / 2, self.xy / 2, -self.depth)

        self.face_bottom = Face3D([pt7, pt6, pt5, pt4])
        self.face_south = Face3D([pt4, pt5, pt1, pt0])
        self.face_east = Face3D([pt5, pt6, pt2, pt1])
        self.face_north = Face3D([pt6, pt7, pt3, pt2])
        self.face_west = Face3D([pt7, pt4, pt0, pt3])

    def to_hb(self):
        return [i.to_hb(self.surface_material.to_hb(), name="ground_{0:04.0f}".format(n), surface_type=2) for n, i in
                enumerate(self.faces_top)]

    def to_eppy(self, idd_file: str):
        try:
            IDF.setiddname(str(idd_file))
        except IDDAlreadySetError as e:
            pass
        idf = IDF(io.StringIO(""))
        eppy_objects = []

        # Create ground zone object
        zone = idf.newidfobject("ZONE")
        zone.Name = "ground_zone"
        eppy_objects.append(zone)

        # Convert ground surface material to eppy objects
        [eppy_objects.append(i) for i in self.surface_material.to_eppy(idd_file)]

        # Convert ground subsurface material to eppy objects
        [eppy_objects.append(i) for i in self.underground_material.to_eppy(idd_file)]

        # Convert ground geometry to eppy objects
        for n, srf in enumerate(self.faces_top):
            surface_name = "ground_top_{0:03.0f}".format(n)

            ground_surface = idf.newidfobject("BUILDINGSURFACE:DETAILED")
            ground_surface.Name = surface_name
            ground_surface.Surface_Type = "Roof"
            ground_surface.Construction_Name = self.surface_material.name
            ground_surface.Zone_Name = "ground_zone"
            ground_surface.Outside_Boundary_Condition = "Outdoors"
            ground_surface.Sun_Exposure = "Sunexposed"
            ground_surface.Wind_Exposure = "Windexposed"
            for nn, vtx in enumerate(srf.vertices):
                setattr(ground_surface, "Vertex_{}_Xcoordinate".format(nn + 1), vtx.x)
                setattr(ground_surface, "Vertex_{}_Ycoordinate".format(nn + 1), vtx.y)
                setattr(ground_surface, "Vertex_{}_Zcoordinate".format(nn + 1), vtx.z)
            eppy_objects.append(ground_surface)

            # Add ground top surface temperatures to hourly output
            output_variable = idf.newidfobject("OUTPUT:VARIABLE")
            output_variable.Key_Value = surface_name
            output_variable.Variable_Name = "Surface Outside Face Temperature"
            output_variable.Reporting_Frequency = "hourly"
            eppy_objects.append(output_variable)

        # Add ground edges and bottom
        for name, srf in list(zip(*[["north", "east", "south", "west", "bottom"],
                                    [self.face_north, self.face_east, self.face_south, self.face_west,
                                     self.face_bottom]])):
            ground_boundary = idf.newidfobject("BUILDINGSURFACE:DETAILED")
            ground_boundary.Name = "ground_{0:}".format(name)
            ground_boundary.Surface_Type = "Wall" if name != "bottom" else "Floor"
            ground_boundary.Construction_Name = self.underground_material.name
            ground_boundary.Zone_Name = "ground_zone"
            ground_boundary.Outside_Boundary_Condition = "Ground"
            ground_boundary.Sun_Exposure = "Nosun"
            ground_boundary.Wind_Exposure = "NoWind"
            for nn, vtx in enumerate(srf.vertices):
                setattr(ground_boundary, "Vertex_{}_Xcoordinate".format(nn + 1), vtx.x)
                setattr(ground_boundary, "Vertex_{}_Ycoordinate".format(nn + 1), vtx.y)
                setattr(ground_boundary, "Vertex_{}_Zcoordinate".format(nn + 1), vtx.z)
            eppy_objects.append(ground_boundary)

        return eppy_objects


class Shade(object):
    def __init__(self, vertices: typing.List[typing.List[float]] = [[-500, -500, 3], [500, -500, 3], [500, 500, 3],
                                                                    [-500, 500, 3]],
                 material: mat.MaterialBase = mat.material_dict["CONCRETE"]):
        self.vertices = [Point3D(i[0], i[1], i[2]) for i in vertices]
        self.material = material
        self._create_face()

    def _create_face(self):
        self.face = Face3D([Point3D(i[0], i[1], i[2]) for i in self.vertices])

    def to_hb(self):
        return self.face.to_hb(rad_properties=self.material.to_hb(), name="shade", surface_type=6)

    def to_eppy(self, idd_file: str):
        try:
            IDF.setiddname(str(idd_file))
        except IDDAlreadySetError as e:
            pass
        idf = IDF(io.StringIO(""))
        eppy_objects = []

        # Create referencible shade schedule
        shade_schedule_id = random_id()

        # Create shade object
        transmittance_schedule_type_limit = idf.newidfobject("SCHEDULETYPELIMITS")
        transmittance_schedule_type_limit.Name = "shade_schedule_type_limit_{0:}".format(shade_schedule_id)
        transmittance_schedule_type_limit.Lower_Limit_Value = 0
        transmittance_schedule_type_limit.Upper_Limit_Value = 1
        transmittance_schedule_type_limit.Numeric_Type = "Continuous"
        eppy_objects.append(transmittance_schedule_type_limit)

        transmittance_schedule = idf.newidfobject("SCHEDULE:CONSTANT")
        transmittance_schedule.Name = "shade_schedule_constant_{0:}".format(shade_schedule_id)
        transmittance_schedule.Schedule_Type_Limits_Name = "shade_schedule_type_limit_{0:}".format(shade_schedule_id)
        transmittance_schedule.Hourly_Value = 0
        eppy_objects.append(transmittance_schedule)

        shade = idf.newidfobject("SHADING:SITE:DETAILED")
        shade.Name = random_id()
        shade.Transmittance_Schedule_Name = "shade_schedule_constant_{0:}".format(shade_schedule_id)
        shade.Number_of_Vertices = len(self.vertices)
        for nn, vtx in enumerate(self.vertices):
            setattr(shade, "Vertex_{}_Xcoordinate".format(nn + 1), vtx.x)
            setattr(shade, "Vertex_{}_Ycoordinate".format(nn + 1), vtx.y)
            setattr(shade, "Vertex_{}_Zcoordinate".format(nn + 1), vtx.z)
        eppy_objects.append(shade)

        return eppy_objects
