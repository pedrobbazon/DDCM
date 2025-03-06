from firedrake import *
from netgen.csg import *
import netgen.gui
geo = CSGeometry()
brick = OrthoBrick(Pnt(0,0,0),Pnt(1,1,1))
geo.Add(brick)
ngmesh = geo.GenerateMesh(maxh=.1)
mesh = Mesh(ngmesh)

def Mark(mesh):
    W = FunctionSpace(mesh, "DG", 0)
    markers = Function(W)
    (x, y, z) = SpatialCoordinate(mesh)
    markers.interpolate(conditional(z > .8, 1, 0))
    return markers

for i in range(2):
    markers = Mark(mesh)
    mesh = mesh.refine_marked_elements(markers)
