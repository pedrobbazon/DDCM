from firedrake import *
from pyop2 import op2

def add_nodes(source, target):
    Vsource = source.function_space()
    Vtarget = target.function_space()
    size = Vtarget.finat_element.space_dimension() * Vtarget.value_size
    kernel_code = f"""
    void add_kernel(PetscScalar *restrict source, PetscScalar *restrict target){{
        for(PetscInt i=0; i<{size}; i++) target[i] += source[0];
    }}
    """
    kernel = op2.Kernel(kernel_code, "add_kernel", requires_zeroed_output_arguments=True)
    op2.par_loop(kernel, source.cell_set,
                 source.dat(op2.READ, source.cell_node_map()),
                 target.dat(op2.INC, target.cell_node_map()))


nx = 2
mesh = UnitSquareMesh(nx, nx, quadrilateral=True)
V = FunctionSpace(mesh, "CG", 1)
Q = FunctionSpace(mesh, "DG", 0)


# compute the weight of each node
w = Function(Q)
w.assign(1)
weights = Function(V)
add_nodes(w, weights)

# source = cell-wise data
source = Function(Q)
# target = node-wise average of the source
target = Function(V)

import numpy
source.dat.data[:] = numpy.arange(len(source.dat.data))
add_nodes(source, target)
#target.dat.data[:] /= weights.dat.data_ro[:]


print(target.dat.data)
