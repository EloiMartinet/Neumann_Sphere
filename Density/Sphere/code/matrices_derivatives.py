import sys
sys.path.insert(0, "/home/eloi/")       # Add getfem to pythonpath
import getfem as gf
import numpy as np
from scipy import sparse                # Sparse matrices type
from tqdm import tqdm                   # Progress bar in console. Can be omited
import pickle                           # Serialization of assembled matrices

class MySparse(object):

    def __init__(self):
        self.I = np.array([],dtype=np.int)
        self.J = np.array([],dtype=np.int)
        self.V = np.array([],dtype=np.float64)
        self.N = np.array([],dtype=np.int)
        self.size = 0

    def append(self, I, J, V, n):
        """
        appends a new sparse matrix at the en of the list
        """
        N = np.full(shape=I.size, fill_value=n,dtype=np.int)
        self.I = np.concatenate((self.I, I))
        self.J = np.concatenate((self.J, J))
        self.V = np.concatenate((self.V, V))
        self.N = np.concatenate((self.N, N)) # The index of dof
        self.size = self.I.size

def get_I_J_val(M):
    # Getfem stores its matrices like (data, indices, indptr)
    # whereas we would to have (i, j, data). We use scipy.
    # cf https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html
    J, I =  gf.Spmat.get(M, "csc_ind")
    val = gf.Spmat.get(M, "csc_val")

    Mscipy = sparse.csc_matrix((val, I, J))
    I, J, val = sparse.find(Mscipy)
    return I, J, val

def compute_matrices_derivatives(mesh_path, out_path):

    # Mesh generation
    mesh = gf.Mesh('load', mesh_path)

    # Defining the continuous FE (for the eigenfunctions)
    V = gf.MeshFem(mesh, 1)
    V.set_fem(gf.Fem("FEM_PK(2, 1)")) # 2 for the (geometric) dimension of the mesh, 1 for the order of the element

    # Defining the (dis)continuous FE (for the density)
    VD = gf.MeshFem(mesh, 1)
    VD.set_fem(gf.Fem("FEM_PK(2, 1)")) # 2 for the dimension of the mesh, 1 for the order of the element
    nbdof_VD = gf.MeshFem.get(VD, "nbdof") # Should be equal to the number of triangles in the mesh

    # Defining the integration method (will be exact if the order is not lower than the element degree)
    IM = gf.MeshIm(mesh,5)

    # The total derivative matrix and triangle areas
    DK, DM = MySparse(), MySparse()
    tri_area = np.zeros(nbdof_VD)

    # Creation of the model
    md = gf.Model("real")
    md.add_fem_variable("u", V)
    md.add_fem_data("drho", VD, 1)

    for i in tqdm(range(nbdof_VD)):
        # We set drho as the characteristic function of the i-th triangle
        drho = np.zeros(nbdof_VD)
        drho[i] = 1
        md.set_variable("drho", drho)

        # generic assembly of the derivative of the stiffness matrix
        dKg = gf.asm("generic", IM, 2, "drho * Grad_Test2_u . Grad_Test_u", -1, md)
        I, J, val = get_I_J_val(dKg)
        DK.append(I,J,val,i)

        # generic assembly of the derivative of the stiffness matrix
        dMg = gf.asm("generic", IM, 2, "drho * Test2_u * Test_u", -1, md)
        I, J, val = get_I_J_val(dMg)
        DM.append(I,J,val,i)

        # generic assembly of the elements area
        Tg = gf.asm("generic", IM, 0, "drho", -1, md)
        tri_area[i] = Tg

    # Pickle the matrices
    dict = { 'DK' : DK, 'DM' : DM, 'tri_area' : tri_area }
    outfile = open(out_path,'wb')
    pickle.dump(dict,outfile)
    outfile.close()


if __name__=="__main__":
    compute_matrices_derivatives(sys.argv[1], sys.argv[2])
