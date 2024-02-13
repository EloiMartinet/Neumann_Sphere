import sys
sys.path.insert(0, "/home/eloi/")       # Add getfem to pythonpath
import subprocess                       # To use advection, mmg, mshdist...which are command line
import os                               # Same
import getfem as gf                     # FEM solver
import numpy as np                      # I'm sure you know this one ;)
from scipy.sparse.linalg import eigsh   # Computation of sparse hermitian matrices eigenvalues
from medit_to_getfem import *           # Medit <> GetFEM utiliy functions (mesh conversions basically)

# Shut GetFEM up
gf.util_trace_level(level=0)

# The regions used by MMG
INNER_REGION = 3
OUTER_REGION = 2
BOUNDARY_REGION = 10

def mmgs(mesh, sol, level=0.0, hmin=1e-5, hmax=1e-2, mesh_out=None, log=None):
    #
    #   Remeshs with respect to the level set function
    #
    if mesh_out == None:
        mesh_out = mesh

    proc = subprocess.Popen(["mmgs_O3  {msh} -sol {sol} -ls {ls} -out {out} -hmax {hmax} -hmin {hmin} -nr -v -1".format(msh=mesh,sol=sol,ls=level,hmin=hmin,hmax=hmax,out=mesh_out)],shell=True,stdout=open(os.devnull, 'w'))
    proc.wait()

def advect_surface(mesh,sol,vel,step,solout,log) :
    #
    #   Advect the domain defined by the LS sol along velocity vel and put the result in res
    #
    proc = subprocess.Popen(["Advection {msh} -s {vit} -c {chi} -dt {dt} -o {out} -surf -nocfl +v".format(msh=mesh,vit=vel,chi=sol,dt=step,out=solout)],shell=True,stdout=open(os.devnull, 'w'))
    proc.wait()

def dist_subdom_surface(mesh,log) :
    #
    #   Create the distance function to a subdomain the ls function provided 
    #   in the .sol associated to the mesh file. 
    #   WARNING : overwrite the .sol file
    #

    # We use the flag -fmm to use the fast marching method (the only that works on surface right now)
    proc = subprocess.Popen(["mshdist {msh} -surf -fmm -dom -ncpu 4".format(msh=mesh)],shell=True,stdout=open(os.devnull, 'w'))
    proc.wait()

def characteristic(ls, eps=1e-5):
    #
    #   Compute the approximate characteristic function of {ls < 0}
    #
    return 0.5 * (1 - ls/np.sqrt(np.square(ls)+eps*eps))

def neumann_ev_ls(k, max_multiplicity, V, IM, ls, eps=1e-5, v0=None):
    #
    #   Compute the neumann eigenvalues and eigenvectors associated to ls (given in the GetFEM order)
    #

    # Assemble the stiffness and mass matrices
    md = gf.Model("real")
    md.add_fem_variable("u", V)
    md.add_fem_data("chi", V)
    md.set_variable("chi", characteristic(ls)) # The characteristic function of the set
    md.add_data("eps", 1)
    md.set_variable("eps", eps)
    Kg = gf.asm_generic(IM, 2, "(chi+eps)*Grad_Test2_u.Grad_Test_u", -1, md)
    Mg = gf.asm_generic(IM, 2, "(chi+eps*eps)*Test2_u*Test_u", -1, md)

    # Convert to scipy sparse
    K = getfem_to_scipy_sparse(Kg)
    M = getfem_to_scipy_sparse(Mg)

    # Compute the eigenvalues
    eVal, eVec = eigsh(K, k+max_multiplicity+1, M, sigma=0, which='LM', v0=v0)

    # Sort it by crescent order if they aren't
    permutation = eVal.argsort()
    eVal = eVal[permutation]
    eVec = eVec[:,permutation]

    print(eVal)
    eVal = eVal[k:k+max_multiplicity]
    eVec = eVec[:,k:k+max_multiplicity]

    return eVal, eVec

def total_mass_ls(IM, V, ls):
    #
    #   The surface of the set
    #
    md = gf.Model("real")
    md.add_fem_data("chi", V)
    chi = np.zeros(ls.shape)
    chi[ls<0] = 1
    md.set_variable("chi", chi) # The characteristic function of the set

    return  gf.asm_generic(IM, 0, 'chi', -1, md)

def gradient_MU(MU, U, V):
    #
    #   Compute the shape gradient of the k-th eigenvalue
    #
    dMu = np.zeros(np.shape(U))
    _, multiplicity = np.shape(dMu)

    md = gf.Model("real")
    md.add_fem_data("u", V, 1)
    md.add_data("mu", 1)

    for i in range(multiplicity):
        md.set_variable("u", U[:,i])
        md.set_variable("mu", MU[i])
        dMu[:,i] = md.interpolation("Grad_u.Grad_u-mu*u*u", V)


    return dMu

def shape_gradient_ls(MU, U, IM, V, ls, b=1, target_vol=1., epsilon=1e-2, p=20, v0=None):
    #
    # Compute the (scalar) shape gradient of the k-th eigenvalue
    # k is the eigenvalue to optimize
    # alpha is some weight applied to the volume constraint
    # traget_vol is the target volume
    # epsilon is the threshold under which eigenvalues are supposed to be multiple
    #
    dMu = gradient_MU(MU, U, V)
    nbdof, multiplicity = np.shape(dMu)

    # Tderivative of the regularized gradient
    grad = np.zeros(nbdof)

    for i in range(multiplicity):
        grad += np.power(MU[i], -(p+1))*dMu[:,i]

    obj = np.power(np.sum(np.power(MU, -p)), -1/p)
    grad = np.power(obj, p+1)*grad

    # Compute the volume of the domain by integrating the characteristic function
    vol_domain = total_mass_ls(IM, V, ls)

    # Initialize b if it is not
    if b is None :
        b = infty_norm(vol_domain*theta+MU[0])

    # Compute the extended normal vector
    grad_ls = gf.compute_gradient(V, ls, V)
    norms_square = np.square(np.linalg.norm(grad_ls, axis=0))
    n = grad_ls/np.sqrt(norms_square+1e-5)
    vel = vol_domain*grad+MU[0] - b*(vol_domain-target_vol)
    vel = vel*n

    print("|Ω| = {}".format(vol_domain))
    print("μ(Ω) = "+str(MU))

    return vel

def init_ls_random(p, m=3, n=3):
    #
    #   Initialize the level set function with a random truncated Fourier series
    #
    _ , nb_dof = np.shape(p)
    ls = np.zeros(nb_dof)

    theta = np.arccos(p[2,:])
    phi = np.arccos(p[0,:]/np.linalg.norm(p[0:2,:], axis=0)) # Ou un truc du genre
    inegative = np.where(p[1,:] < 0)
    phi[inegative] = -phi[inegative]

    for j in range(m):
        for k in range(n):
            a = 2*np.random.rand(1)-1
            b = 2*np.random.rand(1)-1
            c = complex(a,b)
            ls += np.real(c*np.exp(1j*(k*theta+j*phi)))

    return ls/np.max(np.abs(ls))

def infty_norm(v):
    #
    # Compute the infinity norm of a velocity field
    #
    return np.max(np.linalg.norm(v, axis=0))

def renormalize(p):
    #
    # Used to replace the points mesh on the sphere after
    # a large number of iteration
    #
    return p/np.linalg.norm(p, axis=0)

def main():

    # Problem parameters
    mesh_path = "meshes/sphere_fine.mesh"   # The mesh we use
    tmp_folder = "results" # The output folder
    k = 1                   # The eigenvalue to optimize
    max_multiplicity = 3    # Maximal expected multiplicity of the eigenvalue
    target_vol = 9.0        # The target volume (actually m' in the paper)
    nIter = 150             # Total number of iterations
    hmax = 5e-2             # The maximal mesh size
    hmin = 1e-3             # The minimal mesh size
    b = 5                   # Penalty coefficient for the mass constraint
    epsilon = 1             # Below that theshold, eigenvalues are considered multiple

    # Load the mesh and init the solution
    medit_mesh = MeditS().load(mesh_path)
    ls = init_ls_random(medit_mesh.vertices, 4, 4)
    medit_mesh.save(f"{tmp_folder}/ls.0.mesh").save_solution(ls, f"{tmp_folder}/ls.0.sol")
    medit_mesh.save(f"{tmp_folder}/chi.0.mesh").save_solution(characteristic(ls), f"{tmp_folder}/chi.0.sol") # Save the characteristic function (for vizualization)

    # Create the getfem mesh
    mesh = medit_mesh.to_getfem()

    # Defining integration method
    IM = gf.MeshIm(mesh, gf.Integ("IM_TRIANGLE(10)"))

    # Defining the FE space
    V = gf.MeshFem(mesh, 1)
    V.set_fem(gf.Fem("FEM_PK(2, 1)"))

    # For interpolation purpose
    _, dof_to_medit_indices = KDTree(V.basic_dof_nodes().T, leafsize=30).query(medit_mesh.vertices.T) # To send the dof of V to the vertices of the medit mesh
    _, medit_to_dof_indices = KDTree(medit_mesh.vertices.T, leafsize=30).query(V.basic_dof_nodes().T) # The opposite

    for i in range(1,nIter+1):
        print(f"\n\nIteration n°{i} :")

        if i%20 == 0:
            # Regularizing the levelset function
            ls = medit_mesh.load_solution(f"{tmp_folder}/ls.{i-1}.sol")
            gf_ls = ls[medit_to_dof_indices]
            gf_ls = heat_equation(gf_ls, V, IM, 1e-3, step=1)
            ls = gf_ls[dof_to_medit_indices]
            medit_mesh.save_solution(ls, f"{tmp_folder}/ls.{i-1}.sol")

            # Using MMG tools to remesh
            mmgs(f"{tmp_folder}/ls.{i-1}.mesh", f"{tmp_folder}/ls.{i-1}.sol",
                 mesh_out=f"{tmp_folder}/ls.{i}.mesh", hmin=hmin, hmax=hmax)

            # We reload the new mesh
            medit_mesh = MeditS().load(f"{tmp_folder}/ls.{i}.mesh")
            medit_mesh.vertices = renormalize(medit_mesh.vertices)  # The sphere tends to bulge after a lot of remeshing
            medit_mesh.delete_edges() # The behavior of MMG has changed since the preprint. Need to delete the marked edges (the 0 level set at each remeshing) or MMG will keep them and this will result in too thin triangles
            medit_mesh.save(f"{tmp_folder}/ls.{i}.mesh")
            mesh = medit_mesh.to_getfem()

            # Defining integration method
            IM = gf.MeshIm(mesh, gf.Integ("IM_TRIANGLE(10)"))

            # Defining the FE space
            V = gf.MeshFem(mesh, 1)
            V.set_fem(gf.Fem("FEM_PK(2,1)"))

            # For interpolation purpose
            _, dof_to_medit_indices = KDTree(V.basic_dof_nodes().T, leafsize=30).query(medit_mesh.vertices.T) # To send the dof of V to the vertices of the medit mesh
            _, medit_to_dof_indices = KDTree(medit_mesh.vertices.T, leafsize=30).query(V.basic_dof_nodes().T) # The opposite

            # Recomputing the levelset function
            dist_subdom_surface(f"{tmp_folder}/ls.{i}.mesh", log=None)
            ls = medit_mesh.load_solution(f"{tmp_folder}/ls.{i}.sol")
            gf_ls = ls[medit_to_dof_indices]


        else:
            ls = medit_mesh.load_solution(f"{tmp_folder}/ls.{i-1}.sol")
            gf_ls = ls[medit_to_dof_indices]

        # Compute the eigen elements
        print("Compute eigenvalues")
        eVal, eVec = neumann_ev_ls(k, max_multiplicity, V, IM, gf_ls, eps=1e-4)

        # Get the multiplicity
        multiplicity = np.count_nonzero(eVal[0:] - eVal[0] <= epsilon)
        multiplicity = min(multiplicity, max_multiplicity)
        MU, U = eVal[0:multiplicity], eVec[:,0:multiplicity]

        # Compute the velocity
        print("Compute velocity field")
        gf_velocity = shape_gradient_ls(MU, U, IM, V, gf_ls, b, target_vol)
        velocity = gf_velocity[:,dof_to_medit_indices]

        # Save the velocity, level set and characteristic function of the set (for visualization purposes)
        medit_mesh.save(f"{tmp_folder}/vel.{i}.mesh").save_solution(velocity, f"{tmp_folder}/vel.{i}.sol")
        medit_mesh.save(f"{tmp_folder}/ls.{i}.mesh").save_solution(ls, f"{tmp_folder}/ls.{i}.sol")
        medit_mesh.save(f"{tmp_folder}/chi.{i}.mesh").save_solution(characteristic(ls), f"{tmp_folder}/chi.{i}.sol")

        # Advect
        dt = 4*hmax/infty_norm(velocity)
        print("Advect surface")
        advect_surface(f"{tmp_folder}/ls.{i}.mesh", f"{tmp_folder}/ls.{i}.sol",
                       f"{tmp_folder}/vel.{i}.sol", dt, f"{tmp_folder}/ls.{i}.sol",None)


if __name__=="__main__":
    main()