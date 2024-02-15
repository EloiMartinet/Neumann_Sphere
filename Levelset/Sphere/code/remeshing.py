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
    """
    Remeshs with respect to the level set function
    """
    if mesh_out == None:
        mesh_out = mesh

    proc = subprocess.Popen(["mmgs_O3  {msh} -sol {sol} -ls {ls} -out {out} -hmax {hmax} -hmin {hmin} -nr -v -1".format(msh=mesh,sol=sol,ls=level,hmin=hmin,hmax=hmax,out=mesh_out)],shell=True,stdout=open(os.devnull, 'w'))
    #proc = subprocess.Popen(["mmgs_O3  {msh} -sol {sol} -ls {ls} -hgrad 1.1 -hmin {hmin} -hmax {hmax} -out {out} -nr -v -1".format(msh=mesh,sol=sol,ls=level,hmin=hmin,hmax=hmax,out=mesh_out)],shell=True,stdout=None)
    #proc = subprocess.Popen(["mmgs_O3  {msh} -sol {sol} -ls {ls} -hgrad 1.1 -hmax {hmax} -out {out}".format(msh=mesh,sol=sol,ls=level,hmax=hmax,out=mesh_out)],shell=True,stdout=log)
    proc.wait()

def advect_surface(mesh,sol,vel,step,solout,log) :
    """ Advect the domain defined by the LS sol along velocity vel and put the result in res
        (Fonction de Charles légèrement modifiée)"""

    # Advection by calling advect
    proc = subprocess.Popen(["Advection {msh} -s {vit} -c {chi} -dt {dt} -o {out} -surf -nocfl +v".format(msh=mesh,vit=vel,chi=sol,dt=step,out=solout)],shell=True,stdout=open(os.devnull, 'w'))
    proc.wait()

def dist_subdom_surface(mesh,log) :
    """ Create the distance function to a subdomain the ls function provided in the .sol associated
    to the mesh file. WARNING : overwrite the .sol file """

    # Redistancing
    # We use the flag -fmm to use the fast marching method (the only that works on surface right now)
    proc = subprocess.Popen(["mshdist {msh} -surf -fmm -dom -ncpu 4".format(msh=mesh)],shell=True,stdout=open(os.devnull, 'w'))
    proc.wait()

def compute_normal(V, gf_ls, eps=1e-5):
    #
    # Compute the normal field n(x) = ∇ls(x)/|∇ls(x)| with a regularization term
    # gf_grad_ls is supposed to be of GetFEM type, i.e. of size (dim, nbdof) i.e.
    # gf_grad_ls[:,i] is the gradient at dof(i)
    #
    gf_grad_ls = gf.compute_gradient(V, gf_ls, V)
    norms_square = np.square(np.linalg.norm(gf_grad_ls, axis=0))
    return gf_grad_ls/np.sqrt(norms_square+eps*eps)

def hilbertian_extension(IM, V, scalar_velocity, ls, boundary_region=BOUNDARY_REGION, alpha=1e-1):
    #
    # Hilbertian regularization and extension procedure
    # that allows to extend the field defined on the meshFEM mesh_fem_from
    # to mesh_fem_to in a smooth way depending on a smoothing parameter alpha
    #

    # Define the model to solve the extension
    md = gf.Model("real")
    md.add_fem_variable("vto", V) # The extended field
    #ADD FILTERED FEM DATA POUR VFROM ? CA EXISTE ?
    #md.add_filtered_fem_variable("vfrom", V, boundary_region) # The original field
    #md.set_variable("vfrom", scalar_velocity[V.basic_dof_on_region(boundary_region)])
    md.add_fem_data("vfrom", V, 1) # The original field
    md.set_variable("vfrom", scalar_velocity)
    md.add_data("alpha", 1)
    md.set_variable("alpha", alpha)
    md.add_linear_term(IM, "alpha * Grad_vto.Grad_Test_vto + vto*Test_vto")
    md.add_Dirichlet_condition_with_penalization(IM,"vto", 1e10, BOUNDARY_REGION, dataname="vfrom")
    md.solve()

    # Compute the normal field
    n = compute_normal(V, ls)

    #print(md.variable("vto"))
    velocity = md.variable("vto")*n

    return velocity

def neumann_ev(k, max_multiplicity, V, IM, region=-1, v0=None):
    """
    Compute the neumann eigenvalues and eigenvectors associated to ls (given in the GetFEM order)
    """
    # Build the partial mesh fem
    #V.linked_mesh().region_merge(region, BOUNDARY_REGION)

    # Assemble the stiffness and mass matrices
    md = gf.Model("real")
    #md.add_fem_variable("u", partial_V)
    md.add_filtered_fem_variable("u", V, region)
    Kg = gf.asm_generic(IM, 2, "Grad_Test2_u.Grad_Test_u", region, md)
    Mg = gf.asm_generic(IM, 2, "Test2_u*Test_u", region, md)

    # Convert to scipy sparse (TODO:vraiment utile ? voir getfem eigen)
    K = getfem_to_scipy_sparse(Kg)
    M = getfem_to_scipy_sparse(Mg)

    # Compute the eigenvalues
    eVal, partial_eVec = eigsh(K, k+max_multiplicity+1, M, sigma=0, which='LM', v0=v0)

    # Sort it by crescent order if they aren't
    permutation = eVal.argsort()
    eVal = eVal[permutation]
    partial_eVec = partial_eVec[:,permutation]

    eVal = eVal[k:k+max_multiplicity]
    partial_eVec = partial_eVec[:,k:k+max_multiplicity]

    return eVal, partial_eVec

def total_mass(IM, V, region=-1):
    return  gf.asm_generic(IM, 0, '1', region, gf.Model("real"))

def gradient_MU(MU, U, V):
    """
    Compute the shape gradient of the k-th eigenvalue
    k is the eigenvalue to optimize
    alpha is some weight applied to the volume constraint
    traget_vol is the target volume
    """

    dMu = np.zeros(np.shape(U))
    _, multiplicity = np.shape(dMu)

    md = gf.Model("real")
    md.add_fem_data("u", V, 1)
    md.add_data("mu", 1)

    for i in range(multiplicity):
        md.set_variable("u", U[:,i])
        md.set_variable("mu", MU[i])
        dMu[:,i] = md.interpolation("Grad_u.Grad_u-mu*u*u", V, BOUNDARY_REGION)

    return dMu

def shape_gradient(MU, U, IM, V, b, target_vol=1., p=50, v0=None):
    #
    # Compute the (scalar) shape gradient of the k-th eigenvalue
    # k is the eigenvalue to optimize
    # alpha is some weight applied to the volume constraint
    # traget_vol is the target volume
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
    vol_domain = total_mass(IM, V)

    #vel = vol_domain*theta1+MU[0] - 2*alpha*(vol_domain-target_vol) - 2*beta*theta2
    vel = vol_domain*grad+MU[0] - b*(vol_domain-target_vol)

    return vel

def renormalize(p):
    #
    # Used to replace the points mesh on the sphere after
    # a large number of iteration
    #
    return p/np.linalg.norm(p, axis=0)

def main():

    # Problem parameters
    mesh_path = "init.mesh" # Obtained by the ersatz procedure
    tmp_folder = "results" # The output folder
    k = 1                   # The eigenvalue to optimize
    max_multiplicity = 3    # Maximal expected multiplicity of the eigenvalue
    target_vol = 9.0        # The target volume (actually m' in the paper)
    nIter = 150             # Total number of iterations
    hmax = 5e-2             # The maximal mesh size
    hmin = 1e-3             # The minimal mesh size
    b = 5                   # Penalty coefficient for the mass constraint
    epsilon = 1             # Below that theshold, eigenvalues are considered multiple
    p = 20                  # Regularization parameter for the eigenvalue
    step = 1e-3             # The initial step size
    step_decrease = 0.5     # For basic "linesearch" procedure
    step_increase = 1.1     # If the previous iteration was succesful, increase the step by this amount
    out_file_mesh = "best.mesh" # The mesh where the best result is stored

    # Build levelset function
    medit_mesh = MeditS().load(mesh_path)
    dist_subdom_surface(mesh_path, log=None)
    ls = medit_mesh.load_solution(mesh_path[:-5]+".sol")
    medit_mesh.save(f"{tmp_folder}/ls.0.mesh").save_solution(ls, f"{tmp_folder}/ls.0.sol")

    obj = [0]   # The history of the objective

    for i in range(1,nIter+1):
        print(f"\n\nIteration n°{i} :")
        
        # The behavior of MMG has changed since the preprint. Need to delete the marked edges 
        # (the 0 level set at each remeshing) or MMG will keep them and this will result in too thin triangles
        medit_mesh.set_all_triangles_labels_to(0)   # Also drop the regions otherwise MMG automatically add an edge between them
        medit_mesh.delete_edges() 
        medit_mesh.save(f"{tmp_folder}/tmp.mesh")

        # Using MMG tools to remesh
        print("Remeshing")
        mmgs(f"{tmp_folder}/tmp.mesh", f"{tmp_folder}/ls.{i-1}.sol",
             mesh_out=f"{tmp_folder}/ls.{i}.mesh", hmin=hmin, hmax=hmax)
        medit_mesh = MeditS().load(f"{tmp_folder}/ls.{i}.mesh") # We reload the new mesh

        medit_mesh.vertices = renormalize(medit_mesh.vertices)  # The sphere tends to bulge after a lot of remeshing
        medit_mesh.save(f"{tmp_folder}/ls.{i}.mesh")            # Probably totally useless
        mesh = medit_mesh.to_getfem()

        # Defining integration method
        IM = gf.MeshIm(mesh, gf.Integ("IM_TRIANGLE(10)"))

        # Defining the FE space
        V = gf.MeshFem(mesh, 1)
        V.set_fem(gf.Fem("FEM_PK(2, 1)"))

        submesh_medit = medit_mesh.extract_region(INNER_REGION)
        submesh = submesh_medit.to_getfem()

        V_submesh = gf.MeshFem(submesh, 1)
        V_submesh.set_fem(gf.Fem("FEM_PK(2, 1)"))
        IM_submesh = gf.MeshIm(submesh, gf.Integ("IM_TRIANGLE(10)"))

        # For interpolation purpose
        _, dof_to_medit_indices = KDTree(V.basic_dof_nodes().T, leafsize=30).query(medit_mesh.vertices.T) # To send the dof of V to the vertices of the medit mesh
        _, medit_to_dof_indices = KDTree(medit_mesh.vertices.T, leafsize=30).query(V.basic_dof_nodes().T) # The opposite
        # This one send the dof of the boundary
        _, dof_to_submesh_boundary_dof_indices = KDTree(V.basic_dof_nodes().T,leafsize=30).query(V_submesh.basic_dof_nodes()[:,V_submesh.basic_dof_on_region(BOUNDARY_REGION)].T)

        # Recomputing the levelset function
        dist_subdom_surface(f"{tmp_folder}/ls.{i}.mesh", log=None)
        ls = medit_mesh.load_solution(f"{tmp_folder}/ls.{i}.sol")
        gf_ls = ls[medit_to_dof_indices]

        # Compute the eigen elements
        print("Computing the eigen elements")
        eVal, eVec = neumann_ev(k, max_multiplicity, V_submesh, IM_submesh, region=-1)

        # Get the multiplicity
        multiplicity = np.count_nonzero(eVal[0:] - eVal[0] <= epsilon)
        multiplicity = min(multiplicity, max_multiplicity)
        MU, U = eVal[0:multiplicity], eVec[:,0:multiplicity]

        # If it's the best candidate so far we save it to a file
        mass = total_mass(IM_submesh,V_submesh)
        approx_min = np.power(np.sum(np.power(MU, -p)), -1/p)
        current_obj = mass*approx_min - 0.5*b*(mass-target_vol)**2


        # Check if the objective increased
        if current_obj > obj[-1]:
            obj.append(current_obj)
            medit_mesh.save(out_file_mesh)
            step *= step_increase
        # Otherwise we decrease the step
        else :
            step *= step_decrease
            print("Decreasing the step.")

        # If the step is too small we break
        if step < 1e-6:
            break

        # If there is no boundary (i.e. our set is the whole sphere or empty) we break
        if np.where(medit_mesh.vertices_labels == BOUNDARY_REGION)[0].size == 0:
            break

        # This means that the domain is disconnected with more than k connected components
        if MU[0] < 1e-14 :
            print(f"Disconnected")
            break

        # Compute the shape gradient
        partial_gf_scalar_velocity = shape_gradient(MU, U, IM_submesh, V_submesh, b, target_vol, p=p)

        # Extend to the whole mesh
        print("Extension-regularization")
        gf_scalar_velocity = np.zeros(V.nbdof())
        gf_scalar_velocity[dof_to_submesh_boundary_dof_indices] = partial_gf_scalar_velocity[V_submesh.basic_dof_on_region(BOUNDARY_REGION)]
        gf_velocity = hilbertian_extension(IM, V, gf_scalar_velocity, gf_ls, alpha=0.1)

        # Send the dof of GetFEM to the points of medit and save it
        velocity = gf_velocity[:,dof_to_medit_indices]
        medit_mesh.save(f"{tmp_folder}/vel.{i}.mesh").save_solution(velocity, f"{tmp_folder}/vel.{i}.sol")

        # Advect
        print("Advection")
        dt = step/hmax
        advect_surface(f"{tmp_folder}/ls.{i}.mesh", f"{tmp_folder}/ls.{i}.sol",
                      f"{tmp_folder}/vel.{i}.sol",dt,f"{tmp_folder}/ls.{i}.sol",None)

        # Printing the computed stuff
        print(f"\nstep = {step}")
        print(f"Best objective = {obj[-1]}")
        print(f"Current objective = {current_obj}")
        print(f"|Ω| = {mass}")
        print(f"μ(Ω) = {MU[0]}")



if __name__=="__main__":
    main()
