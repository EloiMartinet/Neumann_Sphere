import sys
sys.path.insert(0, "/home/eloi/")       # Add getfem to pythonpath
import os                               # To use MMG which is command-line
import getfem as gf                     # FEM solver
import numpy as np                      # I'm sure you know this one ;)
import matplotlib.pyplot as plt         # Data vizualization
from scipy.sparse.linalg import eigsh   # Computation of sparse hermitian matrices eigenvalues
import cyipopt                          # Interior point method library
from medit_to_getfem import *           # To convert getfem mesh to medit format for vizualization
import pickle                           # Serialization of assembled matrices


class MySparse(object):
    #
    #   A custom sparse matrix data type
    #
    def __init__(self):
        self.I = np.array([],dtype=np.int)
        self.J = np.array([],dtype=np.int)
        self.V = np.array([],dtype=np.float64)
        self.N = np.array([],dtype=np.int)

    def append(self, I, J, V, n):
        """
        appends a new sparse matrix at the en of the list
        """
        N = np.full(shape=I.size, fill_value=n,dtype=np.int)
        self.I = np.concatenate((self.I, I))
        self.J = np.concatenate((self.J, J))
        self.V = np.concatenate((self.V, V))
        self.N = np.concatenate((self.N, N)) # The index of dof

class Neumann_regularized(object):
    #
    #   Handles the multiplicity of the eigenvalues by regularizing the min
    #   in max_\rho min_i \{ \mu_i(\rho) \}
    #
    def __init__(self, mesh_path, derivative_path, out_folder="Results/test/", k=1, multiplicity=1, p=20, epsilon = 1e-2, target_mass=5):
        # The threshold to guarantee ellipticity
        self.epsilon = epsilon

        # We set the parameters needed for optimization
        self.k = k
        self.multiplicity = multiplicity
        self.p = p # The exponent of regularization
        self.target_mass = target_mass
        self.history = []

        # We load the derivatives of the matrices
        infile = open(derivative_path,'rb')
        dict = pickle.load(infile)
        infile.close()

        self.DK = dict['DK']
        self.DM = dict['DM']
        self.tri_area = dict['tri_area'] # NOTE THAT IT's NO LONGER THE TRIANGLE AREA, it's the derivative wrt rho_i

        # Mesh import
        self.mesh = gf.Mesh('load', mesh_path)

        # Defining the continuous FE (for the eigenfunctions)
        self.V = gf.MeshFem(self.mesh, 1)
        self.V.set_fem(gf.Fem("FEM_PK(2, 1)")) # 2 for the dimension of the mesh, 1 for the order of the element
        self.nbdof_V = gf.MeshFem.get(self.V, "nbdof")

        # Defining the discontinuous FE (for the density)
        self.VD = gf.MeshFem(self.mesh, 1)
        self.VD.set_fem(gf.Fem("FEM_PK(2, 1)")) # 2 for the dimension of the mesh, 1 for the order of the element
        self.nbdof_VD = gf.MeshFem.get(self.VD, "nbdof") # Should be equal to the number of edges in the mesh

        # Defining the integration method (will be exact if the order is not lower than the element degree)
        self.IM = gf.MeshIm(self.mesh, 10)

        # For the constraint
        self.tri_area = gf.asm_generic(self.IM, 1, 'one', -1,
                           'one', 1, self.VD, np.ones(self.nbdof_VD))

        # We store the ancient rho, mu, u...to avoid recomputing
        self.current_rho = None
        self.current_MU = None
        self.current_U = None

        # Setting the output folder and create it if it doesn't exists
        self.out_folder = out_folder
        if not os.path.exists(out_folder) :
            os.makedirs(out_folder)
            print("Creation of the output directory.")

        self.values_file = open(out_folder+"/values.txt", 'w')

        # Define the model
        self.md = gf.Model("real")
        self.md.add_fem_variable("u", self.V)
        self.md.add_fem_data("rho", self.VD, 1)
        self.md.add_data("epsilon", 1)
        self.md.set_variable("epsilon", self.epsilon)

        # For vizualization purposes (export as medit file)
        self.medit_mesh = MeditS().from_getfem(self.mesh)
        _, self.dof_to_medit_indices = KDTree(self.VD.basic_dof_nodes().T, leafsize=30).query(self.medit_mesh.vertices.T) # To send the dof of V to the vertices of the medit mesh

    def eigen(self, rho):
        #
        #   Compute the eigen elements of the problem
        #   rho : the current density
        #

        # We don't recompute if we are at the same rho
        if np.array_equal(self.current_rho, rho):
            return self.current_MU, self.current_U

        # Update the value of rho_func
        self.md.set_variable("rho", rho)

        # Assemble the stiffness and load matrices
        Kg = gf.asm("generic", self.IM, 2, "(rho+epsilon) * Grad_u.Grad_Test_u", -1, self.md)
        Mg = gf.asm("generic", self.IM, 2, "(rho+epsilon*epsilon) * u*Test_u", -1, self.md)

        # Convert to scipy sparse (TODO:vraiment utile ? voir getfem eigen)
        K = getfem_to_scipy_sparse(Kg)
        M = getfem_to_scipy_sparse(Mg)

        # Compute the eigenvalues
        eVal, eVec = eigsh(K, self.k+self.multiplicity+4, M, sigma=0.0, which='LM')

        # Sort it by crescent order if they aren't
        permutation = eVal.argsort()
        eVal = eVal[permutation]
        eVec = eVec[:,permutation]

        # Eliminate the k-1 first ones
        eVal = eVal[self.k:self.k+self.multiplicity]
        eVec = eVec[:,self.k:self.k+self.multiplicity]

        # Save the current values to avoid multiple recomputing
        self.current_rho = rho
        self.current_MU = eVal
        self.current_U = eVec

        return eVal, eVec

    def gradient_Mu(self, rho, multiplicity):
        #
        #   Compute the gradient of mu_i w.r.t. rho
        #

        # We initialize dMu
        dMu = np.zeros((multiplicity, self.nbdof_VD))

        # We compute the eigenvalue and it's associated eigenvector
        MU, U = self.eigen(rho)

        # We compute dMu
        for k in range(multiplicity):
            M = U[self.DK.I,k]*U[self.DK.J,k]*self.DK.V
            for l in range(self.DK.size):
                dMu[k, self.DK.N[l]] += M[l]
            M = MU[k]*U[self.DM.I,k]*U[self.DM.J,k]*self.DM.V
            for l in range(self.DM.size):
                dMu[k, self.DM.N[l]] -= M[l]

        return dMu

    def objective(self, rho):
        #
        # The callback for calculating the objective. wE PUT A MINUS TO MAXIMIZE
        #
        MU, _ = self.eigen(rho)

        actual_multiplicity = self.multiplicity #np.count_nonzero(MU - MU[0] <= self.epsilon)

        return np.power(np.sum(np.power(MU, -self.p)), -1/self.p)

    def gradient(self, rho):
        #
        # The callback for calculating the gradient
        #
        MU, _ = self.eigen(rho)

        actual_multiplicity = self.multiplicity
        self.history.append(MU[0])

        dMu = self.gradient_Mu(rho, actual_multiplicity)

        grad = np.zeros(self.nbdof_VD)

        for i in range(actual_multiplicity):
            grad += np.power(MU[i], -(self.p+1))*dMu[i]

        return np.power(self.objective(rho), self.p+1)*grad

    def constraints(self, rho):
        #
        # The constraints computation
        #
        return gf.asm_generic(self.IM, 0, 'rho', -1, 'rho', 1, self.VD, rho)

    def jacobian(self, rho):
        #
        # The Jacobian of the constraints
        #
        return self.tri_area

    def intermediate(self,alg_mod,iter_count,obj_value,inf_pr,inf_du,mu,d_norm,regularization_size,alpha_du,alpha_pr,ls_trials):
        #
        #   Plots and save a function defined on VD
        #   rho is a numpy vector of size nbdof_VD
        #
        medit_rho = self.current_rho[self.dof_to_medit_indices]
        self.medit_mesh.save(self.out_folder+f"/rho.{iter_count}.mesh")
        self.medit_mesh.save_solution(medit_rho, self.out_folder+f"/rho.{iter_count}.sol")

def main():
    #
    #   Computes the optimal density using the regularized eigenvalue method
    #
    mesh_path = "meshes/sphere_fine"
    derivative_path = "derivatives/sphere_fine"
    k = 1
    target_mass = 3.0
    epsilon = 1e-4        # The elliptic threshold
    folder = "results/" # meshes output
    nIter = 300
    max_mutiplicity = 3

    # Define the prolem
    pb_regularized = Neumann_regularized(mesh_path, derivative_path, out_folder=folder, k=k,
                 multiplicity=max_mutiplicity, p=20, epsilon=epsilon, target_mass=target_mass)
    dim = pb_regularized.nbdof_VD # Dimension of the problem

    # Define the initial density (will be the same for every problem)
    rho0 = np.random.rand(dim)

    # Lower and upper bounds for rho (element wise)
    lb = np.zeros(dim)
    ub = np.ones(dim)

    # The value of the equality constraint (resp. the equality of eigenvalues supposed to be multiple, and the mass)
    cl = [target_mass]
    cu = [target_mass]

    # The IpOPT problem
    nlp_regularized = cyipopt.problem(n=dim, m=len(cl), problem_obj=pb_regularized, lb=lb, ub=ub, cl=cl, cu=cu)

    # Different optimization options
    nlp_regularized.addOption('obj_scaling_factor', -1.0)   # Maximizes intead of minimizing
    nlp_regularized.addOption('max_iter', nIter)
    nlp_regularized.addOption('mu_strategy', 'monotone')

    # Problem solving
    nlp_regularized.solve(rho0)

    ################################
    #     Plot the graph of mu     #
    ################################
    plt.plot(pb_regularized.history, label="Regularized eigenvalue")
    plt.show()

if __name__=="__main__":
    main()