import sys
sys.path.insert(0, "/usr/lib/python3/dist-packages/") # On ajoute le path du dossier ou se trouve getfem à pythonpath
import getfem as gf                     # FEM solver
import numpy as np                      # I'm sure you know this one ;)
#import re                               # For processing strings with regular expressisons
import pickle                           # Serialization of assembled matrices
from scipy.spatial import KDTree
import time
from scipy import sparse                # Sparse matrices type
from tqdm import tqdm

class MeditS:
    #
    # A Surface Medit Wrapper
    #
    def __init__(self):
        #
        # Creates a void object
        #
        self.dimension = None

        self.nb_vertices = None
        self.vertices = None
        self.vertices_labels = None

        self.nb_triangles = None
        self.triangles = None
        self.triangles_labels = None

        self.nb_edges = None
        self.edges = None
        self.edges_labels = None

    def load(self, filename):
        #
        # Reads and parse the Medit .mesh file (WITHOUT ANY CONTROL)
        #

        # Open the medit file and convert to an array of strings
        content = open(filename, "r").read().split()

        dim_index = content.index("Dimension")+1
        self.dimension = int(content[dim_index])

        #
        # We build the array of vertices
        #
        if "Vertices" in content :
            vert_index = content.index("Vertices")+1
            self.nb_vertices = int(content[vert_index])

            current_index = vert_index + 1

            self.vertices = np.zeros((self.dimension, self.nb_vertices))
            self.vertices_labels = np.zeros(self.nb_vertices, dtype=int)     # List of the region ID of the triangles


            for i in range(self.nb_vertices):
                str_list = content[current_index:current_index+self.dimension]
                self.vertices[:,i] = [float(elt) for elt in str_list]
                self.vertices_labels[i] = int(content[current_index+self.dimension])
                current_index += self.dimension + 1 # The +1 accounts for the reference

        #
        # We build the array of triangles
        #
        if "Triangles" in content :
            tri_index = content.index("Triangles")+1
            self.nb_triangles = int(content[tri_index])

            current_index = tri_index + 1

            self.triangles = np.zeros((3, self.nb_triangles),dtype=int) # List of triangles
            self.triangles_labels = np.zeros(self.nb_triangles, dtype=int)     # List of the region ID of the triangles

            for i in range(self.nb_triangles):
                str_list = content[current_index:current_index+3]
                self.triangles[:,i] = [int(elt) for elt in str_list]
                self.triangles_labels[i] = int(content[current_index+3])
                current_index += 3 + 1 # The +1 accounts for the reference

        #
        # We build the array of edges
        #
        if "Edges" in content :
            edges_index = content.index("Edges")+1
            self.nb_edges = int(content[edges_index])

            current_index = edges_index + 1

            self.edges = np.zeros((2, self.nb_edges),dtype=int) # List of triangles
            self.edges_labels = np.zeros(self.nb_edges, dtype=int)     # List of the region ID of the triangles

            for i in range(self.nb_edges):
                str_list = content[current_index:current_index+2]
                self.edges[:,i] = [int(elt) for elt in str_list]
                self.edges_labels[i] = int(content[current_index+2])
                current_index += 2 + 1 # The +1 accounts for the reference

        return self

    def save(self, filename):
        #
        # Save the medit mesh to a file
        #
        f = open(filename, "w")

        f.write("MeshVersionFormatted\n1\n\n")
        f.write("Dimension {}\n\n".format(self.dimension))

        #
        # Writes the vertices
        #
        if self.vertices is not None :
            f.write("\nVertices\n{}\n".format(self.nb_vertices))

            str_to_format = "{} "*self.dimension + " 0\n" # We add a zero for the ref of the point
            for i in range(self.nb_vertices):
                f.write(str_to_format.format(*(self.vertices[:,i])))

        #
        # Write the triangles
        #
        if self.triangles is not None :
            f.write("\nTriangles\n{}\n".format(self.nb_triangles))

            str_to_format = "{} {} {} {}\n"
            for i in range(self.nb_triangles):
                f.write(str_to_format.format(*(self.triangles[:,i]), self.triangles_labels[i]))

        #
        # Write the edges
        #
        if self.edges is not None :
            f.write("\nEdges\n{}\n".format(self.nb_edges))

            str_to_format = "{} {} {}\n"
            for i in range(self.nb_edges):
                f.write(str_to_format.format(*(self.edges[:,i]), self.edges_labels[i]))

        # End of file
        f.write("\nEnd")

        f.close()

        return self

    def from_getfem(self, gf_mesh):
        #
        # Fills our medit mesh from the getfem one gf_mesh
        #

        # Omptimize the structure : After optimisation, the points
        # (resp. convexes) will be consecutively numbered from 0 to
        # Mesh.max_pid()-1 (resp. Mesh.max_cvid()-1).
        #gf_mesh.optimize_structure() # Don't optimize if you want to keep a correspondance between a getfem mesh and this one

        self.dimension = gf_mesh.dim()

        #
        # Add the triangles
        #
        cvid = gf_mesh.cvid()
        self.nb_triangles = gf_mesh.max_cvid()+1
        self.triangles = np.zeros((3, self.nb_triangles), dtype=np.int64)
        self.triangles_labels = np.zeros(self.nb_triangles, dtype=np.int64)

        start = time.time()
        for i in range(self.nb_triangles):
            self.triangles[:,cvid[i]] = gf_mesh.pid_from_cvid(cvid[i])[0]+1 # +1 for the medit indexation

        start = time.time()
        pids, idx = gf_mesh.pid_from_cvid(cvid)
        idx = idx[1:-1] # On ne compte pas le premier ni le dernier convexe. cf pid_from_cvid
        pids = np.array(np.split(pids, idx)).T
        self.triangles = pids+1


        #
        # Add the vertices
        #
        self.vertices = gf_mesh.pts()
        self.nb_vertices = gf_mesh.max_pid()+1
        self.vertices_labels = np.zeros(self.nb_vertices, dtype=np.int64)

        #
        # Add the regions
        #
        # TODO: vectoriser pour que ça aille plus vite
        edges = []
        edges_labels = []
        for rid in gf_mesh.regions():
            cvfids = gf_mesh.region(rid)
            nb_convex_in_region = np.shape(cvfids)[1]

            for j in range(nb_convex_in_region):
                cvid, fid = cvfids[0,j], cvfids[1,j]

                # Add the whole triangle if fid = -1
                if fid == 65535 : # = -1 mod 2^16. Getfem must encode it on an unsigned int
                    self.triangles_labels[cvid] = rid

                # Else add only the edge
                else:
                    edges.append(gf_mesh.pid_in_faces(cvfids[:,j])+1)
                    edges_labels.append(rid)


        self.edges = np.array(edges).T
        self.edges_labels = np.array(edges_labels)
        self.nb_edges = self.edges_labels.size

        """
        for rid in gf_mesh.regions():
            cvfids = gf_mesh.region(rid)
            nb_convex_in_region = np.shape(cvfids)[1]
            convex_type = 'TRIANGLE' if cvfids[1,0] == 65535 else 'EDGE'

            if convex_type == 'TRIANGLE' :
                self.triangles_labels[cvfids[0,:]] = rid

            elif convex_type == 'EDGE' :
                continue
        """
        return self

    def to_getfem(self):
        #
        # Creates a getfem mesh out of this one
        #

        # Create the mesh (getfem indexation begins at 0)
        gf_mesh = gf.Mesh('ptND', self.vertices, self.triangles-1)

        # We set the different regions for the triangles
        cvid = gf_mesh.cvid()

        # List all the labels
        if self.triangles_labels is not None :
            unique_labels = np.unique(self.triangles_labels)
            if self.edges_labels is not None :
                unique_labels = np.concatenate((unique_labels, np.unique(self.edges_labels)))

        # WARNING : EN FAISANT COMME CA, L'INDICE 0, S'IL APPARAIT DANS LES
        # EDGES ET DANS LES TRIANGLES, SE VERRA DONNER DEUX INDICES DIFFERENTS
        if self.triangles_labels is not None :
            for label in np.unique(self.triangles_labels):
                labeled_cv = cvid[self.triangles_labels == label]
                labeled_cv = np.array([ cvid[self.triangles_labels == label], -1*np.ones(labeled_cv.size)])
                if label == 0:
                    gf_mesh.set_region(np.amax(unique_labels)+1, labeled_cv)
                    print(f"Warning : assigning label {np.amax(unique_labels)+1} to GetFEM region instead of 0")
                else:
                    gf_mesh.set_region(label, labeled_cv)


        # We set the region for the edges
        if self.edges_labels is not None :
            for label in np.unique(self.edges_labels):
                # WARNING : GetFEM doesn't accept region number equal to 0.
                # We replace it with the maximum of the labels + 1
                if label == 0:
                    #print(self.edges[:,self.edges_labels == label]-1)
                    labeled_faces = gf_mesh.faces_from_pid(self.edges[:,self.edges_labels == label]-1)
                    gf_mesh.set_region(np.amax(unique_labels)+1, labeled_faces)
                    print(f"Warning : assigning label {np.amax(unique_labels)+1} to GetFEM region instead of 0")
                else:
                    labeled_faces = gf_mesh.faces_from_pid(self.edges[:,self.edges_labels == label]-1)
                    gf_mesh.set_region(label, labeled_faces)

        gf_mesh.optimize_structure()

        return gf_mesh

    def load_solution(self, filename):
        #
        # Load a medit solution .sol file
        # WARNING : NE DOIT PAS MARCHER POUR LES CHAMPS TENSORIELS
        #
        # Open the medit file and convert to an array of strings
        content = open(filename, "r").read().split()

        # We build the array of vertices
        vert_index = content.index("SolAtVertices")+1
        nb_dof = int(content[vert_index])
        nb_fields = int(content[vert_index+1]) # ON CONSIDEREA ICI QUE C'EST TOUJOURS 1
        field_type = int(content[vert_index+2]) # 1 : scalaire, 2 : vecteur, 3 : tenseur

        current_index = vert_index + 3

        nb_values_in_line = 1 if field_type == 1 else self.dimension

        sol = np.zeros((nb_values_in_line, nb_dof))

        for i in range(nb_dof):
            """
                ALERT WARNING TODO WHATEVER : CEST PAS + dim_field !!!!!!
                PENSER AUX TENSEURS !!!!
            """
            str_list = content[current_index:current_index+nb_values_in_line]

            sol[:,i] = [float(elt) for elt in str_list]
            current_index += nb_values_in_line

        if field_type == 1 :
            sol = sol.flatten()

        return sol

    def save_solution(self, sol, filename, sol_type=None):
        #
        # Save the solution defined on the Medit mesh
        #

        # The type of the solution (0 : scalar field, 1:vector field, 2: tensor field)
        if sol_type is None:
            sol_type = sol.ndim

        f = open(filename, "w")

        f.write("MeshVersionFormatted 1\n\n")
        f.write("Dimension {}\n\n".format(self.dimension))
        f.write("SolAtVertices\n{}\n{} {}\n\n".format(self.nb_vertices, 1,sol_type)) # 1 pour le nombre de champ, sol_type pour le type de champ (1 pour scalaire, 3 pour tenseur)

        # Si la dimension est de 1 on l'étend pour pouvoir utiliser *(sol[...,i]) aussi
        if sol.ndim == 1:
            sol = np.array([sol])

        str_to_format = "{} "*sol[...,0].size + "\n"
        for i in range(self.nb_vertices):
            f.write(str_to_format.format(*(sol[...,i].flatten()))) # TODO : est-ce que c'est flatten ou flatten('C') ?


        f.write("\nEnd")

        return self

    def save_metric(self, met, filename):
        #
        # Saves a metric, which requires to extract the lower matrix
        #
        pass

    def extract_region(self, region):
        #
        # We create a medit mesh made of the triangles of
        # only one special region
        #
        submesh = MeditS()

        submesh.dimension = self.dimension

        # Trim the triangles
        submesh.triangles = self.triangles[:,self.triangles_labels == region]
        submesh.triangles_labels = self.triangles_labels[self.triangles_labels == region]
        submesh.nb_triangles = submesh.triangles_labels.size

        # Trim the vertices
        vertices_still_present = np.unique(submesh.triangles.flatten())-1 # -1 because of medit indexation
        submesh.vertices = self.vertices[:,vertices_still_present]
        submesh.vertices_labels = self.vertices_labels[vertices_still_present]
        submesh.nb_vertices = vertices_still_present.size

        # Renumber the vertices in the submesh.triangles array
        to_values = -np.ones(self.nb_vertices, dtype=np.int64) # Vertices indices can't take negative values so it's safe #LOL TURNED OUT THAT WAS FALSE YOU DICKHEAD
        to_values[vertices_still_present] = np.arange(1,submesh.nb_vertices+1)
        submesh.triangles = to_values[submesh.triangles-1]

        # Relabel the edges (we keep potentially labels of edges that doesn't exists anymore)
        if self.edges_labels is not None :
            edges = to_values[self.edges-1]
            edges_to_keep = np.logical_and(edges[0,:] > 0 , edges[1,:] > 0) # Some edges may have a vertex that have been deleted; we don't keep them
            submesh.edges = edges[:,edges_to_keep]
            submesh.edges_labels = np.copy(self.edges_labels[edges_to_keep]) if self.edges_labels is not None else None
            submesh.nb_edges = submesh.edges[0,:].size

        return submesh

    def adjacent_triangles(self, triangle_id, triangles_ids=None):
        #
        # Finds the triangles in the list triangles_ids sharing
        # an edge with triangle_id
        #
        if triangles_ids is None :
            triangles_ids = np.arange(self.nb_triangles)

        # The triangles vertices ids
        v1, v2, v3 = self.triangles[:,triangle_id]

        # True on the indices of triangles containing the given vertices, False otherwise
        contains_v1 = np.any(self.triangles[:,triangles_ids] == v1, axis=0)
        contains_v2 = np.any(self.triangles[:,triangles_ids] == v2, axis=0)
        contains_v3 = np.any(self.triangles[:,triangles_ids] == v3, axis=0)

        # Check if it contains the edges (the order of the vertices isn't taken into account)
        contains_v1_v2 = np.logical_and(contains_v1, contains_v2)
        contains_v2_v3 = np.logical_and(contains_v2, contains_v3)
        contains_v3_v1 = np.logical_and(contains_v3, contains_v1)

        # Build the array of neighbours
        neighbours = triangles_ids[np.where(np.logical_or(contains_v1_v2, np.logical_or(contains_v2_v3, contains_v3_v1)))]

        return neighbours

    def connected_component(self, triangle_id, triangles_ids=None):
        #
        # Finds the connected component (in term of triangles)
        # in which lies the triangle triangle_id
        # Also return the triangles that are not in the connected component
        #

        # The list of triangles to still visit
        to_visit = [triangle_id]

        # The list of all triangles
        if triangles_ids is None :
            triangles_ids = np.arange(self.nb_triangles)
        triangles_ids = np.delete(triangles_ids, np.where(triangles_ids ==  to_visit[0]))

        # Connected_component
        connected_component_triangles = []

        while to_visit :
            #print(to_visit)
            adjacent = self.adjacent_triangles(to_visit[0], triangles_ids)
            #print(len(to_visit))
            connected_component_triangles.append(to_visit[0])
            to_visit += adjacent.tolist()
            to_visit = to_visit[1:]

            for elt in adjacent:
                triangles_ids = np.delete(triangles_ids, np.searchsorted(triangles_ids,  elt))
            """
            triangles_ids = np.delete(triangles_ids, np.searchsorted(triangles_ids, adjacent))
            """

        return connected_component_triangles, triangles_ids

    def connected_components(self, region=None):
        #
        # Returns the list of all the connected components of the mesh
        #
        if region is None :
            triangles_ids = np.arange(self.nb_triangles)
        else :
            triangles_ids = np.where(self.triangles_labels == region)[0]
        #print(triangles_ids)

        components_list = []

        while triangles_ids.size > 0 :
            triangle_id = triangles_ids[0]
            cc, triangles_ids = self.connected_component(triangle_id, triangles_ids)
            components_list.append(cc)

        return components_list

    def get_area(self, triangles_ids):
        #
        # Compute the total area of the triangles listes in triangles_ids
        # Warning : master-level vectorization
        #
        tri = self.vertices[:,self.triangles[:,triangles_ids]-1]
        cross = np.cross(tri[:,1,:] - tri[:,0,:], tri[:,2,:] - tri[:,0,:], axis=0)
        area = np.sum(np.linalg.norm(cross, axis=0))/2

        return area

    def keep_n_components_in_region(self, n, region, new_region):
        #
        # Deletes the components that have the smallest area until
        # there is only n left. The deleted components will be relabeld
        # with the label new_region
        #
        n = int(n)

        # Compute the connected components
        cc = self.connected_components(region)

        if len(cc) <= n:
            return self

        areas = np.array([ self.get_area(component) for component in cc ])
        permutation = np.array(areas.argsort(), dtype=np.int8)

        # We only keep the n bigger components
        deleted_components = [ cc[permutation[i]] for i in range(len(cc)-n) ]

        # Identify the triangles that we keep
        deleted_triangles = np.concatenate(deleted_components)

        # Trim the triangles
        self.triangles_labels[deleted_triangles] = new_region

        # Delete the edges that no longer belong to triangles
        to_keep = []
        for i in range(self.nb_edges):
            v1, v2 = self.edges[:,i]

            # True on the indices of triangles containing the given vertices, False otherwise
            contains_v1 = np.any(self.triangles[:,self.triangles_labels == region] == v1, axis=0)
            contains_v2 = np.any(self.triangles[:,self.triangles_labels == region], axis=0)

            # Check if it contains the edges (the order of the vertices isn't taken into account)
            contains_v1_v2 = np.any(np.logical_and(contains_v1, contains_v2))

            if contains_v1_v2 :
                to_keep.append(i)

        to_keep = np.array(to_keep)

        self.edges = self.edges[:,to_keep]
        self.edges_labels = self.edges_labels[to_keep]
        self.nb_edges = self.edges_labels.size

        return self

    def delete_all_but_n_components(self, n):
        #
        # Deletes the components that have the smallest area until
        # there is only n left
        #
        n = int(n)

        # Compute the connected components
        cc = self.connected_components()

        if len(cc) <= n:
            return self

        areas = np.array([ self.get_area(component) for component in cc ])
        permutation = np.array(areas.argsort(), dtype=np.int8)

        # We only keep the n bigger components
        kept_components = [ cc[permutation[i]] for i in range(len(cc)-n,len(cc)) ]

        # Identify the triangles that we keep
        kept_triangles = np.concatenate(kept_components)

        # Create the submesh
        submesh = MeditS()

        submesh.dimension = self.dimension

        # Trim the triangles
        submesh.triangles = self.triangles[:,kept_triangles]
        submesh.triangles_labels = self.triangles_labels[kept_triangles]
        submesh.nb_triangles = submesh.triangles_labels.size

        """
        # Trim the vertices
        vertices_still_present = np.unique(sumbmesh.triangles.flatten())-1 # -1 because of medit indexation
        sumbmesh.vertices = self.vertices[vertices_still_present]
        sumbmesh.vertices_labels = self.vertices_labels[vertices_still_present]
        sumbmesh.nb_vertices = submesh.vertices_labels.size

        # Trim the edges information
        self.nb_edges = None
        self.edges = None
        self.edges_labels = None
        """
        # Let the vertices as they are
        submesh.vertices = np.copy(self.vertices)
        submesh.vertices_labels = np.copy(self.vertices_labels)
        submesh.nb_vertices = self.nb_vertices

        # Delete the edges that no longer belong to triangles
        to_keep = []
        for i in range(self.nb_edges):
            v1, v2 = self.edges[:,i]

            # True on the indices of triangles containing the given vertices, False otherwise
            contains_v1 = np.any(submesh.triangles == v1, axis=0)
            contains_v2 = np.any(submesh.triangles == v2, axis=0)

            # Check if it contains the edges (the order of the vertices isn't taken into account)
            contains_v1_v2 = np.any(np.logical_and(contains_v1, contains_v2))

            if contains_v1_v2 :
                to_keep.append(i)

        to_keep = np.array(to_keep)

        submesh.edges = self.edges[:,to_keep]
        submesh.edges_labels = self.edges_labels[to_keep]
        submesh.nb_edges = submesh.edges_labels.size

        return submesh

    def delete_edges(self):
        #
        #   Removes the edgeslabelled as "label"
        #

        self.edges = None
        self.edges_labels = None
        self.nb_edges = None

        return self

    def set_all_triangles_labels_to(self, label):
        #
        #   No comment
        #
        self.triangles_labels = [label]*self.nb_triangles

        return self
#
# Other functions
#
def getfem_to_scipy_sparse(Kin):

    JK, IK = gf.Spmat.get(Kin, "csc_ind")
    SK = gf.Spmat.get(Kin, "csc_val")
    Kpy = sparse.csc_matrix((SK, IK, JK), shape = Kin.size()) #TODO: VOIR SI CSR EST PAS PLUS RAPIDE POUR E CALCUL DES VP

    return Kpy

def heat_equation(u0, V, IM, T, step=100):
    #
    # Approximate resolution of the heat equation du/dt = Δu with u(0)=u0 between
    # 0 < t < T using upwind finite difference scheme with step timesteps
    #

    # Init the stuff
    dt = T/step
    u_old = u0

    # Define the model
    md = gf.Model("real")
    md.add_fem_variable("u", V)
    md.add_fem_data("u_old", V, 1)
    md.add_data("dt", 1)
    md.set_variable("dt", dt)
    md.add_linear_term(IM, "u*Test_u + dt*Grad_u.Grad_Test_u")
    md.add_source_term(IM, "u_old*Test_u")

    for i in range(step):
        md.set_variable("u_old", u_old)
        md.solve()
        u = md.variable("u")
        u_old = u

    return u

def extrapolate(val_from, points_from, tri_from, points_to, tri_to):
    #
    # Interpolates between two P1 functions defined on the points and triangles
    # Uses the getfem extrapolation function
    #
    mesh_from = gf.Mesh('ptND', points_from, tri_from-1) # Medit indexation begins at 1
    mesh_to = gf.Mesh('ptND', points_to, tri_to-1) # Medit indexation begins at 1

    # Defining the FE spaces
    V_from = gf.MeshFem(mesh_from, 1)
    V_from.set_fem(gf.Fem("FEM_PK(2, 1)"))
    V_to = gf.MeshFem(mesh_to, 1)
    V_to.set_fem(gf.Fem("FEM_PK(2, 1)"))


    gf_val_from = naive_interpolate(val_from, points_from, V_from.basic_dof_nodes())
    gf_val_to = gf.compute_extrapolate_on(V_from, gf_val_from, V_to).flatten()

    # We regularize a bit
    IM = gf.MeshIm(mesh_to, gf.Integ("IM_TRIANGLE(10)"))
    gf_val_to = heat_equation(gf_val_to, V_to, IM, 1e-4, 1)
    val_to = naive_interpolate(gf_val_to[np.newaxis,...], V_to.basic_dof_nodes(), points_to)

    return val_to

def flatten_lower_matrix(matrix_stack):
    #
    # Prend un empilement de matrices (array size_matrix*size_matrix*n_matrices)
    # et retourne un empilement de vecteur size_matrix*(size_matrix+1)/2 * n_matrices
    # dont chaque vecteur est "l'applati" de la matrice triangulaire inférieure
    # extraite (fonction pour créer le fichier de metrique anisotropique de MMG)
    #
    n_matrices = np.shape(matrix_stack)[-1]
    size_matrix = np.shape(matrix_stack)[0] # On fait l'hypothèse que la matrice est carrée

    vector_stack = np.zeros((int(size_matrix*(size_matrix+1)/2), n_matrices))

    lower_indices = np.tril_indices(size_matrix)

    # TODO : on peut surement vectoriser
    for i in range(n_matrices):
        vector_stack[:,i] = matrix_stack[..., i][lower_indices]

    return vector_stack

def naive_interpolate(val_from, points_from, points_to):
    #
    # Performs the interpolation of val_from (defined on the points points_from)
    # onto the points points_to. Here for each p in points_to we just assign
    # the value of the nearest points to p in points_from
    # TODO : REGARDER DU COTE DE CGAL POUR LA RECHERCHE RAPIDE DE PLUS PROCHES VOISINS
    #
    tree = KDTree(points_from.T)
    _, indices = tree.query(points_to.T)

    if val_from.ndim == 1:
        val_to = val_from[indices]

    else :
        val_to = val_from[...,indices]

    return val_to

def permutation(a,b,tol=1e-14):
    """ Determinates the permutation such that if a and b
    are two n*m arrays representing a set of points then
    a[:,perm] = b (i.e. the permutation of the columns of a that gives b)"""

    perm = np.empty(np.shape(a)[1],dtype=int)

    for i in range(perm.size):
        column = np.array([b[:,i]]).T
        almost_equal_columns = (np.abs(a-column)<tol).all(0)
        perm[i] = np.argwhere(almost_equal_columns)[0,0]

    return perm

def read_medit(medit_path):

    # Open the medit file and convert to an array of strings
    content = open(medit_path, "r").read().split()

    dim_index = content.index("Dimension")+1
    mesh_dim = int(content[dim_index])

    # We build the array of vertices
    vert_index = content.index("Vertices")+1
    nb_vertices = int(content[vert_index])

    current_index = vert_index + 1

    p = np.zeros((mesh_dim, nb_vertices))

    for i in range(nb_vertices):
        str_list = content[current_index:current_index+mesh_dim]
        p[:,i] = [float(elt) for elt in str_list]
        current_index += mesh_dim + 1 # The +1 accounts for the reference

    # We build the array of triangles
    tri_index = content.index("Triangles")+1
    nb_tri = int(content[tri_index])

    current_index = tri_index + 1

    t = np.zeros((3, nb_tri),dtype=int) # List of triangles
    r = np.zeros(nb_tri, dtype=int)     # List of the region ID of the triangles

    for i in range(nb_tri):
        str_list = content[current_index:current_index+3]
        t[:,i] = [int(elt) for elt in str_list]
        r[i] = int(content[current_index+3])
        current_index += 3 + 1 # The +1 accounts for the reference

    return (p,t,r)

def read_medit_sol(medit_path):

    print("ATTENTION : FONCTION FAUSSE DANS LA PREMIERE LIGNE DE LA BOUCLE, CF MeshS.load_solution()")
    # Open the medit file and convert to an array of strings
    content = open(medit_path, "r").read().split()
    dim_index = content.index("Dimension")+1
    mesh_dim = int(content[dim_index])

    # We build the array of vertices
    vert_index = content.index("SolAtVertices")+1
    nb_dof = int(content[vert_index])
    nb_fields = int(content[vert_index+1]) # ON CONSIDEREA ICI QUE C'EST TOUJOURS 1
    dim_field = int(content[vert_index+2]) # 1 : scalaire, 2 : vecteur, 3 : tenseur

    current_index = vert_index + 3

    sol = np.zeros((dim_field, nb_dof))

    for i in range(nb_dof):
        str_list = content[current_index:current_index+dim_field]
        #print(str_list)
        sol[:,i] = [float(elt) for elt in str_list]
        current_index += dim_field


    return sol

def getfem_mesh_to_p_t(mesh):
    """
    Returns the set of vertices p and set of triangles of a mesh
    in the getfem format
    """
    cvid = mesh.cvid()
    nbConvex = np.shape(cvid)[0]
    p = mesh.pts() #WARNING : the points index may not be the ones listed in getfem, see PID
    t = np.zeros((mesh.dim(), nbConvex), dtype=np.int64) # Il faut dim+1 points pour faire un simplexe

    for i in range(nbConvex):
        t[:,i] = mesh.pid_from_cvid(cvid[i])[0]

    return (p,t)

def medit_mesh_to_getfem(in_path, out_path):
    p,t,_ = read_medit(in_path)
    mesh = gf.Mesh("ptND", p,t-1) # Medit indeation starts at 1
    mesh.save(out_path)

def getlfem_mesh_to_medit(mesh_path_gf, mesh_path_medit):
    mesh = gf.Mesh('load', mesh_path_gf)
    p,t, _ = getfem_mesh_to_p_t(mesh)
    t = t+1 #Medit indexing
    save_medit(mesh_path_medit, p,t)

def save_medit_sol(filename_sol, sol, dimension, nPoints, sol_type=None):

    if sol_type is None :
        sol_type = sol.ndim

    f = open(filename_sol, "w")

    f.write("MeshVersionFormatted 1\n\n")
    f.write("Dimension {}\n\n".format(dimension))
    f.write("SolAtVertices\n{}\n{} {}\n\n".format(nPoints, 1,sol_type)) # 1 pour le nombre de champ, sol_type pour le type de champ (1 pour scalaire, 3 pour tenseur)

    # Si la dimension est de 1 on l'étend pour pouvoir utiliser *(sol[...,i]) aussi
    if sol.ndim == 1:
        sol = np.array([sol])

    for i in range(nPoints):
        str_to_format = "{} "*sol[...,0].size + "\n"
        f.write(str_to_format.format(*(sol[...,i].flatten()))) # TODO : est-ce que c'est flatten ou flatten('C') ?

    f.write("\nEnd")

def save_medit(filename_mesh,p,t,filename_sol=None,sol=None, sol_type=None):
    """
    Save the mesh represented by the points p and triangles/tetahedra t
    into the medit .mesh format in filename
    (t is already in the medit format, i.e. the indices of the points
    it refers to begins at 1)
    Here we suppose that the points in p and triangles doesn't have ref
    """
    dimp,nPoints = np.shape(p)
    dimt, nTriangles = np.shape(t)

    f = open(filename_mesh, "w")

    f.write("MeshVersionFormatted\n1\n\n")
    f.write("Dimension {}\n\n".format(dimp))

    # Writes the vertices
    f.write("Vertices\n{}\n\n".format(nPoints))

    for i in range(nPoints):
        # Will serve to format the line
        str_to_format = "{} "*dimp + " 0\n" # We add a zero for the ref of the point
        f.write(str_to_format.format(*(p[:,i])))

    f.write("\n")

    # Write the triangles
    f.write("Triangles\n{}\n".format(nTriangles))

    for i in range(nTriangles):
        str_to_format = "{} "*dimt + " 0\n" # We add a zero for the ref of the triangle/tetrahedra
        f.write(str_to_format.format(*(t[:,i])))


    # End of file
    f.write("\nEnd")

    f.close()

    # We save the solution if provided
    if filename_sol is not None and sol is not None :
        save_medit_sol(filename_sol, sol, dimp, nPoints, sol_type)

def tmp_convert(mesh_path_gf, mesh_path_medit, sol_path_gf, sol_path_medit):
    mesh = gf.Mesh('load', mesh_path_gf)
    p,t = getfem_mesh_to_p_t(mesh)
    t = t+1 #Medit indexing

    sol_gf = pickle.load(open(sol_path_gf,'rb'))
    V = gf.MeshFem(mesh,1)
    V.set_fem(gf.Fem("FEM_PK(2, 1)")) # 2 for the dimension of the mesh, 1 for the order of the element
    dof_nodes = V.basic_dof_nodes()
    perm = permutation(dof_nodes, p)
    #sol_medit = naive_interpolate(sol_gf[np.newaxis,:], dof_nodes, p)
    sol_medit = sol_gf[perm]

    save_medit(mesh_path_medit, p,t, sol_path_medit, sol_medit)


if __name__=="__main__":
    medit_mesh = MeditS().load("../Tore/meshes/medit/torus_fine.mesh")
    medit_mesh.to_getfem().save("../Tore/meshes/GetFEM/torus_fine.mesh")
