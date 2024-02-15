# Numerical shape optimization of Neumann eigenvalues on the sphere and the torus

![Image of optimal densities for the first eigenvalue](https://github.com/EloiMartinet/Neumann_Sphere/blob/main/banner.png)

Welcome to my GitHub repository! Here you can find the numerical optima that appears in the paper https://arxiv.org/abs/2303.12389.

In order to use the files of this repository, you first have to clone it :
```
https://github.com/EloiMartinet/Neumann_Sphere.git
```

### Results visualization

In both level-set and density cases, a FreeFem++ script is provided to compute the eigenvalues of the optimums, which are stored in MEDIT format.
To execute the scripts, first install FreeFem++ (https://freefem.org/). You can then go into the Density folder in your command line and execute

```
FreeFem++ Neumann_Density.edp Sphere/mu_1/11.6
```

Note that we don't precise any extension, since the script will automatically load the *.mesh and *.sol files.
Similarly, you can run the command
```
FreeFem++ Neumann_LS.edp Sphere/mu_2/7.121706429550119
```
from the Levelset folder to get the corresponding eigenvalues.

To visualize the results, you will need to install MEDIT (https://hal.inria.fr/inria-00069921/document) and from the Density folder, execute
```
medit Torus/mu_1/11.6
```
and then simply type the  "m" key to show the density. You can also uncomment the line
```
// medit("Optimal density.", Th, rho, wait=1);
```
in the file Neumann_Sphere_Density.edp but be aware that the display takes a lot of time in this case.

### Density optimization

The code to generate the optimal density for a target mass on a sphere can be found in `Density/Sphere/code`. The main file is `optimizer.py`. To be able to run it, you will need to install python and GetFEM (https://getfem.org/) for the finite elements computations, as well as cyipopt (https://cyipopt.readthedocs.io/en/stable/) for the interior point optimization procedure. You can then execute the file by executing 
```
python3 optimizer.py
```
This will output the result of each iteration in medit format in the folder `results`. You can then visualize an animation of the iterations by executing
```
medit results/rho -a 0 N
```
where N is the number of iterations. The parameters of the optimization can be changed directly inside the `main()` function of `optimizer.py` (maximum number of iterations, regularization parameters, meshes used, etc).

The `optimizer.py` file needs some pre-computed derivatives (in `derivatives`) which depend on the GetFEM meshes (in `meshes`). The derivatives are provided, but you can re-compute them by executing 
```
python3 martices_derivatives.py input_mesh output_derivative_file
``` 

Similar simulations on the plane can be found at https://github.com/EloiMartinet/Maximization_Of_Neumann_Eigenvalues.


### Level-set optimization

In addition to the previous tools, you will need to install the Advection tool (https://github.com/ISCDtoolbox/Advection) which allows you to advect the level-set function. For remeshing, please install mmgtools (https://www.mmgtools.org/). For distance function related operations, please install mshdist (https://github.com/ISCDtoolbox/Mshdist).

Please navigate to the folder `Levelset/Sphere/code`. There is two main files : `ersatz.py` and `remeshing.py`, each corresponding to a method implemented in the paper. Executing 
```
python3 ersatz.py
```
will output the meshes and level sets obtained by the ersatz procedure to the folder `results`. To visualize them, please run
```
medit results/ls -a 0 150
```
You can press `C`,`E` in the medit window to display the set (in yellow).

Executing 
```
python3 remeshing.py
```
will do the same but with the remeshing procedure. The initial mesh is given in `init.mesh` but you can specify another one obtained by the previous algorithm, for instance `results/ls.137.mesh`.

Enjoy!

You can find more wonderful simulations at https://eloimartinet.github.io/ !
