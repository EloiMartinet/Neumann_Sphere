# Neumann_Sphere

![Image of optimal densities for the first eigenvalue](https://github.com/EloiMartinet/Neumann_Sphere/blob/master/banner.png)

Welcome to my GitHub repository ! Here you can find the numerical optima that appears in the paper (link).

In order to use the files of this repository, you first have to clone it :
```
https://github.com/EloiMartinet/Neumann_Sphere.git
```

In both levelset and density cases, a FreeFem++ script is provided in order to compute the eigenvalues of the optimums, which are stored in MEDIT format.
In order to execute the scripts, first install FreeFem++ (https://freefem.org/). You can then go into the density folder in your command-line and execute

```
FreeFem++ Neumann_Sphere.edp mu_1/11.6
```

Note that we don't precise any extension, since the script will automatically load the *.mesh and *.sol files.
You can run the same command from the levelset folder to get the corresponding eigenvalues.

In order to visualize the results, you will need to install MEDIT (https://hal.inria.fr/inria-00069921/document) and, still from the Density folder, execute
```
medit mu_1/11.6
```
and then simply type the  "m" key to show the density.

Similar simulations on the plane can be found at https://github.com/EloiMartinet/Maximization_Of_Neumann_Eigenvalues.

You can also find more wonderful simulations at https://eloi-martinet.netlify.app/ !
