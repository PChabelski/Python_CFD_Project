# Python_CFD_Project
Sample Code: Implementation of Upwind and Quick Schemes for 2D Diffusion/Advection CFD solvers

The purpose of this code was to model 2D Diffusion and Advection using Upwind and Central Differencing schemes.
The user inputs data for:
- grid generation (number of nodes in each direction, size of domain, inflation factor), 
- CFD solver type (Upwind or Entral Differencing)
- fluid properties (x and y velocities, diffusivity factor).

As a default case, can select velocities = 0, Diffusivity constant K = 5, Number of nodes = 50, inflation factor = 1, and side length = 1.
This will result in a smooth temperature gradient contour plot output. The user can then play around with velocities (1m/s) or other parameters
to see how each scheme will model the flow flow and temperature gradients. 
More details about the physics and code background can be found in the report pdf file.



The code consists of two parts: 

1: MAIN_FILE.py, which is where the initialization, user input, and function calling is performed. 
Once node coefficients are derived, the resulting 2D grid will be plot to show temperature contours. 

2: LIBRARY.py, which contains the following modules for the analysis:
--> Grid Generation (uniform and non-uniform): based on spacing data (number of nodes, inflation factor, etc) provided by the user, 
    this will generate the 2D mesh for the analysis. This will also plot the resulting grid for visualization purposes, with the points
    representing the nodes.
--> TDMA Solver: Custom built to efficiently iterate and solve tri-diagonal matrices
--> Central Differencing Scheme: Will solve for coefficients at each node based on the CD scheme. 
--> Upwind Scheme: Will solve for coefficients at each node based on Upwind scheme. 
