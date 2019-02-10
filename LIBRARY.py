# =======================================================================================================================
# LIBRARY FILE
# Patrick Chabelski, 998242012, November 2016
# Main code will call out modules that are defined here
# Modules include: grid generators (uniform and non-uniform), TDMA solver, Central Differencing Scheme, Upwind Scheme

import numpy as np
import matplotlib.pyplot as plt

# =======================================================================================================================
# This module will take the inflation number, # of nodes, and size of domain and will generate node coordinates,
# associated areas and deltas (distances between nodes)
def nonuniform_grid (r_inf, nodes, size):
    coords = np.zeros(nodes)
    areas = np.zeros(nodes)
    deltas = np.zeros(nodes + 1)
    d_init = ((1.0 - r_inf) / (1.0 - r_inf ** ((nodes + 1) / 2))) * (size / 2) # define initial delta; space between boundary and node 1
    deltas[0] = d_init
    deltas[nodes] = d_init
    for r in range(1, int((nodes + 1) / 2)): # fill in remaining deltas based on inflation factor
        deltas[r] = deltas[r - 1] * r_inf
        deltas[nodes - r] = deltas[r]
    if (nodes % 2 == 0):  # for an even number of points
        deltas[int((nodes + 1) / 2)] = 1 - np.sum(deltas)

    print("deltas", deltas)
    coords[0] = deltas[0]
    for i in range(1, nodes):
        coords[i] = coords[i - 1] + deltas[i]
    print("Coords", coords)

    for a in range(0, nodes):  # generate areas for each node; nodes near BCs will have larger areas
        areas[a] = (deltas[a] / 2) + (deltas[a + 1] / 2)
        if a == 0:
            areas[a] = deltas[a] + (deltas[a + 1] / 2)
        if a == nodes - 1:
            areas[a] = (deltas[a] / 2) + deltas[a + 1]
    print("Areas", areas)

    numpoints = nodes * nodes
    grid = np.zeros((numpoints, 2))
    k = 0
    for i in coords:
        for j in coords:
            grid[k, 0] = j
            grid[k, 1] = i
            k = k + 1
    # Can uncomment the following block to create a plot of where the nodes will be
    print(grid)
    plt.plot(grid[:,0],grid[:,1],'ro')
    plt.axis([0,size,0,size])
    plt.grid()
    plt.show()
    return (coords,areas,deltas,grid)

# =======================================================================================================================
# This module will generate a uniform grid, in a similar fashion to the non-uniform grid generator
def uniform_grid (r_inf, nodes, size):
    coords = np.zeros(nodes)
    areas = np.zeros(nodes)
    deltas = np.zeros(nodes + 1)
    d_init = size/(nodes+1)
    deltas[0] = d_init
    deltas[nodes] = d_init
    for r in range(1, int((nodes + 1) / 2)):
        deltas[r] = deltas[r - 1] * r_inf
        deltas[nodes - r] = deltas[r]
    if (nodes % 2 == 0):  # its even
        deltas[int((nodes + 1) / 2)] = 1 - np.sum(deltas)

    print("deltas", deltas)
    coords[0] = deltas[0]
    for i in range(1, nodes):
        coords[i] = coords[i - 1] + deltas[i]
    print("Coords", coords)

    for a in range(0, nodes):
        areas[a] = (deltas[a] / 2) + (deltas[a + 1] / 2)
        if a == 0:
            areas[a] = deltas[a] + (deltas[a + 1] / 2)
        if a == nodes - 1:
            areas[a] = (deltas[a] / 2) + deltas[a + 1]
    print("Areas", areas)

    numpoints = nodes * nodes
    grid = np.zeros((numpoints, 2))
    k = 0
    for i in coords:
        for j in coords:
            grid[k, 0] = j
            grid[k, 1] = i
            k = k + 1

    # Can uncomment the following block to create a plot of where the nodes will be
    print(grid)
    plt.plot(grid[:,0],grid[:,1],'ro')
    plt.axis([0,size,0,size])
    plt.grid()
    plt.show()
    return (coords, areas, deltas,grid)

# =======================================================================================================================
# The TDMA module is the same module that has been in use since Assignment 1
def TDMA_solver (a_diag, b_diag, c_diag, d_vector):

    mat_size = b_diag.size              # get size of entire A matrix; this will be the number of entries in x matrix
    x_vector = np.zeros(mat_size)       # define vector of results for which we are solving for (Ax = D)

    # define initial new parameters for TDMA
    # Note: to save space, the old values of c and d are replaced by corresponding C' and D' values; these old values are
    # not needed after C' and D' are calculated, therefore this will reduce space and computational time

    c_diag[0] = c_diag[0] / b_diag[0]       # as per TDMA method, first C' value follows a different formula
    d_vector[0] = d_vector[0] / b_diag[0]   # as per TDMA method, first D' value follows a different formula

    # forward sweep loop
    for i in range(1, mat_size):  # loop to generate C' and D' values from 2nd element to final (nth) element
        c_diag[i] = c_diag[i] / (b_diag[i] - (a_diag[i] * c_diag[i - 1]))
        d_vector[i] = (d_vector[i] - (a_diag[i] * d_vector[i - 1])) / (b_diag[i] - (a_diag[i] * c_diag[i - 1]))

    x_vector[mat_size - 1] = d_vector[mat_size - 1]     # according to TDMA final x value (nth x value) is equal to the nth d_vector value

    # backward sweep loop
    for j in range(mat_size - 2, -1, -1):   # loop to back-calculate x values using known C' and D' values
        x_vector[j] = d_vector[j] - (c_diag[j] * x_vector[j + 1])

    return(x_vector)

# =======================================================================================================================
# Central Differencing Scheme Solver: will solve for coefficients of each node based on CD scheme. It is essentially a nested
# loop that calculates the values for each point, then stores them in a coefficient matrix. Boundary conditions are implemented within the
# nested loop, if the counter detects that it is near a "wall" (ie i or j = 0 or max number of nodes)
def CD_scheme (nodes, K, density, u_v, v_v, temp_west, temp_east, temp_north, temp_south, H, T_o, areas, deltas):
    numpoints = nodes*nodes
    coeffs = np.zeros((numpoints,7))  # [Point ID, an, as, aw, ae, ap, D]
    ID = 0
    for i in range (0,nodes): # outer loop is lines, going west to east (ie, i = 0 is west, i = max is east)
        for j in range (0,nodes): # inner loop is individual line points, going south to north (ie, j = 0 is south, j = max is north)
            api_term = 0
            apj_term = 0
            Di_term = 0
            Dj_term = 0
            Diff_term_w = ((K * areas[j]) / deltas[i])      # Diff terms are the diffusion component of the solver
            Diff_term_e = ((K * areas[j]) / deltas[i+1])
            Diff_term_n = ((K * areas[i]) / deltas[j+1])
            Diff_term_s = ((K * areas[i]) / deltas[j])
            F_ew = float(density*u_v*areas[j])              # F terms are the velocity component of the solver
            F_ns = float(density*v_v*areas[i])

            a_w = Diff_term_w + (F_ew/2)  # area dependent on j increment
            a_e = Diff_term_e - (F_ew/2)  # area dependent on j increment
            a_n = Diff_term_n - (F_ns/2)  # area dependent on i increment
            a_s = Diff_term_s + (F_ns/2)  # area dependent on i increment
            
            # Boundary handling:
            if i == 0:  # WEST WALL
                api_term = a_w
                Di_term = api_term * temp_west
                a_w = 0
            if i == (nodes - 1):  # EAST WALL
                api_term = a_e
                Di_term = api_term * temp_east
                a_e = 0
            if j == 0:  # SOUTH WALL
                apj_term = a_s
                Dj_term = apj_term*temp_south
                a_s = 0
                # Can uncomment the block below, comment the above to set up the BCs for assignment 2
                #Dj_term = 0
                #apj_term = 0
                #a_s = 0
            if j == (nodes-1):  # NORTH WALL
                apj_term = a_n
                Dj_term = apj_term*temp_north
                a_n = 0
                # Can uncomment the block below, comment the above to set up the BCs for assignment 2
                #Dj_term = H*deltas[j+1]*T_o
                #apj_term = H*deltas[j+1]
                #a_n = 0
            a_p = a_n + a_s + a_w + a_e + api_term + apj_term
            D = 0 + Di_term + Dj_term
            values = [ID+1,a_n, a_s, a_w, a_e, a_p, D]
            coeffs[ID,:] = values
            ID = ID + 1

    return (coeffs)


# =======================================================================================================================
# Upwind Scheme Solver: will solve for coefficients of each node based on UP scheme.
def UP_scheme (nodes, K, density, u_v, v_v, temp_west, temp_east, temp_north, temp_south, H, T_o, areas, deltas):
    numpoints = nodes*nodes
    coeffs = np.zeros((numpoints,7))  # [Point ID, an, as, aw, ae, ap, D]
    ID = 0
    for i in range (0,nodes): # outer loop is lines, going west to east (ie, i = 0 is west, i = max is east)
        for j in range (0,nodes): # inner loop is individual line points, going south to north (ie, j = 0 is south, j = max is north)
            api_term = 0
            apj_term = 0
            Di_term = 0
            Dj_term = 0
            Diff_term_w = ((K * areas[j]) / deltas[i])
            Diff_term_e = ((K * areas[j]) / deltas[i+1])
            Diff_term_n = ((K * areas[i]) / deltas[j+1])
            Diff_term_s = ((K * areas[i]) / deltas[j])
            F_ew = float(density*u_v*areas[j])
            F_ns = float(density*v_v*areas[i])

            a_w = Diff_term_w + max((F_ew/2),0)  # area dependent on j increment
            a_e = Diff_term_e + max((-F_ew/2),0)  # area dependent on j increment
            a_n = Diff_term_n + max((-F_ns/2),0)  # area dependent on i increment
            a_s = Diff_term_s + max((F_ns/2),0)  # area dependent on i increment
            
            # Boundary handling:
            if i == 0:  # WEST WALL
                api_term = a_w
                Di_term = api_term * temp_west
                a_w = 0
            if i == (nodes - 1):  # EAST WALL
                api_term = a_e
                Di_term = api_term * temp_east
                a_e = 0
            if j == 0:  # SOUTH WALL
                apj_term = a_s
                Dj_term = apj_term*temp_south
                a_s = 0
                # Can uncomment the block below, comment the above to set up the BCs for assignment 2
                #Dj_term 0= 0
                #apj_term = 0
                #a_s = 0
            if j == (nodes-1):  # NORTH WALL
                apj_term = a_n
                Dj_term = apj_term*temp_north
                a_n = 0
                # Can uncomment the block below, comment the above to set up the BCs for assignment 2
                #Dj_term = H*deltas[j+1]*T_o
                #apj_term = H*deltas[j+1]
                #a_n = 0
            a_p = a_n + a_s + a_w + a_e + api_term + apj_term
            D = 0 + Di_term + Dj_term
            values = [ID+1,a_n, a_s, a_w, a_e, a_p, D]
            coeffs[ID,:] = values
            ID = ID + 1

    return (coeffs)
