# ======================================================================================================================
# Assignment 3 Code
# Patrick Chabelski, 998242012
# November 2016
# - Run this main file (with LIBRARY.py in same folder) to run diffusion-advection solver
# - By default it is set to Assignment 3 Boundary conditions
# - To Change this (ie to Assignment 2), simply change the temp_west, temp_east, etc values below, and then uncomment/comment
# the relevant sections in the CD/UP solvers in the LIBRARY FILE.



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as ml
import timeit

from LIBRARY import uniform_grid
from LIBRARY import nonuniform_grid
from LIBRARY import TDMA_solver
from LIBRARY import CD_scheme
from LIBRARY import UP_scheme

start = timeit.default_timer()
np.set_printoptions(linewidth=300)

# user input and hardcoded variables, based on assignment requirements

nodes = int(input("Input number of nodes (x and y):"))
size = int(input("Input side length (square): "))
r_inf = float(input("Input inflation factor: "))
u_v = float(input("U (x) velocity (positive being west to east flow):"))
v_v = float(input("V (y) velocity (positive being south to north flow):"))
K = float(input("Input Diffusivity Constant K: "))
solver_type = input("Which solver should be used? (CD, UP, QK):")
temp_west = 0
temp_east = 100
temp_north = 100
temp_south = 0
T_o = 300
H = 10
zero = 10**-9
density = 100

#
# ===========================================================================
# GRID INFO GENERATION
if r_inf == 1:
    (coords,areas,deltas,grid) = uniform_grid(r_inf,nodes,size)
else:
    (coords,areas,deltas,grid) = nonuniform_grid(r_inf,nodes,size)

# ===========================================================================
# COEFFICIENT GENERATION
if solver_type == 'CD':
    print("Central Difference Scheme Selected")
    coeffs = CD_scheme(nodes,K,density,u_v,v_v,temp_west,temp_east,temp_north,temp_south,H,T_o,areas,deltas)
elif solver_type == 'UP':
    print("Upwind Scheme Selected")
    coeffs = UP_scheme(nodes,K,density,u_v,v_v,temp_west,temp_east,temp_north,temp_south,H,T_o,areas,deltas)
else:
    print("INVALID SCHEME SELECTED")
    exit()
print(coeffs)
#
# initial TDMA runthrough =========================================================================================

count = 1
a_diag = np.zeros((nodes))
b_diag = np.zeros((nodes))
c_diag = np.zeros((nodes))
d_vector = np.zeros((nodes))
t_vector = np.zeros((nodes))
temp_matrix = np.zeros((nodes*nodes))
prev_error = 100
delta_error = 100
while abs(delta_error) > 0.01:
     nextline = 0
     for i in range (0,nodes):   # fill the relevant data for FIRST LINE
         # [Point ID, an, as, aw, ae, ap, D]
         a_diag[i] = -coeffs[i, 2]  # south coefs
         c_diag[i] = -coeffs[i, 1]  # north coefs
         b_diag[i] = coeffs[i, 5]   # p coefs
         d_vector[i] = coeffs[i, 6] + temp_matrix[i+nodes]*coeffs[i,4]  # D value and eastern coef*temp

     temps = TDMA_solver(a_diag,b_diag,c_diag,d_vector)
     nextline = nodes
     for i in range (0,nodes):
         temp_matrix[i] = temps[i]
     for line in range (1,nodes-1):  # fill in relevant data for INTERIOR LINES
         for i in range(0, nodes):
             #print("ID", coeffs[i+nextline,0])
             a_diag[i] = -coeffs[i+nextline, 2]
             c_diag[i] = -coeffs[i+nextline, 1]
             b_diag[i] = coeffs[i+nextline, 5]
             d_vector[i] = coeffs[i+nextline, 6] + temp_matrix[i+nextline+nodes]*coeffs[i+nextline,4] + temp_matrix[i+nextline-nodes]*coeffs[i+nextline,3]
         temps = TDMA_solver(a_diag,b_diag,c_diag,d_vector)
         for i in range (0,nodes):
             temp_matrix[i+nextline] = temps[i]
         nextline = nextline + nodes
     for i in range (0,nodes):   # fill the relevant data for LAST LINE
         a_diag[i] = -coeffs[i+nextline, 2]
         c_diag[i] = -coeffs[i+nextline, 1]
         b_diag[i] = coeffs[i+nextline, 5]
         d_vector[i] = coeffs[i+nextline, 6] + temp_matrix[i+nextline-nodes]*coeffs[i+nextline,3]  # D value and western coef*temp
     temps = TDMA_solver(a_diag,b_diag,c_diag,d_vector)
     for i in range (0,nodes):
         temp_matrix[i+nextline] = temps[i]
     temp_matrix = np.transpose(temp_matrix)
     avg_error = (np.sum(temp_matrix))/(nodes*nodes)
     delta_error = prev_error-avg_error
     prev_error = avg_error
     count = count + 1

# ======================================================================================================
# The following code blocks will generate contour plots, as well as line plot for diagonal temperatures


stop = timeit.default_timer()
print("Time taken to reach convergence, in seconds: ", stop-start)
print("Number of iterations: ", count)
y_coords = grid[:,0]
x_coords = grid[:,1]
xi = np.linspace(min(x_coords),max(x_coords),(nodes*nodes))
yi = np.linspace(min(y_coords),max(y_coords),(nodes*nodes))
zi = ml.griddata(x_coords,y_coords,temp_matrix,xi,yi, interp='linear')
plt.contour(xi, yi, zi, colors='k')
plt.pcolormesh(xi, yi, zi, cmap = plt.get_cmap('rainbow'))
plt.colorbar()
plt.show()
print(temp_matrix)

xx_line_temp = np.zeros((nodes))
xx_line_coords = np.zeros((nodes))
xx_line_temp[0] = temp_matrix[nodes - 1]
#xx_line_coords[0] = float(math.sqrt((grid[nodes-1,0]**2)+(grid[nodes-1,1]**2)))
xx_line_coords[0] = grid[nodes-1,1]
increment = nodes-1
for i in range (1, nodes):
    diag_coord = (nodes-1)+i*increment
    xx_line_temp[i] = temp_matrix[diag_coord]
    #xx_line_coords[i] = float(math.sqrt((grid[diag_coord,0]**2)+(grid[diag_coord,1]**2)))
    xx_line_coords[i] = grid[diag_coord,1]

print(xx_line_temp)
print(xx_line_coords)
plt.plot(xx_line_coords,xx_line_temp)
plt.axis([0,np.max(xx_line_coords),0,np.max(xx_line_temp)])
plt.grid()
plt.xlabel("Distance along x-axis")
plt.ylabel("Phi (temperature) value")
plt.show()
