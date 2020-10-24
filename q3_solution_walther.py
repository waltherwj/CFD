# -*- coding: utf-8 -*-
"""
This script can be run on any Python intepreter to generate the answers to Q3 in HW2 of ENME 572
Walther Wennholz Johnson - walther.johnson@ucalgary.ca
"""
## Imports
import numpy as np
import matplotlib.pyplot as plt

## Given Variables
a = [1,10] # convective velocities
k = 1 # thermal diffusivity
L = 1 # domain length
phi_0 = 0 # boundary value at 0
phi_1 = 1 # boundary value at 1
n_nodes_exact = 100 # number of points used to plot the exact solution
n_iter = 100 # maximum number of iterations on the entire mesh
number_of_nodes = [3,5] # number of nodes the space will be discretized to
error_threshold = 0.001 # error threshold to stop iterating

###################
#### FUNCTIONS ####
###################

## define exact solution equations
def phi_exact(x,  Pe, L):
    "calculates the exact value of the solution"
    phi = (np.exp(Pe*(x/L))-1)/(np.exp(Pe)-1)
    return phi
                    
def Peclet(a, L, k):
    "calculates peclet number"
    Pe = a*L/k
    return Pe    

## Function to discretize the space
def create_mesh(boundary_values, boundary_locations, n_nodes, dimensions):
    """
    takes 1d mesh conditions and returns a vector that only has values
    where boundary conditions were specified. Row 0 is the coordinates
    and Row 1 is the array values. Values that have not been initialized
    are initialized to NaN
    """
    ## create array of invalid values
    mesh_values = np.full((n_nodes), np.nan)
    
    ## create array with correct coordinates
    mesh_coordinates = np.linspace(*dimensions, n_nodes)
    
    ## stack the two arrays
    mesh = np.stack([mesh_coordinates, mesh_values])
    
    ## find the array index to apply the boundary condition to
    boundary_indices = np.in1d(mesh[0, :], boundary_locations)
    
    ## update the values array with the boundary conditions
    mesh[1, boundary_indices] = boundary_values
    
    return mesh, boundary_indices

## Setting up the basic finite difference functions
def phi_CD_only(mesh_values, i, dx, k=1, a=1):
    """
    takes a mesh array and a current index
    and applies the equation at the current location based on current values
    using the central difference method for both the convection and diffusion.
    Operates in-place
    """
    
    ## get the function values around the node of interest
    phi = mesh_values[i-1:i+2]
    
    ## if a node is NaN, replace it with 0. Better strategies could be devised
    ## for this step, but this naive approach is enough in this case
    phi[np.isnan(phi)]=0
    
    ## calculate the values of each of the terms that affect the current index,
    ## and then add them up
    term_1 = phi[0]*(a*dx + 2*k)
    term_2 = phi[2]*(2*k-a*dx)
    phi[1] = (term_1 + term_2)/(4*k)
    return None

def phi_CD_and_FD(mesh_values, i, dx, k=1, a=1):
    """
    takes a mesh array and a current index
    and applies the equation at the current location based on current values
    using the central difference method for diffusion, and forward difference 
    for convection.
    Operates in-place
    """
    
    ## get the function values around the node of interest
    phi = mesh_values[i-1:i+2]
    
    ## if a node is NaN, replace it with 0. Better strategies could be devised
    ## for this step, but this naive approach is enough in this case
    phi[np.isnan(phi)]=0
    
    ## calculate the values of each of the terms that affect the current index,
    ## and then add them up
    term_1 = phi[0]*(k)
    term_2 = phi[2]*(k-a*dx)
    phi[1] = (term_1 + term_2)/(2*k-a*dx)
    return None

def phi_CD_and_BD(mesh_values, i, dx, k=1, a=1):
    """
    takes a mesh array and a current index
    and applies the equation at the current location based on current values
    using the central difference method for diffusion, and forward difference 
    for convection.
    Operates in-place
    """
    
    ## get the function values around the node of interest
    phi = mesh_values[i-1:i+2]
    
    ## if a node is NaN, replace it with 0. Better strategies could be devised
    ## for this step, but this naive approach is enough in this case
    phi[np.isnan(phi)]=0
    
    ## calculate the values of each of the terms that affect the current index,
    ## and then add them up
    term_1 = phi[0]*(k+a*dx)
    term_2 = phi[2]*(k)
    phi[1] = (term_1 + term_2)/(2*k+a*dx)
    return None

def find_dx(mesh_coordinates, i):
    """
    takes a mesh array and a current index
    and finds the average delta x around that index. 
    It finds the exact dx for a uniform node distribution
    but also handles the case where there are small 
    variations around a node of a uniform mesh
    """
    ## get differential of coordinates around node
    dx_array = np.diff(mesh_coordinates[i-1:i+2])
    
    ## get average delta
    dx = dx_array.mean()
    
    return dx
    
def array_iteration(fun, mesh, fixed_index, kappa, a):
    """
    takes a function and a mesh, and iterates through the mesh
    values with the function
    """
    ## iterates through every node in the mesh coordinates
    ## updating the entire mesh
    for i, coordinate in enumerate(mesh[0]):
        ## check if the location isn't a boundary condition
        ## and if it is not find the delta x around that node, and then
        ## update that node based on the chosen finite difference function
        if not(fixed_index[i]):
            #print(i)
            dx = find_dx(mesh[0], i)
            fun(mesh[1], i, dx, kappa, a)
            
#############################          
#### CALCULATE SOLUTIONS #### 
#############################   

## Create a for loop to create all necessary graphs
## iterate through convective velocities
for velocity in a: #[1,10]

    ## iterate through all possible numbers of nodes finite difference
    for n_nodes in number_of_nodes: #[3,5]
        
        ##iterate each of the schemes
        for scheme in [phi_CD_only, phi_CD_and_FD]:
            
            ## create the mesh to iterate through and store the indices of the
            ## boundary conditions
            mesh, boundary_indices = create_mesh(boundary_values = [phi_0, phi_1],
                               boundary_locations = [0,L], 
                               n_nodes = n_nodes, 
                               dimensions = [0, L])
            
            ## start an array to store the delta between each solution
            error_array = [np.inf, np.inf]
            
            ## apply the scheme iteratively until the change between the
            ## previous and the next itireation is below the threshold, or 
            ## until we get to the maximum allowed number of iterations,
            ## when the program prints it failed to converge on a solution.
            while np.abs(error_array[-1]-error_array[-2])>=error_threshold or len(error_array)<5:
                
                ## store previous iteration of the mesh to calculate error
                prev_mesh_iter = mesh[1].copy()
                
                ## applies the scheme to the mesh once
                array_iteration(scheme, mesh, boundary_indices, kappa = k, a = velocity)
                
                ## appends the average numberical change between 
                ## the last two iterations
                
                error_array.append((mesh[1]-prev_mesh_iter).mean())
                
                ## if exceeding the max number of iterations, end iterations
                if len(error_array)>=n_iter:
                    print("failed to converge")
                    break
            
            ## create data for plotting exact solution
            x_exact_plot = np.linspace(0,1,n_nodes_exact)
            Pe_plot = Peclet(velocity, L, k)
            phi_ex_plot = phi_exact(x_exact_plot, Pe_plot, L)
            
            ## plot the finite difference solution
            plt.plot(mesh[0], mesh[1])
            ## plot the exact solution on the same graph
            plt.plot(x_exact_plot, phi_ex_plot)
            ## add a legend
            plt.legend(["finite difference", "exact"])
            ## add a title
            plt.title(f"velocity = {velocity} ,  number of nodes = {n_nodes}"
                      f"\n with function {scheme.__name__}"
                      f"\n n_iter = {len(error_array)-2} for delta in solution = {error_array[-1]}")
            plt.ylabel('phi')
            plt.xlabel('x coordinate')
            
            ## force multiple graphs to be created
            plt.pause(0.001)
            























