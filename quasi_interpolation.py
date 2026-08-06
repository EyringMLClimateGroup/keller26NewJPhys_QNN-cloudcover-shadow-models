import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from itertools import  product
import jax
from jax import numpy as jnp
import scipy as sc
from  scipy.interpolate import RBFInterpolator
#from itertools import permutations


###--------------------------------------------------------sparse grid interpolation -----------------------------------------------------------------###


def locate_cell_sparse_grid_v2(x,L,d):
    # We assume we have sparse grid in d dimensions with refinement parameter L
    # We want to locate a cell in the sparse grid, in which x lies. (to reduce the size of the points for which we compute the Interpolant
    # The issue is, that this depends on the "lowest refinement" we find in the sparse grid (the convex hull of the points choosen should be a d-dimensional cube), 
    # s.t. in higher dimensions unfortunatly this leads only to a small reduction
    
    no_levels = 2# int(L-max(1,np.floor((L+d-1)/d)))
 
    
    if x.shape[0] < 2:
        x = x.reshape((1,d))
    #for x in [0,1], locate index in which grid cell it is
    y = x % 1 # (x/(2*np.pi)) % 1 # (#in case that we are on a [0,L]^d grid this needs to be rescaled
    y = np.squeeze(y) 
    cell_index = np.floor([y[i]*2**no_levels for i in range(d)])

    return cell_index
    
def fun_interpolant_sparse(X,L,d, sparse_tensor ,sparse_grid,sparse_grid_val):
    # Assuming we have a sparse grid, with values of function to interpolation in sparse_grid_val, compute the interpolant on X
    # L,d : refinement parameter and dimension of grid
    # hardcoded to 6 dimensions
    Y = np.empty((X.shape[0],))
    for k in range(X.shape[0]):
        if (k % 100) == 0:
            print(k)
        x = X[k,:].reshape((1,d)) % 1# assume periodicity
        cell_index = locate_cell_sparse_grid_v2(x,L,d)
        no_levels = 2 
        block_shape = (2**(L-no_levels)+1,)*d
        
        # Extract only relevant nonzeros
        indices = sparse_tensor.indices()
        vals = sparse_tensor.values()
        #fixed to 6 dimension
        mask = (
            (indices[0] >= cell_index[0]) & (indices[0] < (cell_index[0]+2**(L-no_levels)+1)) & #i0,i1 usw 
            (indices[1] >= cell_index[1]) & (indices[1] <  (cell_index[1]+2**(L-no_levels)+1)) &
            (indices[2] >= cell_index[2]) & (indices[2] <  (cell_index[2]+2**(L-no_levels)+1)) &   
            (indices[3] >= cell_index[3]) & (indices[3] <  (cell_index[3]+2**(L-no_levels)+1)) &
            (indices[4] >= cell_index[4]) & (indices[4] <  (cell_index[4]+2**(L-no_levels)+1)) &
            (indices[5] >= cell_index[5]) & (indices[5] <  (cell_index[5]+2**(L-no_levels)+1))
        )

        block_indices = indices[:, mask]
        block_values = vals[mask]
        
    
        # Shift coordinates so they start at 0 within the block

        block_indices = block_indices - torch.tensor([[cell_index[0]], [cell_index[1]], [cell_index[2]],[cell_index[3]], [cell_index[4]], [cell_index[5]]])
    
        # Create tahe sparse block tensor
        block_sparse = torch.sparse_coo_tensor(block_indices, block_values, size=block_shape)
        # make block dense
        block_sparse = block_sparse.coalesce()
        indices = block_sparse.values()
        
        indices=indices.numpy()-1

      
        x_val = sparse_grid[indices,:]

        y_val = sparse_grid_val[indices]
        
        Y[k] = RBFInterpolator(x_val,y_val)(x)[0] # this is slow and could be replaced by a more efficient interpolator, but enough to just test basic error rate
    return Y



### -------------------------------------------------------sparse grid generation  ----------------------------------------------- ####


def generate_sparse_grid(n,d):
    #generates sparse grid in [0,1]^d with refinement parameter n
    # Input: 
    # n (integer): refinement level of sparse grid
    # d (integer): dimension of domain
    # Output:
    # Array with unique points of sparse grids, shape: (number of gridpoint x dimension of input data)
    a = product(range(1,n+d),repeat=d) 
    cntr = 0
    for i in a:
        if np.sum(i) == (n+d-1):
            grid_axes = [ 2**(-i[k])* np.arange(0,2**i[k]+1,1) for k in range(d)]
            grids = np.meshgrid(*grid_axes,indexing = 'ij')
            grids = np.stack(grids,axis = -1)
            Wp = grids.reshape(-1,d)       
            if cntr == 0:
                W = Wp
                cntr = 1
            else: 
                W = np.concatenate([W,Wp])               
    sparse_grid= pd.DataFrame(W)
    sparse_grid= sparse_grid.drop_duplicates()
    sparse_grid= sparse_grid.to_numpy()
    return sparse_grid



### ------------------------------ Piecewise linear approximation--------------------------------------- ###


@jax.jit 
def phi(x): 
    # Basis function 1 - abs(x) on [-1,1], 0 otherwise
    y = jnp.maximum(jnp.zeros(x.shape),jnp.ones(x.shape)-jnp.abs(x))
    return y

def phi_half_start(x): 
    #half hat, for cells at boundary of cube
    #for local interpolant this function is not necessary
    y = np.maximum(np.zeros(x.shape),(np.ones(x.shape)-np.abs(x))*(x>0))
    return y

def phi_half_end(x): 
    #half hat at end of grid
    y = np.maximum(np.zeros(x.shape),(np.ones(x.shape)-np.abs(x))*(x<0))
    return y

#@jax.jit
def phi_scaled_nd(x,ell,position,h_inv): 
    #Input:
    #ell: grid shape, i.e. grid points in L*j2^(-ell) j in (0,2^ell), for cube [0,L]^d
    #h_inv. for interval_lentgh L . h^(-1) = 1/L*ell
    #position: in case not local interpolant is used, to determine if cell is at boundary of grid
    #Output:
    # tensor product of basis functions evaluated (scaled to grid, size h_inv)
    if len(x.shape) <2: 
        x = x.reshape((x.shape[0],1))
    val = np.ones((x.shape[0],))
    for i in range(x.shape[1]): # tensor product
        val *= phi(h_inv[i]*x[:,i])
        """
        # if interpolant is evaluated globally (i.e. without locating cell in which point lies in prior)
        if position is None: #
            val *= phi(h_inv[i]*x[:,i])
        else:
            if position[i] == 0: # at end of interval
                val *= phi_half_start(h_inv[i]*x[:,i])
            elif position[i] == ell[i]:
                val *= phi_half_end(h_inv[i]*x[:,i])
            else: 
                val *= phi(h_inv[i]*x[:,i])
        """
    
    return val # tensor product over dimension


def locate_cell(x,J,d):
    # locate the cell of directionally uniform grid (refinementparamter J), in which point x lies
    # Input:
    #x Point in grid [0,2*pi]^d
    #J specification of grid refinement (2*np.pi*2^(-J)*j for j in 0,...2^(J)
    # d dimension of grid
    # Output:
    # index of cell, in which point lies
    
 
    if x.shape[0] <2:
        x = x.reshape((1,d))
        

    y = (x /(2*np.pi)) % 1   
    cell_index = np.floor([y[:,i]*2**J[i] for i in range(d)]) 
    
    return cell_index



def local_interpolant(x,f,J,d):
    #piecewice affine linear inteprolant of f at point x, evalauted only locally, i.e. by locating cell in which point lies and only evaluating interpolant in cell
    #Input:
    # x: (number of points x input dimension) variabels at which to evaluate interpolant of f
    # f: function to interpolate
    # J: refinements of directionally uniform grid
    # d: dimension of input data
    #Output:
    #Y (number of points x output dimension): values of interpolant interpolated at x

    Y = np.zeros((x.shape[0],))
    cell_index = locate_cell(x,J,d)
    #generate the grid for the cell in which x lies
    grid_axes = [np.arange(0,2)]*d
    grids = jnp.meshgrid(*grid_axes,indexing = 'ij')# for k in range(x.shape[0])]
    grids = jnp.stack(grids,axis = -1)
    grids = grids.reshape(-1,d) 
    
    h_inv = jnp.asarray([1/(np.pi)*2**(j-1) for j in J])
    ell = jnp.asarray([2**j for j in J])
    for k in range(grids.shape[0]):
        Xk = np.asarray([np.pi*2**(-J[i]+1)*(cell_index[i,:] + grids[k,i]) for i in range(d)]).T
        Fxk = f(Xk)
        

        #evaluate basis function scaled and transfromed to cell + multiply by coefficeints
        Y += Fxk*phi_scaled_nd(x-Xk,ell = ell,position = None,h_inv = h_inv) 

    return Y
