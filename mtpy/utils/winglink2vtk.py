#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created Oct,2012
@author: Lars Krieger
Transform the .txt output from WingLink into a vtk file.
"""
import numpy as np
import os, sys, platform

import mtpy.utils.array2vtk as array2vtk

class WingLinkException(Exception):
    pass



def cellwidths_to_coords(widths_list):
    """Convert list of cell widths into list of cell center positions (1D)"""

    origin = 0.
    coords = []
    n_cells = len(widths_list)

    first_cell = origin + widths_list[0]/2.
    coords.append(first_cell)

    origin = first_cell
    
    for idx in range(n_cells-1):
            current_cell = origin + widths_list[idx]/2.+widths_list[idx+1]/2.
            coords.append(current_cell)
            origin       = current_cell


    return coords



def convert_3D_to_table(raw_data):

    x_dim = int(raw_data[0])
    y_dim = int(raw_data[1])
    z_dim = int(raw_data[2])

    lo_x_values = [float(data) for data in raw_data[5:5+x_dim] ]
    lo_y_values = [float(data) for data in raw_data[5+x_dim:5+x_dim+y_dim]]
    #depth as negative values!!
    lo_z_values = [-float(data) for data in raw_data[5+x_dim+y_dim:5+x_dim+y_dim+z_dim]]

    x_coords = cellwidths_to_coords(lo_x_values)
    y_coords = cellwidths_to_coords(lo_y_values)
    z_coords = cellwidths_to_coords(lo_z_values)
    
    current_idx_tot = 5+x_dim+y_dim+z_dim

    lo_data = []
    
    for idx_depth in range(z_dim):
        current_idx_tot += 1
        z_cur = z_coords[idx_depth]
        
        for idx_y in range(y_dim):
            y_cur = y_coords[idx_y]

            for idx_x in range(x_dim):
                x_cur = x_coords[idx_x]

                data_cur = float(raw_data[current_idx_tot])

                datapoint = [x_cur, y_cur,z_cur,data_cur]
                lo_data.append(datapoint)

                current_idx_tot += 1
                

    data_as_table = np.array(lo_data)
    
    return data_as_table


    
def main(arglist):
    """
    Wrapping script - converting WingLink ascii output into VTK file.

    3D input expected! (Include dummy dimension, if neccessary)

    Input:
    - WingLink model output ascii file,
    - file type :
      - 2: 2D output - containing columns (x,y,z,resistivity)
      - 3: 3D output - containing 1 line of dimensions parameters/3 blocks of x,y,z values/N_z*N_y blocks of resistivities (with N_x values each) 

    - Output file name 

    Output:
    - unstructured grid VTK file - can be visualised using e.g. Mayavi or Paraview
    
    """
    
    usage_message = "Usage: \n\n winglink2vtk.py <ascii file with model data> <file type (2 or 3)> <depth scale (m or km)> <output file name> \n\n"

    if not len(arglist) == 5:
        sys.exit(usage_message)

    data_fn      = os.path.abspath(os.path.realpath(arglist[1]))
    datafiletype = int(float(arglist[2])) 
    outfilename  = arglist[4].strip().replace(' ','') 
    outfile      = os.path.abspath(os.path.realpath(outfilename))
    depth_scale  = arglist[4].strip().lower()
    
    if not datafiletype in [2,3]:
       sys.exit(usage_message)
 
    try:
        if datafiletype == 2 :
            data_in = np.loadtxt(data_fn)

            if not data_in.shape[1]==4:
                sys.exit('2D data file not understood - must have 4 columns x,y,z,res')

            if not depth_scale == 'km':
                data_in[:,2]/=1000.

            vtkgrid = array2vtk.VTKGrid(data_in)
            vtkgrid.set_variablename('Resistivity')
            vtkgrid.save(outfile)

        else:
            F        = open(data_fn)
            raw_data = F.read().strip().split()
            F.close()
    
            data_in = convert_3D_to_table(raw_data)

            vtkgrid = array2vtk.VTKGrid(data_in)
            vtkgrid.set_variablename('Resistivity')
            vtkgrid.save(outfile)

            
            

    except:
        raise WingLinkException

    
    

if __name__ == "__main__":
    main(sys.argv)

