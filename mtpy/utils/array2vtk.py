#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created Oct,2012

@author: Aixa Rivera-Rios, Lars Krieger

Creating and saving (unstructured) grid from a cloud of points.

"""
import numpy as np
from scipy.spatial import Delaunay 
import os, sys, platform

class VTKException(Exception):
    pass


if platform.platform().lower().startswith('linux'):
    try:
        from  tvtk.api import tvtk
    except:
        import vtk as tvtk

elif platform.platform().lower().startswith('win'):
    try:
        from enthought.tvtk.api import tvtk
    except:
        raise VTKException

else:
    sys.exit("system could not be determined")        




class Array2VTK():
    """
    Class for defining and handling a VTK grid.

    Input:
    - array of grid points, shape (n_points, n_dimensions)
    - array of data points, shape (n_points, [1/3])
    - grid type ['structured'/'unstructured']
    - data type ['scalar'/'vector']
    
    """

    def __init__(self, points, data, grid_type='unstructured', data_type='scalar'):

        self.grid_type = grid_type
        self.points    = points
        self.data      = data
        self.dimension = self.points.shape[1]


        if self.grid_type=='unstructured':

            interp_grid = Delaunay(self.points)
            new_points  = interp_grid.points

            #include dummy dimension for 2D data set
            if self.points.shape[1]==2:
                tmp_points = np.zeros((new_points.shape[0],3))
                tmp_points[:,0] = new_points[:,0]
                tmp_points[:,2] = new_points[:,1]
                new_points = tmp_points
        
            
            new_cells   = interp_grid.vertices
            
            self.grid = tvtk.UnstructuredGrid(points=new_points)
            self.grid.point_data.scalars = self.data
            self.grid.set_cells(tvtk.Tetra().cell_type,new_cells)
            

    def set_variablename(self, name):
        
        self.grid.point_data.scalars.name = name

    
    def save(self, outfilename):

        if self.grid_type=='unstructured':
            out_fn = os.path.abspath(os.path.realpath(outfilename+'.vtu'))
            writer = tvtk.XMLUnstructuredGridWriter(input=self.grid, file_name=out_fn)
            writer.write()   

        elif self.grid_type=='structured':
            out_fn = os.path.abspath(os.path.realpath(outfilename+'.vts'))
            writer = tvtk.XMLStructuredGridWriter(input=self.grid, file_name=out_fn)
            writer.write()

        else:
            sys.exit('cannot save grid -- grid type is undefined')

    
    

if __name__ == "__main__":

    """Usage: array2vtk.py <ascii file with grid points> <ascii file with data> <Data variable name> <output file name>"""

    if len(sys.argv) != 5:
        print "Usage:\n \n   array2vtk.py <ascii file with grid points> <ascii file with data> <data variable name> <output file name>\n\n"
        sys.exit()

    try:
    
        local_args = sys.argv[1:]
        
        gridfile = os.path.abspath(local_args[0])
        datafile = os.path.abspath(local_args[1])

        grid_in = np.loadtxt(gridfile)
        data_in = np.loadtxt(datafile)

        field_name  = local_args[2]

        outfilename = local_args[3]

    except:
        print 'ERROR: wrong input parameters'
        raise VTKException('Could not digest input parameters!')
    

    vtkgrid = Array2VTK(grid_in,data_in)
    vtkgrid.set_variablename(field_name)
    vtkgrid.save(outfilename)

