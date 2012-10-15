#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created Oct,2012
@author: Lars Krieger
Transform the .txt output from WingLink into a vtk file.
"""
import numpy as np
import os, sys, platform

import mtpy.utils.array2vtk 

class WingLinkException(Exception):
    pass





    
def main(arglist):
    """
    Wrapping script - converting WingLink ascii output into VTK file.

    Input:
    - WingLink model output ascii file, containing columns (x,y,z,resistivity)
    - Output file name 

    Output:
    - VTK file - to be imported by Mayavi or Paraview
    
    """



    

    return VTK_filename
    

if __name__ == "__main__":
    main(sys.argv)

