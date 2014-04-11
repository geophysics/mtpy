#!/usr/bin/env python

"""
mtpy/utils/spectrogram.py


Class and functions for the imaging of a (set of) spectrogram(s). 


Class:
    Spectrogram() -- generated from a output of the time frequency analysis in mtpy.processing.tf 


Functions:
    Batch processing 
    plot setup
    

Output to be sent to mayavi or paraview



@UofA, 2013
(LK)

"""

#=================================================================


import numpy as np

import mtpy.processing.tf as MTtf
import mtpy.core.z as MTz
import mtpy.core.edi as MTedi
import mtpy.analysis.pt as MTpt


from mtpy.utils.exceptions import *


#=================================================================