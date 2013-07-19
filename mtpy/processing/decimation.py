#!/usr/bin/env python

"""
mtpy/processing/decimation.py

Functions for the decimation of raw time series. 



For calling a batch decimation rather than just one file, use the appropriate scripts from the mtpy.utils subpackage. 


@UofA, 2013
(LK)

"""

#=================================================================


import numpy as np
import sys, os
import os.path as op
import time
import copy
import scipy.signal as sps


import  mtpy.utils.exceptions as MTex

#=================================================================

def decimate(ts_array, d, window_function='hanning'):
    """
	decimate the data by d using scipy.signal.resample 
	
	Arguments:
	-----------
		**ts_array** : np.ndarray 

		**d** : int
				decimation factor
				
		**window_function** : [ 'boxcar' | 'triang' | 'parzen' | 'bohman' | 
							    'blackman' | 'nuttall' | 'blackmanharris' |
								'flattop' | 'bartlett' | 'hanning' | 
								'barthann'| 'hamming' | 'kaiser' | 
								'gaussian' | 'general_gaussian' | 'chebwin'|
								'slepian' | 'hann' ]
								
								or input your own function, 
								
								see sps.resample for more details
				
	Outputs:
	---------
		**dec_array** : np.ndarray(n/d)
						decimated array
	"""
    
    
    n = len(ts_array)
    dec_array = sps.resample(farray, n/d, window=window_function)
	
	return dec_array