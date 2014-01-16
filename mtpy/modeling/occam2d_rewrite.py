# -*- coding: utf-8 -*-
"""
Spin-off from 'occamtools'
(Created August 2011, re-written August 2013)

Tools for Occam2D

authors: JP/LK


Classes:
    - Data
    - Model
    - Setup
    - Run
    - Plot
    - Mask


Functions:
    - getdatetime
    - makestartfiles
    - writemeshfile
    - writemodelfile
    - writestartupfile
    - read_datafile
    - get_model_setup
    - blocks_elements_setup


"""
#==============================================================================
import numpy as np
import scipy as sp
from scipy.stats import mode
import sys
import os
import os.path as op
import subprocess
import shutil
import fnmatch
import datetime
from operator import itemgetter
import time
import matplotlib.colorbar as mcb
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import scipy.interpolate as spi

import mtpy.core.edi as MTedi
import mtpy.core.mt as mt
import mtpy.modeling.winglinktools as MTwl
import mtpy.utils.conversions as MTcv
import mtpy.utils.filehandling as MTfh
import mtpy.utils.configfile as MTcf
import mtpy.analysis.geometry as MTgy
import mtpy.utils.exceptions as MTex
import scipy.interpolate as si


reload(MTcv)
reload(MTcf)
reload(MTedi)
reload(MTex)

#==============================================================================

occamdict = {'1':'resxy','2':'phasexy','3':'realtip','4':'imagtip','5':'resyx',
             '6':'phaseyx'}

#------------------------------------------------------------------------------

class Mesh():
    """
    deals only with the finite element mesh.  Builds a finite element mesh 
    based on given parameters
    
    
    """

    def __init__(self, station_locations=None, **kwargs):
        
        self.station_locations = station_locations
        self.n_layers = kwargs.pop('n_layers', 90)
        self.cell_width = kwargs.pop('cell_width', 100)
        self.num_x_pad_cells = kwargs.pop('num_x_pad_cells', 7)
        self.num_z_pad_cells = kwargs.pop('num_z_pad_cells', 5)
        self.x_pad_multiplier = kwargs.pop('x_pad_multiplier', 1.5)
        self.z1_layer = kwargs.pop('z1_layer', 10.0)
        self.z_bottom = kwargs.pop('z_bottom', 200000.0)
        self.z_target_depth = kwargs.pop('z_target_depth', 50000.0)
        self.num_x_pad_small_cells = kwargs.pop('num_x_pad_small_cells', 2)
        self.save_path = kwargs.pop('save_path', None)
        self.mesh_fn = kwargs.pop('mesh_fn', None)
        self.elevation_profile = kwargs.pop('elevation_profile', None)
            
        self.x_nodes = None
        self.z_nodes = None
        self.x_grid = None
        self.z_grid = None
        self.mesh_values = None
        self.air_value = 1e13
        self.air_key = '0'
        
    def build_mesh(self):
        """
        build the finite element mesh given the parameters
        
        """
        
        if self.station_locations is None:
            raise OccamInputError('Need to input station locations to define '
                                  'a finite element mesh')
                                  
        #be sure the station locations are sorted from left to right
        self.station_locations.sort()
        
        #center the stations around 0 like the mesh will be
        self.station_locations -= self.station_locations.mean()
        
        #1) make horizontal nodes at station locations and fill in the cells 
        #   around that area with cell width. This will put the station 
        #   in the center of the regularization block as prescribed for occam
        # the first cell of the station area will be outside of the furthest
        # right hand station to reduce the effect of a large neighboring cell.
        self.x_grid = np.array([self.station_locations[0]-self.cell_width])
                
        for ii, offset in enumerate(self.station_locations[:-1]):
            dx = self.station_locations[ii+1]-offset
            num_cells = int(np.floor(dx/self.cell_width))
            #if the spacing between stations is smaller than mesh set cell
            #size to mid point between stations
            if num_cells == 0:
                cell_width = dx/2.
                num_cells = 1
            #calculate cell spacing so that they are equal between neighboring
            #stations
            else:
                cell_width = dx/num_cells
            if self.x_grid[-1] != offset:
                self.x_grid = np.append(self.x_grid, offset)
            for dd in range(num_cells):
                new_cell = offset+(dd+1)*cell_width
                #make sure cells aren't too close together
                try:
                    if abs(self.station_locations[ii+1]-new_cell) >= cell_width*.9:
                        self.x_grid = np.append(self.x_grid, new_cell)
                    else:
                        pass
                except IndexError:
                    pass
        
        self.x_grid = np.append(self.x_grid, self.station_locations[-1])        
        # add a cell on the right hand side of the station area to reduce 
        # effect of a large cell next to it       
        self.x_grid = np.append(self.x_grid, 
                                self.station_locations[-1]+self.cell_width)
                            
        #--> pad the mesh with exponentially increasing horizontal cells
        #    such that the edge of the mesh can be estimated with a 1D model
        nxpad = self.num_x_pad_cells
        padding_left = np.zeros(nxpad)
        padding_left[0] = self.x_grid[0]*self.x_pad_multiplier
        
        padding_right = np.zeros(nxpad)
        padding_right[0] = self.x_grid[-1]*self.x_pad_multiplier
        for ii in range(1,nxpad):
            padding_left[ii] = padding_left[ii-1]*self.x_pad_multiplier
            padding_right[ii] = padding_right[ii-1]*self.x_pad_multiplier
            
        #pad cells on right
        self.x_grid = np.append(self.x_grid, padding_right)
        
        #pad cells on left
        self.x_grid = np.append(padding_left[::-1], self.x_grid)
        
        #--> compute relative positions for the grid
        self.x_nodes = self.x_grid.copy()
        for ii, xx in enumerate(self.x_grid[:-1]):
            self.x_nodes[ii] = abs(abs(self.x_grid[ii+1])-abs(xx))
        self.x_nodes = self.x_nodes[:-1] 

        #2) make vertical nodes so that they increase with depth
        #--> make depth grid
        log_z = np.logspace(np.log10(self.z1_layer), 
                            np.log10(self.z_target_depth-\
                                     np.logspace(np.log10(self.z1_layer), 
                            np.log10(self.z_target_depth), 
                            num=self.n_layers)[-2]), 
                            num=self.n_layers-self.num_z_pad_cells)
        
        #round the layers to be whole numbers
        ztarget = np.array([zz-zz%10**np.floor(np.log10(zz)) for zz in 
                           log_z])
        
        #--> create padding cells past target depth
        log_zpad = np.logspace(np.log10(self.z_target_depth), 
                            np.log10(self.z_bottom-\
                                    np.logspace(np.log10(self.z_target_depth), 
                            np.log10(self.z_bottom), 
                            num=self.num_z_pad_cells)[-2]), 
                            num=self.num_z_pad_cells)
        #round the layers to be whole numbers
        zpadding = np.array([zz-zz%10**np.floor(np.log10(zz)) for zz in 
                               log_zpad])
                               
        #create the vertical nodes
        self.z_nodes = np.append(ztarget, zpadding)
        
        #calculate actual distances of depth layers
        self.z_grid = np.array([self.z_nodes[:ii+1].sum() 
                                for ii in range(self.z_nodes.shape[0])])
                                    
        self.mesh_values = np.zeros((self.x_grid.shape[0],
                                     self.z_grid.shape[0], 4), dtype=str)
        self.mesh_values[:,:,:] = '?'
        
        #get elevation if elevation_profile is given
        if self.elevation_profile is not None:
            self.add_elevation(self.elevation_profile)
        
        print '='*55
        print '{0:^55}'.format('mesh parameters'.upper())                            
        print '='*55
        print '  number of horizontal nodes = {0}'.format(self.x_grid.shape[0])  
        print '  number of vertical nodes   = {0}'.format(self.z_grid.shape[0])  
        print '  Total Horizontal Distance  = {0:2f}'.format(2*self.x_grid[-1])
        print '  Total Vertical Distance    = {0:2f}'.format(self.z_grid[-1])
        print '='*55
        
    def add_elevation(self, elevation_profile=None):
        """
        the elevation model needs to be in relative coordinates and be a 
        numpy.ndarray(2, num_elevation_points) where the first column is
        the horizontal location and the second column is the elevation at 
        that location.
        
        If you have a elevation model use Profile to project the elevation
        information onto the profile line
    
        To build the elevation I'm going to add the elevation to the top 
        of the model which will add cells to the mesh. there might be a better
        way to do this, but this is the first attempt. So I'm going to assume
        that the first layer of the mesh without elevation is the minimum
        elevation and blocks will be added to max elevation at an increment
        according to z1_layer
        
        .. note:: the elevation model should be symmetrical ie, starting 
                  at the first station and ending on the last station, so for
                  now any elevation outside the station area will be ignored 
                  and set to the elevation of the station at the extremities.
                  This is not ideal but works for now.
                  
        Arguments:
        -----------
            **elevation_profile** : np.ndarray(2, num_elev_points)
                                    - 1st row is for profile location
                                    - 2nd row is for elevation values
                                    
        Computes:
        ---------
            **mesh_values** : mesh values, setting anything above topography
                              to the key for air, which for Occam is '0'
        """
        if elevation_profile is not None:
            self.elevation_profile = elevation_profile
        
        if self.elevation_profile is None:
            raise OccamInputError('Need to input an elevation profile to '
                                  'add elevation into the mesh.')
                                  
        elev_diff = abs(elevation_profile[1].max()-elevation_profile[1].min())
        num_elev_layers = int(elev_diff/self.z1_layer)
        
        #add vertical nodes and values to mesh_values
        self.z_nodes = np.append([self.z1_layer]*num_elev_layers, self.z_nodes)
        self.z_grid = np.array([self.z_nodes[:ii+1].sum() 
                                for ii in range(self.z_nodes.shape[0])]) 
        #this assumes that mesh_values have not been changed yet and are all ?        
        self.mesh_values = np.zeros((self.x_grid.shape[0],
                                     self.z_grid.shape[0], 4), dtype=str)
        self.mesh_values[:,:,:] = '?'
        
        #--> need to interpolate the elevation values onto the mesh nodes
        # first center the locations about 0, this needs to be the same
        # center as the station locations.
        offset = elevation_profile[0]-elevation_profile[0].mean()
        elev = elevation_profile[1]-elevation_profile[1].min()
        
        func_elev = spi.interp1d(offset, elev, kind='linear')

        # need to figure out which cells and triangular cells need to be air
        xpad = self.num_x_pad_cells+1
        for ii, xg in enumerate(self.x_grid[xpad:-xpad], xpad):
            #get the index value for z_grid by calculating the elevation
            #difference relative to the top of the model
            dz = elev.max()-func_elev(xg)
            zz = int(np.ceil(dz/self.z1_layer))
            if zz == 0:
                pass
            else:
                #--> need to figure out the triangular elements 
                #top triangle
                zlayer = elev.max()-self.z_grid[zz]
                try:
                    xtop = xg+(self.x_grid[ii+1]-xg)/2
                    ytop = zlayer+3*(self.z_grid[zz]-self.z_grid[zz-1])/4
                    elev_top = func_elev(xtop)
                    #print xg, xtop, ytop, elev_top, zz
                    if elev_top > ytop:
                        self.mesh_values[ii, 0:zz, 0] = self.air_key
                    else:
                        self.mesh_values[ii, 0:zz-1, 0] = self.air_key
                except ValueError:
                    pass
                
                #left triangle
                try:
                    xleft = xg+(self.x_grid[ii+1]-xg)/4.
                    yleft = zlayer+(self.z_grid[zz]-self.z_grid[zz-1])/2.
                    elev_left = func_elev(xleft)
                    #print xg, xleft, yleft, elev_left, zz
                    if elev_left > yleft:
                        self.mesh_values[ii, 0:zz, 1] = self.air_key
                except ValueError:
                    pass
                
                #bottom triangle
                try:
                    xbottom = xg+(self.x_grid[ii+1]-xg)/2
                    ybottom = zlayer+(self.z_grid[zz]-self.z_grid[zz-1])/4
                    elev_bottom = func_elev(xbottom)
                    #print xg, xbottom, ybottom, elev_bottom, zz
                    if elev_bottom > ybottom:
                        self.mesh_values[ii, 0:zz, 2] = self.air_key
                except ValueError:
                    pass
                
                #right triangle
                try:
                    xright = xg+3*(self.x_grid[ii+1]-xg)/4
                    yright = zlayer+(self.z_grid[zz]-self.z_grid[zz-1])/2
                    elev_right = func_elev(xright)
                    #print xg, xright, yright, elev_right, zz
                    if elev_right > yright:
                        self.mesh_values[ii, 0:zz, 3] = self.air_key
                except ValueError:
                    pass
        #--> need to fill out the padding cells so they have the same elevation
        #    as the extremity stations.
        for ii in range(xpad):
            self.mesh_values[ii, :, :] = self.mesh_values[xpad+1, :, :]
        for ii in range(xpad+1):
            self.mesh_values[-(ii+1), :, :] = self.mesh_values[-xpad-2, :, :]
                       
                       
        print '{0:^55}'.format('--- Added Elevation to Mesh --')
    def plot_mesh(self, **kwargs):
        """
        plot mesh with station locations
        
        """
        fig_num = kwargs.pop('fig_num', 'Projected Profile')
        fig_size = kwargs.pop('fig_size', [5, 5])
        fig_dpi = kwargs.pop('fig_dpi', 300)
        marker = kwargs.pop('marker', r"$\blacktriangledown$")
        ms = kwargs.pop('ms', 5)
        mc = kwargs.pop('mc', 'k')
        lw = kwargs.pop('ls', .35)
        fs = kwargs.pop('fs', 6)
        plot_triangles = kwargs.pop('plot_triangles', 'n')
        
        depth_scale = kwargs.pop('depth_scale', 'km')
        
        #set the scale of the plot
        if depth_scale == 'km':
            dfactor = 1000.
        elif depth_scale == 'm':
            dfactor = 1.
        else:
            dfactor = 1000.
        
        plt.rcParams['figure.subplot.left'] = .12
        plt.rcParams['figure.subplot.right'] = .98
        plt.rcParams['font.size'] = fs
        
        if self.x_grid is None:
            self.build_mesh()
            
        fig = plt.figure(fig_num, figsize=fig_size, dpi=fig_dpi)
        ax = fig.add_subplot(1, 1, 1, aspect='equal')
        
        #plot the station marker
        #plots a V for the station cause when you use scatter the spacing
        #is variable if you change the limits of the y axis, this way it
        #always plots at the surface.
        for offset in self.station_locations:
            ax.text((offset)/dfactor,
                    0,
                    marker,
                    horizontalalignment='center',
                    verticalalignment='baseline',
                    fontdict={'size':ms,'color':mc})
                    

        #--> make list of column lines        
        row_line_xlist = []
        row_line_ylist = []
        for xx in self.x_grid/dfactor:
            row_line_xlist.extend([xx,xx])
            row_line_xlist.append(None)
            row_line_ylist.extend([0, self.z_grid[-1]/dfactor])
            row_line_ylist.append(None)
        
        #plot column lines (variables are a little bit of a misnomer)
        ax.plot(row_line_xlist, 
                row_line_ylist, 
                color='k', 
                lw=lw)

        #--> make list of row lines
        col_line_xlist = [self.x_grid[0]/dfactor, self.x_grid[-1]/dfactor]
        col_line_ylist = [0, 0]            
        for yy in self.z_grid/dfactor:
            col_line_xlist.extend([self.x_grid[0]/dfactor, 
                                  self.x_grid[-1]/dfactor])
            col_line_xlist.append(None)
            col_line_ylist.extend([yy, yy])
            col_line_ylist.append(None)
        
        #plot row lines (variables are a little bit of a misnomer)
        ax.plot(col_line_xlist, 
                col_line_ylist,
                color='k',
                lw=lw)
                    
        if plot_triangles == 'y':
            row_line_xlist = []
            row_line_ylist = []
            for xx in self.x_grid/dfactor:
                row_line_xlist.extend([xx,xx])
                row_line_xlist.append(None)
                row_line_ylist.extend([0, self.z_grid[-1]/dfactor])
                row_line_ylist.append(None)
                
            #plot columns
            ax.plot(row_line_xlist, 
                    row_line_ylist, 
                    color='k', 
                    lw=lw)
    
            col_line_xlist = []
            col_line_ylist = []            
            for yy in self.z_grid/dfactor:
                col_line_xlist.extend([self.x_grid[0]/dfactor, 
                                      self.x_grid[-1]/dfactor])
                col_line_xlist.append(None)
                col_line_ylist.extend([yy, yy])
                col_line_ylist.append(None)
            
            #plot rows
            ax.plot(col_line_xlist, 
                    col_line_ylist,
                    color='k',
                    lw=lw)
    
            diag_line_xlist = []
            diag_line_ylist = []
            for xi, xx in enumerate(self.x_grid[:-1]/dfactor):
                for yi, yy in enumerate(self.z_grid[:-1]/dfactor):
                    diag_line_xlist.extend([xx, self.x_grid[xi+1]/dfactor])
                    diag_line_xlist.append(None)
                    diag_line_xlist.extend([xx, self.x_grid[xi+1]/dfactor])
                    diag_line_xlist.append(None)
                    
                    diag_line_ylist.extend([yy, self.z_grid[yi+1]/dfactor])
                    diag_line_ylist.append(None)
                    diag_line_ylist.extend([self.z_grid[yi+1]/dfactor, yy])
                    diag_line_ylist.append(None)
            
            #plot diagonal lines.
            ax.plot(diag_line_xlist, 
                    diag_line_ylist,
                    color='k',
                    lw=lw)
                    
        #--> set axes properties
        ax.set_ylim(self.z_target_depth/dfactor, -2000/dfactor)
        xpad = self.num_x_pad_cells-1
        ax.set_xlim(self.x_grid[xpad]/dfactor, -self.x_grid[xpad]/dfactor)
        ax.set_xlabel('Easting ({0})'.format(depth_scale), 
                      fontdict={'size':fs+2, 'weight':'bold'})           
        ax.set_ylabel('Depth ({0})'.format(depth_scale), 
                  fontdict={'size':fs+2, 'weight':'bold'})           
        plt.show()
                                     
         
    def write_mesh_file(self, save_path=None, basename='Occam2DMesh'):
        """
        write a finite element mesh file.
        
        Arguments:
        -----------
            **save_path** : string
                            directory path or full path to save file
        Returns:
        ----------
            **mesh_fn** : string
                          full path to mesh file
                          
        :example: ::
        
            >>> import mtpy.modeling.occam2d as occam2d
            >>> edi_path = r"/home/mt/edi_files"
            >>> profile = occam2d.Profile(edi_path)
            >>> profile.plot_profile()
            >>> mesh = occam2d.Mesh(profile.station_locations)
            >>> mesh.build_mesh()
            >>> mesh.write_mesh_file(save_path=r"/home/occam2d/Inv1")
        """
        
        if save_path is not None:
            self.save_path = save_path
        
        if self.save_path is None:
            self.save_path = os.getcwd()
            
         
        self.mesh_fn = os.path.join(self.save_path, basename)

        mesh_lines = []
        nx = self.x_nodes.shape[0]
        nz = self.z_nodes.shape[0]
        mesh_lines.append('MESH FILE Created by mtpy.modeling.occam2d\n')
        mesh_lines.append("   {0}  {1}  {2}  {0}  {0}  {3}\n".format(0, nx ,
                          nz, 2))
        
        #--> write horizontal nodes
        node_str = ''
        for ii, xnode in enumerate(self.x_nodes):
            node_str += '{0:>9} '.format('{0:.1f}'.format(xnode))
            if np.remainder(ii+1, 8) == 0:
                node_str += '\n'
                mesh_lines.append(node_str)
                node_str = ''
                
        node_str += '\n'
        mesh_lines.append(node_str)
                
        #--> write vertical nodes
        node_str = ''
        for ii, znode in enumerate(self.z_nodes):
            node_str += '{0:>9} '.format('{0:.1f}'.format(znode))
            if np.remainder(ii+1, 8) == 0:
                node_str += '\n'
                mesh_lines.append(node_str)
                node_str = ''
        node_str += '\n'
        mesh_lines.append(node_str)

        #--> need a 0 after the nodes
        mesh_lines.append('    0\n')                
        #--> write triangular mesh block values as ?
        for zz in range(self.z_nodes.shape[0]):
            for tt in range(4):
                mesh_lines.append(''.join(self.mesh_values[:, zz, tt])+'\n')
            
        mfid = file(self.mesh_fn, 'w')
        mfid.writelines(mesh_lines)
        mfid.close()
        
        print 'Wrote Mesh file to {0}'.format(self.mesh_fn)
        
    def read_mesh_file(self, mesh_fn):
        """
        reads an occam2d 2D mesh file
        
        Arguments:
        ----------
            **mesh_fn** : string 
                          full path to mesh file
    
        Returns:
        --------
            **x_nodes**: array of horizontal nodes 
                                    (column locations (m))
                                    
            **z_nodes** : array of vertical nodes 
                                      (row locations(m))
                                      
            **mesh_values** : np.array of free parameters
            
        To do:
        ------
            incorporate fixed values
            
        :Example: ::
            
            >>> import mtpy.modeling.occam2d as occam2d 
            >>> mg = occam2d.Mesh()
            >>> mg.mesh_fn = r"/home/mt/occam/line1/Occam2Dmesh"
            >>> mg.read_mesh_file()
        """
        self.mesh_fn = mesh_fn
        
        mfid = file(self.mesh_fn,'r')
        
        mlines = mfid.readlines()
        
        nh = int(mlines[1].strip().split()[1])
        nv = int(mlines[1].strip().split()[2])
        
        self.x_nodes = np.zeros(nh)
        self.z_nodes=np.zeros(nv)
        self.mesh_values = np.zeros((nh, nv, 4), dtype=str)    
        
        #get horizontal nodes
        jj = 2
        ii = 0
        while ii < nh:
            hline = mlines[jj].strip().split()
            for mm in hline:
                self.x_nodes[ii] = float(mm)
                ii += 1
            jj += 1
        
        #get vertical nodes
        ii = 0
        while ii < nv:
            vline = mlines[jj].strip().split()
            for mm in vline:
                self.z_nodes[ii] = float(mm)
                ii += 1
            jj += 1    
        
        #get free parameters        
        for ii ,mm in enumerate(mlines[jj+1:]):
            kk = 0
            while kk < 4:        
                mline = mm.rstrip()
                if mline.lower().find('exception')>0:
                    break
                for jj in range(nh):
                    try:
                        self.mesh_values[jj,ii,kk] = mline[jj]
                    except IndexError:
                        pass
                kk += 1
         
class Profile():
    """
    Takes data from .edi files to create a profile line for 2D modeling.
    Can project the stations onto a profile that is perpendicular to strike
    or a given profile direction.
    
    Arguments:
    -----------
    
        **edi_path** : string
                       full path to edi files
                       
        **station_list** : list of stations to create profile for if None is
                           given all .edi files in edi_path will be used.
                           .. note:: that algorithm assumes .edi files are 
                                     named by station and it only looks for 
                                     the station within the .edi file name
                                     it does not match exactly, so if you have
                                     .edi files with similar names there
                                     might be some problems.
                                     
        **geoelectric_strike** : float
                                 geoelectric strike direction in degrees 
                                 assuming 0 is North and East is 90
                                 
        **profile_angle** : float
                            angle to project the stations onto a profile line
                            .. note:: the geoelectric strike angle and 
                                      profile angle should be orthogonal for
                                      best results from 2D modeling.
                                      
    
    ======================= ===================================================
    **Attributes**          Description    
    ======================= ===================================================
    edi_list                list of mtpy.core.mt.MT instances for each .edi
                            file found in edi_path 
    elevation_model         numpy.ndarray(3, num_elevation_points) elevation
                            values for the profile line (east, north, elev)
    geoelectric_strike      geoelectric strike direction assuming N == 0
    profile_angle           angle of profile line assuming N == 0
    profile_line            (slope, N-intercept) of profile line
    _profile_generated      [ True | False ] True if profile has already been 
                            generated
    edi_path                path to find .edi files
    station_list            list of stations to extract from edi_path
    num_edi                 number of edi files to create a profile for
    _rotate_to_strike       [ True | False] True to project the stations onto
                            a line that is perpendicular to geoelectric strike
                            also Z and Tipper are rotated to strike direction.
    ======================= ===================================================
 
    .. note:: change _rotate_to_strike to False if you want to project the 
              stations onto a given profile direction.  This does not rotate
              Z or Tipper
   
    ======================= ===================================================
    Methods                 Description
    ======================= ===================================================
    generate_profile        generates a profile for the given stations
    plot_profile            plots the profile line along with original station
                            locations to compare.  
    ======================= ===================================================
    
    :Example: ::
        
        >>> import mtpy.modeling.occam2d as occam
        >>> edi_path = r"/home/mt/edi_files"
        >>> station_list = ['mt{0:03}'.format(ss) for ss in range(0, 15)]
        >>> prof_line = occam.Profile(edi_path, station_list=station_list)
        >>> prof_line.plot_profile()
        >>> #if you want to project to a given strike
        >>> prof_line.geoelectric_strike = 36.7
        >>> prof_line.generate_profile()
        >>> prof_line.plot_profile()

        
    """
    
    def __init__(self, edi_path=None, **kwargs):
        
        self.edi_path = edi_path
        self.station_list = kwargs.pop('station_list', None)
        self.geoelectric_strike = kwargs.pop('geoelectri_strike', None)
        self.profile_angle = kwargs.pop('profile_angle', None)
        self.edi_list = []
        self._rotate_to_strike = True
        self.num_edi = 0
        self._profile_generated = False
        self.profile_line = None
        self.station_locations = None
        self.elevation_model = kwargs.pop('elevation_model', None)
        self.elevation_profile = None
        self.estimate_elevation = True
        
        
    def _get_edi_list(self):
        """
        get a list of edi files that coorespond to the station list
        
        each element of the list is a mtpy.core.mt.MT object
        """
        
        if self.station_list is not None:
            for station in self.station_list:
                for edi in os.listdir(self.edi_path):
                    if edi.find(station) == 0 and edi[-3:] == 'edi':
                        self.edi_list.append(mt.MT(os.path.join(self.edi_path, 
                                                                edi)))
                        break
        else:
            self.edi_list = [mt.MT(os.path.join(self.edi_path, edi)) for 
                             edi in os.listdir(self.edi_path) 
                             if edi[-3:]=='edi']
        
        self.num_edi = len(self.edi_list)
    
        
    def generate_profile(self):
        """
        Generate linear profile by regression of station locations.

        Stations are projected orthogonally onto the profile. Calculate 
        orientation of profile (azimuth) and position of stations on the 
        profile.

        Sorting along the profile is always West->East.
        (In unlikely/synthetic case of azimuth=0, it's North->South)
        
        """
        
        self._get_edi_list()
        
        strike_angles = np.zeros(self.num_edi)

        easts = np.zeros(self.num_edi)
        norths = np.zeros(self.num_edi)
        utm_zones = np.zeros(self.num_edi)

        for ii, edi in enumerate(self.edi_list):
            #find strike angles for each station if a strike angle is not given
            if self.geoelectric_strike is None:
                try:
                    #check dimensionality to be sure strike is estimate for 2D
                    dim = MTgy.dimensionality(edi.Z.z)
                    #get strike for only those periods
                    gstrike = MTgy.strike_angle(edi.Z.z[np.where(dim==2)])[:,0] 
                    strike_angles[ii] = np.median(gstrike)
                except:
                    pass

            easts[ii] = edi.east
            norths[ii] = edi.north
            utm_zones[ii] = int(edi.utm_zone[:-1])

        if len(self.edi_list) == 0:
            raise IOError('Could not find and .edi file in {0}'.format(self.edi_path))

        if self.geoelectric_strike is None:
            try:
                #might try mode here instead of mean
                self.geoelectric_strike = np.median(np.nonzero(strike_angles))
            except:
                #empty list or so....
                #can happen, if everyhing is just 1D
                self.geoelectric_strike = 0.

        #need to check the zones of the stations
        main_utmzone = mode(utm_zones)[0][0]

        for ii, zone in enumerate(utm_zones):
            if zone == main_utmzone:
                continue
            else:
                print ('station {0} is out of main utm zone'.format(self.edi_list[ii].station)+\
                       ' will not be included in profile')

        # check regression for 2 profile orientations:
        # horizontal (N=N(E)) or vertical(E=E(N))
        # use the one with the lower standard deviation
        profile1 = sp.stats.linregress(easts, norths)
        profile2 = sp.stats.linregress(norths, easts)
        profile_line = profile1[:2]
        #if the profile is rather E=E(N), the parameters have to converted 
        # into N=N(E) form:
        if profile2[4] < profile1[4]:
            profile_line = (1./profile2[0], -profile2[1]/profile2[0])
        self.profile_line = profile_line
        #profile_line = sp.polyfit(lo_easts, lo_norths, 1)
        if self.profile_angle is None:
            self.profile_angle = (90-(np.arctan(profile_line[0])*180/np.pi))%180
            
        #rotate Z according to strike angle, 

        #if strike was explicitely given, use that value!

        #otherwise:
        #have 90 degree ambiguity in strike determination
        #choose strike which offers larger angle with profile
        #if profile azimuth is in [0,90].

        if self._rotate_to_strike is False:
            if 0 <= self.profile_angle < 90:
                if np.abs(self.profile_angle-self.geoelectric_strike) < 45:
                    self.geoelectric_strike += 90
            elif 90 <= self.profile_angle < 135:
                if self.profile_angle-self.geoelectric_strike < 45:
                    self.geoelectric_strike -= 90
            else:
                if self.profile_angle-self.geoelectric_strike >= 135:
                    self.geoelectric_strike += 90
         
        self.geoelectric_strike = self.geoelectric_strike%180
        
        #rotate components of Z and Tipper to align with geoelectric strike
        #which should now be perpendicular to the profile strike
        if self._rotate_to_strike == True:
            self.profile_angle = self.geoelectric_strike+90
            p1 = np.tan(np.deg2rad(90-self.profile_angle))
            #need to project the y-intercept to the new angle
            p2 = (self.profile_line[0]-p1)*easts[0]+self.profile_line[1] 
            self.profile_line = (p1, p2)
            
            for edi in self.edi_list:
                edi.Z.rotate(self.geoelectric_strike-edi.Z.rotation_angle)
                # rotate tipper to profile azimuth, not strike.
                edi.Tipper.rotate((self.profile_angle-90)%180-edi.Tipper.rotation_angle)
           
            print '='*72
            print 'Rotated Z and Tipper to align with {0:+.2f} degrees E of N'.format(self.geoelectric_strike) 
            print 'Profile angle is {0:+.2f} degrees E of N'.format(self.profile_angle)        
            print '='*72
        
        #--> project stations onto profile line
        projected_stations = np.zeros((self.num_edi,2))
        self.station_locations = np.zeros(self.num_edi)
        
        #create profile vector
        profile_vector = np.array([1, self.profile_line[0]])
        #be sure the amplitude is 1 for a unit vector
        profile_vector /= np.linalg.norm(profile_vector)

        for ii, edi in enumerate(self.edi_list):
            station_vector = np.array([easts[ii], norths[ii]-self.profile_line[1]])
            position = np.dot(profile_vector, station_vector)*profile_vector 
            self.station_locations[ii] = np.linalg.norm(position)
            edi.offset = np.linalg.norm(position)
            edi.projected_east = position[0]
            edi.projected_north = position[1]+self.profile_line[1]
            projected_stations[ii] = [position[0], position[1]+self.profile_line[1]]

        #set the first station to 0
        for edi in self.edi_list:
            edi.offset -= self.station_locations.min()
        self.station_locations -= self.station_locations.min()
        

        #Sort from West to East:
        index_sort = np.argsort(self.station_locations)
        if self.profile_angle == 0:
            #Exception: sort from North to South
            index_sort = np.argsort(norths)


        #sorting along the profile
        self.edi_list = [self.edi_list[ii] for ii in index_sort]
        self.station_locations = np.array([self.station_locations[ii] 
                                           for ii in index_sort])
        if self.estimate_elevation == True:
            self.project_elevation()
         
        self._profile_generated = True
        
    def project_elevation(self, elevation_model=None):
        """
        projects elevation data into the profile
        
        Arguments:
        -------------
            **elevation_model** : np.ndarray(3, num_elevation_points)
                                  (east, north, elevation)
                                  for now needs to be in utm coordinates
                                  if None then elevation is taken from edi_list
                                  
        Returns:
        ----------
            **elevation_profile** : 
        """
        self.elevation_model = elevation_model
        
        
        #--> get an elevation model for the mesh                                       
        if self.elevation_model == None:
            self.elevation_profile = np.zeros((2, len(self.edi_list)))
            self.elevation_profile[0,:] = np.array([ss 
                                            for ss in self.station_locations])
            self.elevation_profile[1,:] = np.array([edi.elev 
                                                  for edi in self.edi_list])
        
        #--> project known elevations onto the profile line
        else:
            self.elevation_profile = np.zeros((2, self.elevation_model.shape[1]))
            #create profile vector
            profile_vector = np.array([1, self.profile_line[0]])
            #be sure the amplitude is 1 for a unit vector
            profile_vector /= np.linalg.norm(profile_vector)
            for ii in range(self.elevation_model.shape[1]):
                east = self.elevation_model[0, ii]
                north = self.elevation_model[1, ii]
                elev = self.elevation_model[2, ii]
                elev_vector = np.array([east, north-self.profile_line[1]])
                position = np.dot(profile_vector, elev_vector)*profile_vector 
                self.elevation_profile[0, ii] = np.linalg.norm(position)
                self.elevation_profile[1, ii] = elev
                

    def plot_profile(self, **kwargs):
        """
        plot the projected profile line
        
        """
        
        fig_num = kwargs.pop('fig_num', 'Projected Profile')
        fig_size = kwargs.pop('fig_size', [5, 5])
        fig_dpi = kwargs.pop('fig_dpi', 300)
        marker = kwargs.pop('marker', 'v')
        ms = kwargs.pop('ms', 5)
        mc = kwargs.pop('mc', 'k')
        lc = kwargs.pop('lc', 'b')
        lw = kwargs.pop('ls', 1)
        fs = kwargs.pop('fs', 6)
        station_id = kwargs.pop('station_id', None)
        
        plt.rcParams['figure.subplot.left'] = .12
        plt.rcParams['figure.subplot.right'] = .98
        plt.rcParams['font.size'] = fs
        
        if self._profile_generated is False:
            self.generate_profile()
            
        fig = plt.figure(fig_num, figsize=fig_size, dpi=fig_dpi)
        ax = fig.add_subplot(1, 1, 1, aspect='equal')

        for edi in self.edi_list:
            m1, = ax.plot(edi.projected_east, edi.projected_north, 
                         marker=marker, ms=ms, mfc=mc, mec=mc, color=lc)
                    
            m2, = ax.plot(edi.east, edi.north, marker=marker,
                         ms=.5*ms, mfc=(.6, .6, .6), mec=(.6, .6, .6), 
                         color=lc)
            
            if station_id is None:
                ax.text(edi.projected_east, edi.projected_north*1.00025, 
                        edi.station,
                        ha='center', va='baseline',
                        fontdict={'size':fs, 'weight':'bold'})
            else:
                ax.text(edi.projected_east, edi.projected_north*1.00025, 
                        edi.station[station_id[0]:station_id[1]],
                        ha='center', va='baseline',
                        fontdict={'size':fs, 'weight':'bold'})
        
        peasts = np.array([edi.projected_east for edi in self.edi_list])
        pnorths = np.array([edi.projected_north for edi in self.edi_list])
        easts = np.array([edi.east for edi in self.edi_list])
        norths = np.array([edi.north for edi in self.edi_list])
        
        ploty = sp.polyval(self.profile_line, easts)
        ax.plot(easts, ploty, lw=lw, color=lc)
        ax.set_title('Original/Projected Stations')
        ax.set_ylim((min([norths.min(), pnorths.min()])*.999,
                     max([norths.max(), pnorths.max()])*1.001))
        ax.set_xlim((min([easts.min(), peasts.min()])*.98,
                     max([easts.max(), peasts.max()])*1.02))
        ax.set_xlabel('Easting (m)', 
                       fontdict={'size':fs+2, 'weight':'bold'})
        ax.set_ylabel('Northing (m)',
                       fontdict={'size':fs+2, 'weight':'bold'})
        ax.grid(alpha=.5)
        ax.legend([m1, m2], ['Projected', 'Original'], loc='upper left',
                  prop={'size':fs})
                  
        plt.show()
        
class Regularization(Mesh):
    """
    Creates a regularization grid based on Mesh.  Note that Mesh is inherited
    by Regularization, therefore the intended use is to build a mesh with 
    the Regularization class.
    
    """
    
    def __init__(self, station_locations=None, **kwargs):
        # Be sure to initialize Mesh        
        Mesh.__init__(self, station_locations, **kwargs)
        
        self.min_block_width = kwargs.pop('min_block_width', 
                                          2*np.median(self.cell_width))
        self.trigger = kwargs.pop('trigger', .75)
        self.model_columns = None
        self.model_rows = None
        self.binding_offset = None
        self.reg_fn = None
        self.reg_basename = 'Occam2DModel'
        self.description = 'model made by mtpy.modeling.occam2d'
        self.num_param = None
        
        
        #--> build mesh         
        if self.station_locations is not None:
            self.build_mesh()
            self.build_regularization()
            
    def build_regularization(self):
        """
        builds larger boxes around existing mesh blocks for the regularization
    
        """
        # list of the mesh columns to combine
        self.model_columns = []
        # list of mesh rows to combine
        self.model_rows = []
        
        #At the top of the mesh model blocks will be 2 combined mesh blocks
        #Note that the padding cells are combined into one model block
        station_col = [2]*((self.x_nodes.shape[0]-2*self.num_x_pad_cells)/2)
        model_cols = [self.num_x_pad_cells]+station_col+[self.num_x_pad_cells]
        station_widths = [self.x_nodes[ii]+self.x_nodes[ii+1] for ii in 
                       range(self.num_x_pad_cells, 
                             self.x_nodes.shape[0]-self.num_x_pad_cells, 2)]
                             
        pad_width = self.x_nodes[0:self.num_x_pad_cells].sum()
        model_widths = [pad_width]+station_widths+[pad_width]
        num_cols = len(model_cols)
        
        model_thickness = np.append(self.z_nodes[0:self.z_nodes.shape[0]-self.num_z_pad_cells], 
                                    self.z_nodes[-self.num_z_pad_cells:].sum())
        
        self.num_param = 0
        #--> now need to calulate model blocks to the bottom of the model
        columns = list(model_cols)
        widths = list(model_widths)
        for zz, thickness in enumerate(model_thickness):
            #index for model column blocks from first_row, start at 1 because
            # 0 is for padding cells            
            block_index = 1
            num_rows = 1
            if zz == 0:
                num_rows += 1
            if zz == len(model_thickness)-1:
                num_rows = self.num_z_pad_cells
            while block_index+1 < num_cols-1:
                #check to see if horizontally merged mesh cells are not larger
                #than the thickness times trigger
                if thickness < self.trigger*(widths[block_index]+\
                                             widths[block_index+1]):
                    block_index += 1
                    continue
                #merge 2 neighboring cells to avoid vertical exaggerations                    
                else:
                    widths[block_index] += widths[block_index+1]
                    columns[block_index] += columns[block_index+1]
                    #remove one of the merged cells                        
                    widths.pop(block_index+1)
                    columns.pop(block_index+1)
                    
                    num_cols -= 1
            self.num_param += num_cols

            self.model_columns.append(list(columns))
            self.model_rows.append([num_rows, num_cols])
                
        #calculate the distance from the right side of the furthest left 
        #model block to the furthest left station.
        self.binding_offset = self.x_grid[self.num_x_pad_cells]
        
        print '='*55
        print '{0:^55}'.format('regularization parameters'.upper())
        print '='*55
        print '   binding offset       = {0:.1f}'.format(self.binding_offset)
        print '   number layers        = {0}'.format(len(self.model_columns))
        print '   number of parameters = {0}'.format(self.num_param)
        print '='*55

        
    def write_regularization_file(self, reg_fn=None, reg_basename=None, 
                                  statics_fn='None', prejudice_fn='None',
                                  save_path=None):
        """
        write a regularization file for input into occam.
        
        if reg_fn is None, then file is written to save_path/reg_basename
        
        Arguments:
        ----------
            **reg_fn** : string
                         full path to regularization file. *default* is None
                         and file will be written to save_path/reg_basename
                         
            **reg_basename** : string
                               basename of regularization file
                               
                                
        """
        if save_path is not None:
            self.save_path = save_path
        if reg_basename is not None:
            self.reg_basename = reg_basename
        if reg_fn is None:
            if self.save_path is None:
                self.save_path = os.getcwd()
            self.reg_fn = os.path.join(self.save_path, self.reg_basename)
        
        reg_lines = []
        
        #--> write out header information
        reg_lines.append('{0:<18}{1}\n'.format('format:'.upper(), 
                                               'occam2dmtmod_1.0'.upper()))
        reg_lines.append('{0:<18}{1}\n'.format('model name:'.upper(), 
                                               self.description.upper()))
        reg_lines.append('{0:<18}{1}\n'.format('mesh file:'.upper(), 
                                               self.mesh_fn))
        reg_lines.append('{0:<18}{1}\n'.format('mesh type:'.upper(), 
                                               'pw2d'.upper()))
        reg_lines.append('{0:<18}{1}\n'.format('statics file:'.upper(), 
                                               statics_fn))
        reg_lines.append('{0:<18}{1}\n'.format('prejudice file:'.upper(), 
                                               prejudice_fn))
        reg_lines.append('{0:<18}{1:.1f}\n'.format('binding offset:'.upper(), 
                                                   self.binding_offset))
        reg_lines.append('{0:<18}{1}\n'.format('num layers:'.upper(), 
                                               len(self.model_columns)))
        
        #--> write rows and columns of regularization grid                                        
        for row, col in zip(self.model_rows, self.model_columns):
            reg_lines.append(''.join(['{0:>5}'.format(rr) for rr in row])+'\n')
            reg_lines.append(''.join(['{0:>5}'.format(cc) for cc in col])+'\n')
                                    
        rfid = file(self.reg_fn, 'w')
        rfid.writelines(reg_lines)
        rfid.close()
        
        print 'Wrote Regularization file to {0}'.format(self.reg_fn)
        
    def read_regularization_file(self, reg_fn):
        """
        read in a regularization file
        
        """
        self.reg_fn = reg_fn
        self.save_path = os.path.dirname(reg_fn)

        rfid = open(self.reg_fn, 'r')
        
        headerdict = {}
        self.model_rows = []
        self.model_columns = []    
        ncols = []
        
        rlines = rfid.readlines()
        
        for ii, iline in enumerate(rlines):
            #read header information
            if iline.find(':') > 0:
                iline = iline.strip().split(':')
                headerdict[iline[0].lower()] = iline[1].strip()
                #append the last line
                if iline[0].lower().find('exception') > 0:
                    self.model_columns.append(ncols)
            
            #get mesh values
            else:
                iline = iline.strip().split()
                iline = [int(jj) for jj in iline]
                if len(iline) == 2:
                    if len(ncols) > 0:
                        self.model_columns.append(ncols)
                    self.model_rows.append(iline)
                    ncols = []
                elif len(iline) > 2:
                    ncols = ncols+iline
                    
        #set mesh file name
        self.mesh_fn = headerdict['mesh file']
        if not os.path.isfile(self.mesh_fn):
            self.mesh_fn = os.path.join(self.save_path, self.mesh_fn)
        
        #set statics file name
        self.statics_fn = headerdict['statics file']
        if not os.path.isfile(self.mesh_fn):
            self.statics_fn = os.path.join(self.save_path, self.statics_fn)
            
        #set prejudice file name
        self.prejudice_fn = headerdict['prejudice file']
        if not os.path.isfile(self.mesh_fn):
            self.prejudice_fn = os.path.join(self.save_path, self.prejudice_fn)
   

class Startup(object):
    """
    deals with startup file for occam2d
    
    """
    
    def __init__(self, **kwargs):
        self.save_path = kwargs.pop('save_path', None)
        self.startup_basename = kwargs.pop('startup_basename', 'Occam2DStartup')
        self.startup_fn = kwargs.pop('startup_fn', None)
        self.model_fn = kwargs.pop('model_fn', None)
        self.data_fn = kwargs.pop('data_fn', None)
        self.description = kwargs.pop('description', 'startup created by mtpy')
        self.max_iteration = kwargs.pop('max_iteration', 20)
        self.roughness_type = kwargs.pop('roughness_type', 1)
        self.target_misfit = kwargs.pop('target_misfit', 1.0)
        self.diagonal_penalties = kwargs.pop('diagonal_penalties', 0)
        self.stepsize_count = kwargs.pop('stepsize_count', 8)
        self.model_limits = kwargs.pop('model_limits', None)
        self.model_value_steps = kwargs.pop('model_value_steps', None)
        self.debug_level = kwargs.pop('debug_level', 1)
        self.iteration_num = kwargs.pop('iteration_num', 0)
        self.lagrange_value = kwargs.pop('lagrange_value', 5.0)
        self.roughness_value = kwargs.pop('roughness_value', 1e10)
        self.misfit_value = kwargs.pop('misfit_value', 1000)
        self.misfit_reached = kwargs.pop('misfit_reached', 0)
        self.param_count = kwargs.pop('param_count', None)
        self.resistivity_start = kwargs.pop('resistivity_start', 2)
        
    def write_startup_file(self, startup_fn=None, save_path=None, 
                           startup_basename=None):
        """
        write a startup file based on the parameters of startup class
        
        """
        if save_path is not None:
            self.save_path = save_path
            
        if self.save_path is None:
            self.save_path = os.getcwd()
        if startup_basename is not None:
            self.startup_basename = startup_basename
            
        if startup_fn is None:
            self.startup_fn = os.path.join(self.save_path, 
                                           self.startup_basename)

        #--> check to make sure all the important input are given                                           
        if self.data_fn is None:
            raise OccamInputError('Need to input data file name')
        
        if self.model_fn is None:
            raise OccamInputError('Need to input model/regularization file name')
        
        if self.param_count is None:
            raise OccamInputError('Need to input number of model parameters')

        slines = []
        slines.append('{0:<20}{1}\n'.format('Format:','OCCAMITER_FLEX'))
        slines.append('{0:<20}{1}\n'.format('Description:', self.description))
        slines.append('{0:<20}{1}\n'.format('Model File:', self.model_fn))
        slines.append('{0:<20}{1}\n'.format('Data File:', self.data_fn))
        slines.append('{0:<20}{1}\n'.format('Date/Time:', time.ctime()))
        slines.append('{0:<20}{1}\n'.format('Iterations to run:',
                                            self.max_iteration))
        slines.append('{0:<20}{1}\n'.format('Target Misfit:', 
                                            self.target_misfit))
        slines.append('{0:<20}{1}\n'.format('Roughness Type:', 
                                            self.roughness_type))
        slines.append('{0:<20}{1}\n'.format('Diagonal Penalties:', 
                                            self.diagonal_penalties))
        slines.append('{0:<20}{1}\n'.format('Stepsize Cut Count:', 
                                            self.stepsize_count))
        if self.model_limits is None:
            slines.append('{0:<20}{1}\n'.format('!Model Limits:', 'none'))
        else:
            slines.append('{0:<20}{1},{2}\n'.format('Model Limits:', 
                                                self.model_limits[0], 
                                                self.model_limits[1]))
        if self.model_value_steps is None:
            slines.append('{0:<20}{1}\n'.format('!Model Value Steps:', 'none'))
        else:
            slines.append('{0:<20}{1}\n'.format('Model Value Steps:',
                                                self.model_value_steps))
        slines.append('{0:<20}{1}\n'.format('Debug Level:', self.debug_level))
        slines.append('{0:<20}{1}\n'.format('Iteration:', self.iteration_num))
        slines.append('{0:<20}{1}\n'.format('Lagrange Value:', 
                                            self.lagrange_value))
        slines.append('{0:<20}{1}\n'.format('Roughness Value:', 
                                            self.roughness_value))
        slines.append('{0:<20}{1}\n'.format('Misfit Value:', self.misfit_value))
        slines.append('{0:<20}{1}\n'.format('Misfit Reached:', 
                                            self.misfit_reached))
        slines.append('{0:<20}{1}\n'.format('Param Count:', self.param_count))
        
        #write out starting resistivity values
        sline = []
        for ii in range(self.param_count):
            sline.append('{0:^10}'.format('{0:.4f}'.format(self.resistivity_start)))
            if np.remainder(ii+1, 4) == 0:
                sline.append('\n')
                slines.append(''.join(list(sline)))
                sline = []
        slines.append(''.join(list(sline+['\n'])))
        #--> write file
        sfid = file(self.startup_fn, 'w')
        sfid.writelines(slines)
        sfid.close()
        
        print 'Wrote Occam2D startup file to {0}'.format(self.startup_fn)
                
#------------------------------------------------------------------------------
class Data(Profile):
    """
    Handling input data.

    Generation of suitable Occam data file(s) from Edi files/directories.
    Reading and writing data files.
    Allow merging of data files.
    """
    def __init__(self, edi_path=None, **kwargs):
        Profile.__init__(self, edi_path, **kwargs)
        
        self.data_basename = kwargs.pop('data_fn', 'OccamDataFile.dat')
        self.save_path = kwargs.pop('save_path', None)
        self.frequency_list = kwargs.pop('frequency_list', None)
        self.model_mode = kwargs.pop('model_mode', 'all')

        self.frequencies = None
        self.data = None
        
        self.reste_err = kwargs.pop('reste_err', 10)
        self.restm_err = kwargs.pop('restm_err', 10)
        self.phasete_err = kwargs.pop('phasete_err', 5)
        self.phasetm_err = kwargs.pop('phasetm_err', 5)
        self.tipper_err = kwargs.pop('tipper_err', None)
        
        self.freq_dict = kwargs.pop('freq_dict', None)
        self.freq_step = kwargs.pop('freq_step', None)
        self.freq_tol = kwargs.pop('freq_tol', 0.05)

        self.format = 'OCCAM2MTDATA_1.0'
        self.title = 'MTpy-OccamDatafile'
        self.edi_type = 'z'


    def readfile(self,fn):
        if not op.isfile(fn):
            print 'Error - not a valid file: {0}'.fn

        self.filename = op.basename(fn)
        self.wd = op.split(fn)[0]

        F_in = file(fn,'r')
        datafile_raw = F_in.read()
        F_in.close()

        #string is reduced each step, i.e. cut off the sections, 
        #which are already read in
        reduced_string = self._read_format(datafile_raw)
        reduced_string = self._read_title(datafile_raw)
        reduced_string = self._read_sites(datafile_raw)
        reduced_string = self._read_offsets(datafile_raw)
        reduced_string = self._read_frequencies(datafile_raw)
        
        self._read_data(reduced_string)

    def _find_string(key,datastring):
        
        index = datastring.lower().find('key')
        return index

    def _read_format(self,datastring):
        idx = _find_string('format',datastring)
        reduced_string = datastring[idx:]
        data_list = datastring.split('\n')
        line = data_list[0]
        line = line.strip().split(':')
        self.format = line[1].strip().lower()

        return reduced_string 

    def _read_title(self,datastring):
        idx = _find_string('title',datastring)
        reduced_string = datastring[idx:]
        data_list = datastring.split('\n')
        line = data_list[0]
        line = line.strip().split(':')
        self.title = line[1].strip().lower()

        return reduced_string 
        

    def _read_sites(self,datastring):

        idx = _find_string('sites',datastring)
        reduced_string = datastring[idx:]
        data_list = datastring.split('\n')
        line = data_list[0]
        line = line.strip().split(':')
        no_sites = int(float(line[1].strip().lower()))
        lo_stations = []
        for idx in range(no_sites):
            sta = data_list[idx+1].strip()
            lo_stations.append(sta)

        self.stations = lo_stations

        return reduced_string 
        

    def _read_offsets(self,datastring):
        idx = _find_string('offsets',datastring)
        reduced_string = datastring[idx:]
        data_list = datastring.split('\n')
        line = data_list[0]
        line = line.strip().split(':')
        no_sites = len(self.stations)
        lo_offsets = []
        for idx in range(no_sites):
            offset = float(data_list[idx+1].strip())
            lo_offsets.append(offset)

        self.stationlocations = lo_offsets

        return reduced_string 
        

    def _read_frequencies(self,datastring):
        idx = _find_string('frequencies',datastring)
        reduced_string = datastring[idx:]
        data_list = datastring.split('\n')
        line = data_list[0]
        line = line.strip().split(':')
        no_freqs = int(float(line[1]))

        lo_freqs = []
        for idx in range(no_freqs):
            freq = float(data_list[idx+1].strip())
            lo_freqs.append(freq)

        self.frequencies = lo_freqs

        return reduced_string 

    def _read_data(self,datastring):
        idx = _find_string('data',datastring)
        reduced_string = datastring[idx:]
        data_list = datastring.split('\n')
        line = data_list[0]
        line = line.strip().split(':')
        no_data = int(float(line[1]))

        lo_data = []
        idx = 0
        row_idx = 2
        while idx < no_data:
            row = data_list[row_idx].strip().split()
            if row[0][0] == '#':
                row_idx += 1
                continue
            rowlist = [float(i) for i in row]
            lo_data.append(rowlist)
            row_idx += 1
            idx += 1


    def build_data(self):
        """Data file Generation

        Read all Edi files. 
        Extract frequencies. 
        Read in strike. If strike = None: find average strike over all stations and frequencies. 
        90 degree strike ambiguity leads to choice of strike: Larger angle with profile line is chosen.
        Rotate Z and Tipper: X components are along strike, Y orthogonal.  
        Extract off-diagonal data from Z. Extract Tipper x-component (along profile).

        Collect all information sorted according to occam specifications.

        Data of Z given in muV/m/nT = km/s
        Error is assumed to be 1 stddev.
        """ 
        

        #set data modes
        lo_modes = []
        modes = self.mode.lower().strip()
        
        if 'both' in modes :
            lo_modes.extend([9,10,2,6])  
        if 'te' in modes:
            lo_modes.extend([9,2])
        if 'tm' in modes:
            lo_modes.extend([10,6])
        if ('tipper' in modes): 
            lo_modes.extend([3,4])
        if 'all' in modes :
            lo_modes.extend([9,10,2,6,3,4])  

        lo_modes = sorted(list(set(lo_modes))) 

        #set data frequencies
        min_freq = self.min_frequency
        max_freq = self.max_frequency
        no_freqs_max = self.max_no_frequencies
 

        lo_all_freqs = []
        for lo_f in self.station_frequencies:
            lo_all_freqs.extend(list(lo_f))
        lo_all_freqs = sorted(list(set(lo_all_freqs)),reverse=True)

        if (min_freq is None) or (min_freq < min(lo_all_freqs) ) or (min_freq > max(lo_all_freqs) ) :
            min_freq = min(lo_all_freqs)
        if (max_freq is None) or (max_freq > max(lo_all_freqs) ) or (max_freq < min(lo_all_freqs) ) :
            max_freq = max(lo_all_freqs)
        
        lo_all_freqs_tmp = []
        for f in  lo_all_freqs:
            if min_freq <= f <= max_freq :
                lo_all_freqs_tmp.append(f)
            else:
                continue

        if len(lo_all_freqs_tmp) == 0 :
            print 'No frequencies in user-defined interval [{0},{1}]'.format(min_freq, max_freq)
            sys.exit()


        #check, if frequency list is longer than given max value
        if no_freqs_max is not None:
            no_freqs_max = int(float(no_freqs_max))
            if no_freqs_max < len(lo_all_freqs_tmp):
                lo_all_freqs_tmp2 = []
                excess = len(lo_all_freqs_tmp)/float(no_freqs_max)
                if excess < 2:
                    offset = 0

                else:
                    stepsize = (len(lo_all_freqs_tmp)-1)/no_freqs_max
                    offset = stepsize/2.
                indices = np.array(np.around(np.linspace(offset,len(lo_all_freqs_tmp)-1-offset,no_freqs_max),0))
                if indices[0]>(len(lo_all_freqs_tmp)-1-indices[-1]):
                    indices -= 1
                for idx in indices:
                    index = int(np.round(idx,0))+1
                    lo_all_freqs_tmp2.insert(0,lo_all_freqs_tmp[-index])
                

                lo_all_freqs_tmp = lo_all_freqs_tmp2

        self.frequencies = np.array(lo_all_freqs_tmp)


        #collect data 
        self.data = []

        for idx_s, station in enumerate(self.stations):
            station_number = idx_s + 1
            Z = self.Z[idx_s]
            T = self.Tipper[idx_s]

            rho = Z.resistivity
            phi = Z.phase
            rho_err = Z.resistivity_err
            phi_err = Z.phase_err
            z_array = Z.z
            zerr_array = Z.zerr

            for freq_num,freq in enumerate(self.frequencies):

                frequency_number = freq_num + 1 #OCCAM indices start with 1 

                #extract the freqs available for the respective station
                station_freqs = self.station_frequencies[idx_s]
                #skip, if the listed frequency is not available for the station
                if not (freq in station_freqs):
                    continue

                #find the respective frequency index for the station     
                idx_f = np.abs(station_freqs-freq).argmin()

                for mode in lo_modes:
                    if mode in [9,2] :
                        raw_rho_value = rho[idx_f][0,1]
                        value = raw_rho_value
                        #value = np.log10(raw_rho_value)
                        absolute_rho_error = rho_err[idx_f][0,1]
                        try:
                            if raw_rho_value == 0:
                                raise
                            relative_rho_error = np.abs(absolute_rho_error/raw_rho_value)
                        except:
                            relative_rho_error = 0.

                        if mode == 9 :
                            if self.rho_errorfloor is not None:
                                if self.rho_errorfloor/100. > relative_rho_error:
                                    relative_rho_error = self.rho_errorfloor/100.
                            error = np.abs(relative_rho_error * raw_rho_value)   #relative_error/np.log(10.)
                            #error = np.abs(relative_rho_error/np.log(10.))

                        elif mode == 2 :
                            raw_phi_value = phi[idx_f][0,1]
                            if raw_phi_value >=180:
                                raw_phi_value -= 180
                            value = raw_phi_value
                            if self.phase_errorfloor is not None:
                                if self.phase_errorfloor/100. > relative_rho_error:
                                    relative_rho_error = self.phase_errorfloor/100.
                            if relative_rho_error >= 2.:
                                error = 180.
                            else:
                                error = np.degrees(np.arcsin(0.5*relative_rho_error))#relative_error*100.*0.285

                    if mode in [10,6] :
                        raw_rho_value = rho[idx_f][1,0]
                        value = raw_rho_value
                        #value = np.log10(raw_rho_value)
                        absolute_rho_error = rho_err[idx_f][1,0]
                        try:
                            if raw_rho_value == 0:
                                raise
                            relative_rho_error = np.abs(absolute_rho_error/raw_rho_value)
                        except:
                            relative_rho_error = 0.
                        if mode == 10 :
                            if self.rho_errorfloor is not None:
                                if self.rho_errorfloor/100. > relative_rho_error:
                                    relative_rho_error = self.rho_errorfloor/100.
                            error = np.abs(relative_rho_error * raw_rho_value)   #relative_error/np.log(10.)
                            #error = np.abs(relative_rho_error /np.log(10.))

                        elif mode == 6 :
                            raw_phi_value = phi[idx_f][1,0]
                            if raw_phi_value >=180:
                                raw_phi_value -= 180
                            value = raw_phi_value
                            if self.phase_errorfloor is not None:
                                if self.phase_errorfloor/100. > relative_rho_error:
                                    relative_rho_error = self.phase_errorfloor/100.
                            if relative_rho_error >= 2.:
                                error = 180.
                            else:
                                error = np.degrees(np.arcsin(0.5*relative_rho_error))#relative_error*100.*0.285
                 
                    elif mode in [3,4] :
                        if T.tipper is None:
                            #print 'no Tipper data for {0} Hz at station {1}'.format(freq, station_number) 
                            continue

                        tipper = T.tipper[idx_f]
                        try: 
                            tippererr = T.tippererr[idx_f]
                        except:
                            #print 'no Tipper error for station {0}/frequency {1}'.format(station_number,frequency_number)
                            tippererr = None


                        if mode == 3 :
                            value = np.real(tipper[0,1])

                        if mode == 4 :
                            value = np.imag(tipper[0,1])


                        if tippererr is None:
                            raw_error = 0
                            if self.tipper_errorfloor is not None:
                                raw_error = (self.tipper_errorfloor/100.)*value
                        else:
                            raw_error = tippererr[0,1] 
                            
                        if value == 0 :
                            rel_error = 0 
                        else:
                            rel_error = raw_error

                        error = raw_error/value
                        if self.tipper_errorfloor is not None:                                
                            if self.tipper_errorfloor/100. > rel_error:
                                error = (self.tipper_errorfloor/100.)#*value
                            

                    self.data.append([station_number,frequency_number,mode,value,np.abs(error)])

    def generate_profile(self):
        """
            Generate linear profile by regression of station locations.

            Stations are projected orthogonally onto the profile. Calculate 
            orientation of profile (azimuth) and position of stations on the 
            profile.

            Sorting along the profile is always West->East.
            (In unlikely/synthetic case of azimuth=0, it's North->South)


            (self.stationlocations, self.azimuth, self.stations)

        """


        self.station_coords = []
        self.stations = []
        self.station_frequencies = []
        
        self.Z = []
        self.Tipper = []

        lo_strike_angles = []

        lo_easts = []
        lo_norths = []
        utmzones = []

        lo_wrong_edifiles = []

        for edifile in self.edilist:
            edi = MTedi.Edi()
            try:
                edi.readfile(edifile,datatype=self.edi_type)
            except:
                lo_wrong_edifiles.append(edifile)
                continue

            if self.strike is None:
                try:
                    lo_strike_angles.extend(list(MTgy.strike_angle(edi.Z.z[np.where(MTgy.dimensionality(edi.Z.z)!=1)])[:,0]%90))
                except:
                    pass
            self.station_coords.append([edi.lat,edi.lon,edi.elev])
            self.stations.append(edi.station)
            self.station_frequencies.append(np.around(edi.freq,5))
            try:
                self.Tipper.append(edi.Tipper)
            except:
                self.Tipper.append(None)
                
            self.Z.append(edi.Z)
            utm = MTcv.LLtoUTM(23,edi.lat,edi.lon)
            lo_easts.append(utm[1])
            lo_norths.append(utm[2])
            utmzones.append(int(utm[0][:-1]))
       

        for i in lo_wrong_edifiles:
            self.edilist.remove(i)

        if len(self.edilist) == 0:
            raise

        if self.strike is None:
            try:
                self.strike = np.mean(lo_strike_angles)
            except:
                #empty list or so....
                #can happen, if everyhing is just 1D
                self.strike = 0.

        main_utmzone = mode(utmzones)[0][0]


        for idx, zone in enumerate(utmzones):
            if zone == main_utmzone:
                continue
            utm = MTcv.LLtoUTM(23,edi.lat,edi.lon,main_utmzone)

            lo_easts[idx] = utm[1]
            lo_norths[idx] = utm[2]
        
        lo_easts = np.array(lo_easts)
        lo_norths = np.array(lo_norths)

        # check regression for 2 profile orientations:
        # horizontal (N=N(E)) or vertical(E=E(N))
        # use the one with the lower standard deviation
        profile1 = sp.stats.linregress(lo_easts, lo_norths)
        profile2 = sp.stats.linregress(lo_norths, lo_easts)
        profile_line = profile1[:2]
        #if the profile is rather E=E(N), the parameters have to converted 
        # into N=N(E) form:
        if profile2[4]<profile1[4]:
            profile_line = (1./profile2[0], -profile2[1]/profile2[0])

        #profile_line = sp.polyfit(lo_easts, lo_norths, 1) 
        self.azimuth = (90-(np.arctan(profile_line[0])*180/np.pi))%180


        
        #rotate Z according to strike angle, 

        #if strike was explicitely given, use that value!

        #otherwise:
        #have 90 degree ambiguity in strike determination
        #choose strike which offers larger angle with profile
        #if profile azimuth is in [0,90].

        if self._strike_set is False:
            if 0 <= self.azimuth < 90:
                if np.abs(self.azimuth - self.strike) < 45:
                    self.strike += 90
            elif 90 <= self.azimuth < 135:
                if self.azimuth - self.strike < 45:
                    self.strike -= 90
            else:
                if self.azimuth - self.strike >= 135:
                    self.strike += 90
         

        self.strike = self.strike%180


        rotation_angle = self.strike
        
        for old_z in self.Z:
            original_rotation_angle = np.array(old_z.rotation_angle)
            effective_rot_angle = rotation_angle - original_rotation_angle
            old_z.rotate(effective_rot_angle)
        
        # rotate tipper to profile azimuth, not strike. Need angle to be between

        rotation_angle = (self.azimuth - 90) % 180
            
        
        for old_tipper in self.Tipper:
            try:
                original_rotation_angle = np.array(old_tipper.rotation_angle)
                effective_rot_angle = rotation_angle - original_rotation_angle
                old_tipper.rotate(effective_rot_angle)
            except:
                pass


        projected_stations = []
        lo_offsets = []
        profile_vector = np.array([1,profile_line[0]])
        profile_vector /= np.linalg.norm(profile_vector)

        for idx,sta in enumerate(self.stations):
            station_vector = np.array([lo_easts[idx],lo_norths[idx]-profile_line[1]])
            position = np.dot(profile_vector,station_vector) * profile_vector 
            lo_offsets.append(np.linalg.norm(position))
            projected_stations.append([position[0],position[1]+profile_line[1]])

        lo_offsets -= min(lo_offsets)


        #Sort from West to East:
        profile_idxs = np.argsort(lo_offsets)
        if self.azimuth == 0:
            #Exception: sort from North to South
            profile_idxs = np.argsort(lo_norths)


        #sorting along the profile
        projected_stations = [projected_stations[i] for i in profile_idxs]
        projected_stations =  np.array(projected_stations)
        lo_offsets = np.array([lo_offsets[i] for i in profile_idxs])
        lo_offsets -= min(lo_offsets)

        self.station_coords = [self.station_coords[i] for i in profile_idxs]
        self.stations = [self.stations[i] for i in profile_idxs]
        self.station_frequencies = [self.station_frequencies[i] for i in profile_idxs]
        self.Z = [self.Z[i] for i in profile_idxs]
        self.Tipper = [self.Tipper[i] for i in profile_idxs]
        lo_easts = np.array([lo_easts[i] for i in profile_idxs])
        lo_norths = np.array([lo_norths[i] for i in profile_idxs])
              

        self.profile = profile_line
        self.stationlocations = lo_offsets
        self.easts = lo_easts
        self.norths = lo_norths
        #print self.stationlocations

        #plot profile and stations:
        if 0:
            lo_all_easts = list(lo_easts)
            lo_all_easts.extend(list(projected_stations[:,0]))
            lo_all_norths = list(lo_norths)
            lo_all_norths.extend(list(projected_stations[:,1]))
            x_extent = max(lo_all_easts) - min(lo_all_easts)
            y_extent = max(lo_all_norths) - min(lo_all_norths)
            plt.close('all')
            lfig = plt.figure(4, dpi=200)#, figsize=(2,2))
            plt.clf()
            ploty = sp.polyval(profile_line, sorted(lo_all_easts))
            lax = lfig.add_subplot(1, 1, 1,aspect='equal')
            lax.plot(sorted(lo_all_easts), ploty, '-k', lw=1)
            lax.scatter(lo_easts,lo_norths,color='b',marker='+')
            lax.scatter(projected_stations[:,0], projected_stations[:,1],color='r',marker='x')
            lax.set_title('Original/Projected Stations')
            lax.set_ylim(np.min([lo_norths.min(),projected_stations[:,1].min()])-0.2*y_extent, 
                                            np.max([lo_norths.max(),projected_stations[:,1].max()])+0.2*y_extent)
            lax.set_xlim(np.min([lo_easts.min(),projected_stations[:,0].min()])-0.2*x_extent, 
                                            np.max([lo_easts.max(),projected_stations[:,0].max()])+0.2*x_extent)
            lax.set_xlabel('Easting (m)', 
                           fontdict={'size':4, 'weight':'bold'})
            lax.set_ylabel('Northing (m)',
                           fontdict={'size':4, 'weight':'bold'})
            plt.show()
            #raw_input()




    def writefile(self, filename = None):

        if filename is not None:
            try:
                fn = op.abspath(op.join(self.wd,filename))
                self.datafile = op.split(fn)[1]
                #self.wd = op.abspath(op.split(fn)[0])
            except:
                self.datafile = op.abspath(op.join(self.wd,'OccamDataFile.dat')) 

        outstring = ''

        outstring += 'FORMAT:'+11*' '+self.format+'\n'
        outstring += 'TITLE:'+12*' '+'{0} - profile azimuth {1:.1f} deg -'\
                    ' strike {2:.1f} deg\n'.format(self.title,self.azimuth,
                                                                 self.strike)
        outstring += 'SITES:'+12*' '+'{0}\n'.format(len(self.stations))
        for s in self.stations:
            outstring += '    {0}\n'.format(s)
        outstring += 'OFFSETS (M):\n'
        for l in self.stationlocations:
            outstring += '    {0}\n'.format(l + self.profile_offset)
        outstring += 'FREQUENCIES:      {0}\n'.format(len(self.frequencies))
        for f in self.frequencies:
            outstring += '    {0}\n'.format(f)
        outstring += 'DATA BLOCKS:      {0}\n'.format(len(self.data))

        outstring += 'SITE    FREQ    TYPE    DATUM    ERROR\n'
        for d in self.data:
            outstring += '{0}    {1}    {2}    {3}    {4}\n'.format(*d)

        outfn = op.abspath(op.join(self.wd,self.datafile))

        F = open(outfn,'w')
        F.write(outstring)
        F.close()

    def get_profile_origin(self):
        """
        get the origin of the profile in real world coordinates
        
        Author: Alison Kirkby (2013)
        """

        x,y = self.easts,self.norths
        x1,y1 = x[0],y[0]
        [m,c1] = self.profile
        x0 = (y1+(1.0/m)*x1-c1)/(m+(1.0/m))
        y0 = m*x0+c1 
        self.profile_origin = [x0,y0]
    


class Model():
    """
    Handling of Occam output files.

    Reading, writing, renaming,... of 'ITER' and 'RESP' files. 
    """


class Plot():
    """
    Graphical representations of in- and output data.

    Provide gui for masking points.
    Represent output models.
    """

class Run():
    """
    Run Occam2D by system call.

    Future plan: implement Occam in Python and call it from here directly.
    """


class Mask(Data):
    """
    Allow masking of points from data file (effectively commenting them out, 
    so the process is reversable). Inheriting from Data class.
    """

class OccamInputError(Exception):
    pass

