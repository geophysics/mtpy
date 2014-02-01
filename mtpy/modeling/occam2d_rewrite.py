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

class Mesh():
    """
    deals only with the finite element mesh.  Builds a finite element mesh 
    based on given parameters
    
    
    """

    def __init__(self, station_locations=None, **kwargs):
        
        self.station_locations = station_locations
        self.rel_station_locations = None
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
        
        self.rel_station_locations = np.copy(self.station_locations)
        
        #center the stations around 0 like the mesh will be
        self.rel_station_locations -= self.rel_station_locations.mean()
        
        #1) make horizontal nodes at station locations and fill in the cells 
        #   around that area with cell width. This will put the station 
        #   in the center of the regularization block as prescribed for occam
        # the first cell of the station area will be outside of the furthest
        # right hand station to reduce the effect of a large neighboring cell.
        self.x_grid = np.array([self.rel_station_locations[0]-self.cell_width])
                
        for ii, offset in enumerate(self.rel_station_locations[:-1]):
            dx = self.rel_station_locations[ii+1]-offset
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
                    if abs(self.rel_station_locations[ii+1]-new_cell) >= cell_width*.9:
                        self.x_grid = np.append(self.x_grid, new_cell)
                    else:
                        pass
                except IndexError:
                    pass
        
        self.x_grid = np.append(self.x_grid, self.rel_station_locations[-1])        
        # add a cell on the right hand side of the station area to reduce 
        # effect of a large cell next to it       
        self.x_grid = np.append(self.x_grid, 
                                self.rel_station_locations[-1]+self.cell_width)
                            
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
            self.x_nodes[ii] = abs(self.x_grid[ii+1]-xx)
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
                                    
        self.mesh_values = np.zeros((self.x_nodes.shape[0],
                                     self.z_nodes.shape[0], 4), dtype=str)
        self.mesh_values[:,:,:] = '?'
        
        #get elevation if elevation_profile is given
        if self.elevation_profile is not None:
            self.add_elevation(self.elevation_profile)
        
        print '='*55
        print '{0:^55}'.format('mesh parameters'.upper())                            
        print '='*55
        print '  number of horizontal nodes = {0}'.format(self.x_nodes.shape[0])  
        print '  number of vertical nodes   = {0}'.format(self.z_nodes.shape[0])  
        print '  Total Horizontal Distance  = {0:2f}'.format(self.x_nodes.sum())
        print '  Total Vertical Distance    = {0:2f}'.format(self.z_nodes.sum())
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
            #index of ground in the model for that x location
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
                    if elev_right > yright*.95:
                        
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
            df = 1000.
        elif depth_scale == 'm':
            df = 1.
        else:
            df = 1000.
        
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
        for offset in self.rel_station_locations:
            ax.text((offset)/df,
                    0,
                    marker,
                    horizontalalignment='center',
                    verticalalignment='baseline',
                    fontdict={'size':ms,'color':mc})
                    

        #--> make list of column lines        
        row_line_xlist = []
        row_line_ylist = []
        for xx in self.x_grid/df:
            row_line_xlist.extend([xx,xx])
            row_line_xlist.append(None)
            row_line_ylist.extend([0, self.z_grid[-1]/df])
            row_line_ylist.append(None)
        
        #plot column lines (variables are a little bit of a misnomer)
        ax.plot(row_line_xlist, 
                row_line_ylist, 
                color='k', 
                lw=lw)

        #--> make list of row lines
        col_line_xlist = [self.x_grid[0]/df, self.x_grid[-1]/df]
        col_line_ylist = [0, 0]            
        for yy in self.z_grid/df:
            col_line_xlist.extend([self.x_grid[0]/df, 
                                  self.x_grid[-1]/df])
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
            for xx in self.x_grid/df:
                row_line_xlist.extend([xx,xx])
                row_line_xlist.append(None)
                row_line_ylist.extend([0, self.z_grid[-1]/df])
                row_line_ylist.append(None)
                
            #plot columns
            ax.plot(row_line_xlist, 
                    row_line_ylist, 
                    color='k', 
                    lw=lw)
    
            col_line_xlist = []
            col_line_ylist = []            
            for yy in self.z_grid/df:
                col_line_xlist.extend([self.x_grid[0]/df, 
                                      self.x_grid[-1]/df])
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
            for xi, xx in enumerate(self.x_grid[:-1]/df):
                for yi, yy in enumerate(self.z_grid[:-1]/df):
                    diag_line_xlist.extend([xx, self.x_grid[xi+1]/df])
                    diag_line_xlist.append(None)
                    diag_line_xlist.extend([xx, self.x_grid[xi+1]/df])
                    diag_line_xlist.append(None)
                    
                    diag_line_ylist.extend([yy, self.z_grid[yi+1]/df])
                    diag_line_ylist.append(None)
                    diag_line_ylist.extend([self.z_grid[yi+1]/df, yy])
                    diag_line_ylist.append(None)
            
            #plot diagonal lines.
            ax.plot(diag_line_xlist, 
                    diag_line_ylist,
                    color='k',
                    lw=lw)
                    
        #--> set axes properties
        ax.set_ylim(self.z_target_depth/df, -2000/df)
        xpad = self.num_x_pad_cells-1
        ax.set_xlim(self.x_grid[xpad]/df, -self.x_grid[xpad]/df)
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
        
        if self.x_nodes is None:
            self.build_mesh()

        mesh_lines = []
        nx = self.x_nodes.shape[0]
        nz = self.z_nodes.shape[0]
        mesh_lines.append('MESH FILE Created by mtpy.modeling.occam2d\n')
        mesh_lines.append("   {0}  {1}  {2}  {0}  {0}  {3}\n".format(0, nx ,
                          nz, 2))
        
        #--> write horizontal nodes
        node_str = ''
        for ii, xnode in enumerate(self.x_nodes):
            node_str += '{0:>9.1f} '.format(xnode)
            if np.remainder(ii+1, 8) == 0:
                node_str += '\n'
                mesh_lines.append(node_str)
                node_str = ''
                
        node_str += '\n'
        mesh_lines.append(node_str)
                
        #--> write vertical nodes
        node_str = ''
        for ii, znode in enumerate(self.z_nodes):
            node_str += '{0:>9.1f} '.format(znode)
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
        h_index = 0
        v_index = 0
        m_index = 0
        line_count = 2
        
        #--> fill horizontal nodes
        for mline in mlines[line_count:]:
            mline = mline.strip().split()
            for m_value in mline:
                self.x_nodes[h_index] = float(m_value)
                h_index += 1
                
            line_count += 1
            if h_index == nh:
                break

        print line_count 
        #--> fill vertical nodes
        for mline in mlines[line_count:]:
            mline = mline.strip().split()
            for m_value in mline:
                self.z_nodes[v_index] = float(m_value)
                v_index += 1
            line_count += 1
            if v_index == nv:
                break    

        #--> fill model values
        for ll, mline in enumerate(mlines[line_count+1:], line_count):
            if m_index == nv or mline.lower().find('exception')>0:
                break
            else:
                mlist = list(mline)
                if len(mlist) != nh:
                    print '--- Line {0} in {1}'.format(ll, self.mesh_fn) 
                    print 'Check mesh file too many columns'
                    print 'Should be {0}, has {1}'.format(nh,len(mlist))
                    mlist = mlist[0:nh]
                for kk in range(4):        
                    for jj, mvalue in enumerate(list(mlist)):
                        self.mesh_values[jj,m_index,kk] = mline[jj]
                m_index += 1

        #sometimes it seems that the number of nodes is not the same as the
        #header would suggest so need to remove the zeros
        self.x_nodes = self.x_nodes[np.nonzero(self.x_nodes)]
        if self.x_nodes.shape[0] != nh:
            new_nh = self.x_nodes.shape[0]
            print 'The header number {0} should read {1}'.format(nh, new_nh)
            self.mesh_values.resize(new_nh, nv, 4)
        else:
            new_nh = nh
            
        self.z_nodes = self.z_nodes[np.nonzero(self.z_nodes)]
        if self.z_nodes.shape[0] != nv:
            new_nv = self.z_nodes.shape[0]
            print 'The header number {0} should read {1}'.format(nv, new_nv)
            self.mesh_values.resize(new_nh, nv, 4)                                            

         
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
        self.geoelectric_strike = kwargs.pop('geoelectric_strike', None)
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
                    dim = MTgy.dimensionality(z_object=edi.Z)
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
                edi.Tipper.rotate((self.profile_angle-90)%180-
                                                    edi.Tipper.rotation_angle)
           
            print '='*72
            print ('Rotated Z and Tipper to align with '
                   '{0:+.2f} degrees E of N'.format(self.geoelectric_strike)) 
            print ('Profile angle is '
                   '{0:+.2f} degrees E of N'.format(self.profile_angle))        
            print '='*72
        else:
            for edi in self.edi_list:
                edi.Z.rotate((self.profile_angle-90)%180-edi.Z.rotation_angle)
                # rotate tipper to profile azimuth, not strike.
                edi.Tipper.rotate((self.profile_angle-90)%180-
                                   edi.Tipper.rotation_angle)
           
            print '='*72
            print ('Rotated Z and Tipper to be perpendicular  with '
                   '{0:+.2f} profile angle'.format((self.profile_angle-90)%180)) 
            print ('Profile angle is'
                   '{0:+.2f} degrees E of N'.format(self.profile_angle))        
            print '='*72
        
        #--> project stations onto profile line
        projected_stations = np.zeros((self.num_edi, 2))
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
        self.model_name = 'model made by mtpy.modeling.occam2d'
        self.description = 'simple Inversion'
        self.num_param = None
        self.num_free_param = None
        self.statics_fn = kwargs.pop('statics_fn', 'none')
        self.prejudice_fn = kwargs.pop('prejudice_fn', 'none')
        self.num_layers = kwargs.pop('num_layers', None)
        
        
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
        
        model_thickness = np.append(self.z_nodes[0:self.z_nodes.shape[0]-
                                                        self.num_z_pad_cells], 
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
        #model block to the furthest left station which is half the distance
        # from the center of the mesh grid.
        self.binding_offset = self.x_grid[self.num_x_pad_cells+1]+\
                                            self.station_locations.mean()
        
        self.get_num_free_params()
        
        print '='*55
        print '{0:^55}'.format('regularization parameters'.upper())
        print '='*55
        print '   binding offset       = {0:.1f}'.format(self.binding_offset)
        print '   number layers        = {0}'.format(len(self.model_columns))
        print '   number of parameters = {0}'.format(self.num_param)
        print '   number of free param = {0}'.format(self.num_free_param)
        print '='*55

    def get_num_free_params(self):
        """
        estimate the number of free parameters in model mesh.
        
        I'm assuming that if there are any fixed parameters in the block, then
        that model block is assumed to be fixed. Not sure if this is write
        cause there is no documentation.
        """
        
        self.num_free_param = 0

        row_count = 0
        #loop over columns and rows of regularization grid
        for col, row in zip(self.model_columns, self.model_rows):
            rr = row[0]
            col_count = 0
            for ii, cc in enumerate(col):
                #make a model block from the index values of the regularization
                #grid
                model_block = self.mesh_values[row_count:row_count+rr, 
                                              col_count:col_count+cc, :]
                     
                #find all the free triangular blocks within that model block                         
                find_free = np.where(model_block=='?')
                try:
                    #test to see if the number of free parameters is equal 
                    #to the number of triangular elements with in the model 
                    #block, if there is the model block is assumed to be free.
                    if find_free[0].size == model_block.size:
                        self.num_free_param  += 1
                except IndexError:
                    pass
                col_count += cc
            row_count += rr 
        
    def write_regularization_file(self, reg_fn=None, reg_basename=None, 
                                  statics_fn='none', prejudice_fn='none',
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
        
        self.statics_fn = statics_fn
        self.prejudice_fn = prejudice_fn
        
        if self.model_columns is None:
            if self.binding_offset is None:
                self.build_mesh()
            self.build_regularization()
        
        reg_lines = []
        
        #--> write out header information
        reg_lines.append('{0:<18}{1}\n'.format('Format:', 
                                               'occam2mtmod_1.0'.upper()))
        reg_lines.append('{0:<18}{1}\n'.format('Model Name:', 
                                               self.model_name.upper()))
        reg_lines.append('{0:<18}{1}\n'.format('Description:', 
                                               self.description.upper()))
        if os.path.dirname(self.mesh_fn) == self.save_path:
            reg_lines.append('{0:<18}{1}\n'.format('Mesh File:', 
                                                   os.path.basename(self.mesh_fn)))
        else:
            reg_lines.append('{0:<18}{1}\n'.format('Mesh File:',self.mesh_fn))
        reg_lines.append('{0:<18}{1}\n'.format('Mesh Type:', 
                                               'pw2d'.upper()))
        if os.path.dirname(self.statics_fn) == self.save_path:
            reg_lines.append('{0:<18}{1}\n'.format('Statics File:', 
                             os.path.basename(self.statics_fn)))
        else:
            reg_lines.append('{0:<18}{1}\n'.format('Statics File:', 
                                                   self.statics_fn))
        if os.path.dirname(self.prejudice_fn) == self.save_path:
            reg_lines.append('{0:<18}{1}\n'.format('Prejudice File:', 
                             os.path.basename(self.prejudice_fn)))
        else:
            reg_lines.append('{0:<18}{1}\n'.format('Prejudice File:', 
                                                   self.prejudice_fn))
        reg_lines.append('{0:<20}{1: .1f}\n'.format('Binding Offset:', 
                                                   self.binding_offset))
        reg_lines.append('{0:<20}{1}\n'.format('Num Layers:', 
                                               len(self.model_columns)))
        
        #--> write rows and columns of regularization grid                                        
        for row, col in zip(self.model_rows, self.model_columns):
            reg_lines.append(''.join([' {0:>5}'.format(rr) for rr in row])+'\n')
            reg_lines.append(''.join(['{0:>5}'.format(cc) for cc in col])+'\n')
        
        reg_lines.append('{0:<18}{1}\n'.format('NO. EXCEPTIONS:', '0'))                           
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

        self.model_rows = []
        self.model_columns = []    
        ncols = []
        
        rlines = rfid.readlines()
        
        for ii, iline in enumerate(rlines):
            #read header information
            if iline.find(':') > 0:
                iline = iline.strip().split(':')
                key = iline[0].strip().lower()
                key = key.replace(' ', '_').replace('file', 'fn')
                value = iline[1].strip()
                try:
                    setattr(self, key, float(value))
                except ValueError:
                    setattr(self, key, value)
                
                #append the last line
                if key.find('exception') > 0:
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
        if not os.path.isfile(self.mesh_fn):
            self.mesh_fn = os.path.join(self.save_path, self.mesh_fn)
        
        #set statics file name
        if not os.path.isfile(self.mesh_fn):
            self.statics_fn = os.path.join(self.save_path, self.statics_fn)
            
        #set prejudice file name
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
        self.format = kwargs.pop('format', 'OCCAMITER_FLEX')
        self.date_time = kwargs.pop('date_time', time.ctime())
        self.description = kwargs.pop('description', 'startup created by mtpy')
        self.iterations_to_run = kwargs.pop('iterations_to_run', 20)
        self.roughness_type = kwargs.pop('roughness_type', 1)
        self.target_misfit = kwargs.pop('target_misfit', 1.0)
        self.diagonal_penalties = kwargs.pop('diagonal_penalties', 0)
        self.stepsize_count = kwargs.pop('stepsize_count', 8)
        self.model_limits = kwargs.pop('model_limits', None)
        self.model_value_steps = kwargs.pop('model_value_steps', None)
        self.debug_level = kwargs.pop('debug_level', 1)
        self.iteration = kwargs.pop('iteration', 0)
        self.lagrange_value = kwargs.pop('lagrange_value', 5.0)
        self.roughness_value = kwargs.pop('roughness_value', 1e10)
        self.misfit_value = kwargs.pop('misfit_value', 1000)
        self.misfit_reached = kwargs.pop('misfit_reached', 0)
        self.param_count = kwargs.pop('param_count', None)
        self.resistivity_start = kwargs.pop('resistivity_start', 2)
        self.model_values = kwargs.pop('model_values', None)
        
    def write_startup_file(self, startup_fn=None, save_path=None, 
                           startup_basename=None):
        """
        write a startup file based on the parameters of startup class
        
        """
        if save_path is not None:
            self.save_path = save_path
            
        if self.save_path is None:
            self.save_path = os.path.dirname(self.data_fn)
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
        slines.append('{0:<20}{1}\n'.format('Format:',self.format))
        slines.append('{0:<20}{1}\n'.format('Description:', self.description))
        if os.path.dirname(self.model_fn) == self.save_path:
            slines.append('{0:<20}{1}\n'.format('Model File:', 
                          os.path.basename(self.model_fn)))
        else:
            slines.append('{0:<20}{1}\n'.format('Model File:', self.model_fn))
        if os.path.dirname(self.data_fn) == self.save_path:
            slines.append('{0:<20}{1}\n'.format('Data File:', 
                          os.path.basename(self.data_fn)))
        else:
            slines.append('{0:<20}{1}\n'.format('Data File:', self.data_fn))
        slines.append('{0:<20}{1}\n'.format('Date/Time:', self.date_time))
        slines.append('{0:<20}{1}\n'.format('Iterations to run:',
                                            self.iterations_to_run))
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
        slines.append('{0:<20}{1}\n'.format('Iteration:', self.iteration))
        slines.append('{0:<20}{1}\n'.format('Lagrange Value:', 
                                            self.lagrange_value))
        slines.append('{0:<20}{1}\n'.format('Roughness Value:', 
                                            self.roughness_value))
        slines.append('{0:<20}{1}\n'.format('Misfit Value:', self.misfit_value))
        slines.append('{0:<20}{1}\n'.format('Misfit Reached:', 
                                            self.misfit_reached))
        slines.append('{0:<20}{1}\n'.format('Param Count:', self.param_count))
        
        #make an array of starting values if not are given
        if self.model_values is None:
            self.model_values = np.zeros(self.param_count)
            self.model_values[:] = self.resistivity_start
        
        if self.model_values.shape[0] != self.param_count:
            raise OccamInputError('length of model vaues array is not equal '
                                  'to param count {0} != {1}'.format(
                                  self.model_values.shape[0], self.param_count))
        
        #write out starting resistivity values    
        sline = []    
        for ii, mv in enumerate(self.model_values):
            sline.append('{0:^10.4f}'.format(mv))
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
    
    **freq_min** : float (Hz)
                   minimum frequency to invert for.
                   *default* is None and will use the data to find min
    
    **freq_max** : float (Hz)
                   maximum frequency to invert for
                   *default* is None and will use the data to find max
                   
    **freq_num** : int
                   number of frequencies to inver for
                   *default* is None and will use the data to find num
                   
    **freq_tol** : float (decimal percent)
                   tolerance to find nearby frequencies.
                   *default* is .05 --> 5 percent.
                   
    **mode_num** : float (decimal percent)
                   percent of stations to have the same frequency to 
                   add it to the frequency list.  *default* is .5 -->
                   50 percent.  Meaning if 50% of stations have the 
                   same frequency within freq_tol it will be its 
                   individual frequency in frequency list.    
    
    ===================== =====================================================
    Model Modes           Description                     
    ===================== =====================================================
    1 or log_all          Log resistivity of TE and TM plus Tipper
    2 or log_te_tip       Log resistivity of TE plus Tipper
    3 or log_tm_tip       Log resistivity of TM plus Tipper
    4 or log_te_tm        Log resistivity of TE and TM
    5 or log_te           Log resistivity of TE
    6 or log_tm           Log resistivity of TM
    7 or all              TE, TM and Tipper
    8 or te_tip           TE plus Tipper
    9 or tm_tip           TM plus Tipper
    10 or te_tm           TE and TM mode
    11 or te              TE mode
    12 or tm              TM mode
    13 or tip             Only Tipper
    ===================== =====================================================

    """
    def __init__(self, edi_path=None, **kwargs):
        Profile.__init__(self, edi_path, **kwargs)
        
        self.data_fn = kwargs.pop('data_fn', None)
        self.fn_basename = kwargs.pop('fn_basename', 'OccamDataFile.dat')
        self.save_path = kwargs.pop('save_path', None)
        self.freq = kwargs.pop('freq', None)
        self.model_mode = kwargs.pop('model_mode', '1')
        self.data = kwargs.pop('data', None)
        self.data_list = None
        
        self.res_te_err = kwargs.pop('res_te_err', 10)
        self.res_tm_err = kwargs.pop('res_tm_err', 10)
        self.phase_te_err = kwargs.pop('phase_te_err', 5)
        self.phase_tm_err = kwargs.pop('phase_tm_err', 5)
        self.tipper_err = kwargs.pop('tipper_err', 10)
        
        self.freq_min = kwargs.pop('freq_min', None)
        self.freq_max = kwargs.pop('freq_max', None)
        self.freq_num = kwargs.pop('freq_num', None)
        self.freq_tol = kwargs.pop('freq_tol', 0.05)
        self.freq_mode_num = kwargs.pop('freq_mode_num', .50)

        self.occam_format = 'OCCAM2MTDATA_1.0'
        self.title = 'MTpy-OccamDatafile'
        self.edi_type = 'z' 
        
        self.occam_dict = {'1':'log_te_res',
                           '2':'te_phase',
                           '3':'re_tip',
                           '4':'im_tip',
                           '5':'log_tm_res',
                           '6':'tm_phase',
                           '9':'te_res',
                           '10':'tm_res'}
                           
        self.mode_dict = {'log_all':[1, 2, 3, 4, 5, 6],
                          'log_te_tip':[1, 2, 3, 4],
                          'log_tm_tip':[5, 6, 3, 4],
                          'log_te_tm':[1, 2, 5, 6],
                          'log_te':[1, 2],
                          'log_tm':[5, 6],
                          'all':[9, 2, 3, 4, 10, 6],
                          'te_tip':[9, 2, 3, 4],
                          'tm_tip':[10, 6, 3, 4], 
                          'te_tm':[9, 2, 10, 6],                         
                          'te':[9, 2],
                          'tm':[10, 6],
                          'tip':[3, 4],
                          '1':[1, 2, 3, 4, 5, 6],
                          '2':[1, 2, 3, 4],
                          '3':[5, 6, 3, 4],
                          '4':[1, 2, 5, 6],
                          '5':[1, 2],
                          '6':[5, 6],
                          '7':[9, 2, 3, 4, 10, 6],
                          '8':[9, 2, 3, 4],
                          '9':[10, 6, 3, 4], 
                          '10':[9, 2, 10, 6],                         
                          '11':[9, 2],
                          '12':[10, 6],
                          '13':[3, 4]}
                          
        self._data_string = '{0:^6}{1:^6}{2:^6} {3: >8} {4: >8}\n'
        self._data_header = '{0:<6}{1:<6}{2:<6} {3:<8} {4:<8}\n'.format(
                            'SITE', 'FREQ', 'TYPE', 'DATUM', 'ERROR')


    def read_data_file(self, data_fn=None):
        """
        
        Returns:
        --------
            **data** : list of dictionaries for each station 
                                         with keywords:
                
                *'station'* : string
                              station name
                
                *'offset'* : float
                            relative offset
                
                *'te_res'* : np.array(nf,2)
                          TE resistivity and error as row 0 and 1 ressectively
                
                *'tm_res'* : np.array(fn,2)
                          TM resistivity and error as row 0 and 1 respectively
                
                *'te_phase'* : np.array(nf,2)
                            TE phase and error as row 0 and 1 respectively
                
                *'tm_phase'* : np.array(nf,2)
                            Tm phase and error as row 0 and 1 respectively
                
                *'re_tip'* : np.array(nf,2)
                            Real Tipper and error as row 0 and 1 respectively
                
                *'im_tip'* : np.array(nf,2)
                            Imaginary Tipper and error as row 0 and 1 
                            respectively
                
            .. note:: resistivity is converted to linear if input as log
                      and data error are converted as dx*ln(10)
                
        :Example: ::
            
            >>> import mtpy.modeling.occam2d as occam2d
            >>> ocd = occam2d.Data()
            >>> ocd.read_data_file(r"/home/Occam2D/Line1/Inv1/Data.dat")
            
        """
        
        if data_fn is not None:
            self.data_fn = data_fn
        
        if os.path.isfile(self.data_fn) == False:
            raise OccamInputError('Could not find {0}'.format(self.data_fn))
        if self.data_fn is None:
            raise OccamInputError('data_fn is None, input filename')
            
        self.save_path = op.dirname(self.data_fn)
        
        dfid = open(self.data_fn,'r')
        
        dlines = dfid.readlines()
        
        #get format of input data
        self.occam_format = dlines[0].strip().split(':')[1].strip()
        
        #get title
        self.title = dlines[1].strip().split(':')[1].strip()
    
        if self.title.find('=') > 0:
            tstr = self.title.split('=')
            self.title = tstr[0]
        else:
            self.title = self.title
            
        #get number of sits
        nsites = int(dlines[2].strip().split(':')[1].strip())
        
        #get station names
        self.station_list = np.array([dlines[ii].strip() 
                                      for ii in range(3, nsites+3)])
        
        #get offsets in meters
        self.station_locations = np.array([float(dlines[ii].strip()) 
                                        for ii in range(4+nsites, 4+2*nsites)])
        
        #get number of frequencies
        nfreq = int(dlines[4+2*nsites].strip().split(':')[1].strip())
    
        #get frequencies
        self.freq = np.array([float(dlines[ii].strip()) 
                                for ii in range(5+2*nsites,5+2*nsites+nfreq)])
        
        #get periods
        self.period = 1./self.freq

        #-----------get data-------------------
        #set zero array size the first row will be the data and second the error
        asize = (2, self.freq.shape[0])
        
        #make a list of dictionaries for each station.
        self.data = [{'station':station,
                      'offset':offset,
                      'te_phase':np.zeros(asize),
                      'tm_phase':np.zeros(asize),
                      're_tip':np.zeros(asize),
                      'im_tip':np.zeros(asize),
                      'te_res':np.zeros(asize),
                      'tm_res':np.zeros(asize)}
                       for station, offset in zip(self.station_list, 
                                                  self.station_locations)]
                                                
        self.data_list = dlines[7+2*nsites+nfreq:]
        for line in self.data_list:
            try:
                station, freq, comp, odata, oerr = line.split()
                #station index -1 cause python starts at 0
                ss = int(station)-1
    
                #frequency index -1 cause python starts at 0       
                ff = int(freq)-1
                #data key
                key = self.occam_dict[comp]
                
                #put into array
                if int(comp) == 1 or int(comp) == 5:
                    self.data[ss][key[4:]][0, ff] = 10**float(odata) 
                    #error       
                    self.data[ss][key[4:]][1, ff] = float(odata)*np.log(10)
                else:
                    self.data[ss][key][0, ff] = float(oerr) 
                    #error       
                    self.data[ss][key][1, ff] = float(oerr)
            except ValueError:
                print 'Could not read line {0}'.format(line)
                
    def _get_frequencies(self):
        """
        from the list of edi's get a frequency list to invert for.
        
        Arguments:
        ------------
            **freq_min** : float (Hz)
                           minimum frequency to invert for.
                           *default* is None and will use the data to find min
            
            **freq_max** : float (Hz)
                           maximum frequency to invert for
                           *default* is None and will use the data to find max
                           
            **freq_num** : int
                           number of frequencies to inver for
                           *default* is None and will use the data to find num
                           
            **freq_tol** : float (decimal percent)
                           tolerance to find nearby frequencies.
                           *default* is .05 --> 5 percent.
                           
            **mode_num** : float (decimal percent)
                           percent of stations to have the same frequency to 
                           add it to the frequency list.  *default* is .5 -->
                           50 percent.  Meaning if 50% of stations have the 
                           same frequency within freq_tol it will be its 
                           individual frequency in frequency list.
        """

        #get all frequencies from all edi files
        lo_all_freqs = []
        for edi in self.edi_list:
            lo_all_freqs.extend(list(edi.Z.freq))
        
        #sort all frequencies so that they are in descending order,
        #use set to remove repeats and make an array
        all_freqs = np.array(sorted(list(set(lo_all_freqs)), reverse=True))

        #--> get min and max values if none are given
        if (self.freq_min is None) or (self.freq_min < all_freqs.min()) or\
           (self.freq_min > all_freqs.max()):
            self.freq_min = all_freqs.min()
            
        if (self.freq_max is None) or (self.freq_max > all_freqs.max()) or\
           (self.freq_max < all_freqs.max()):
            self.freq_max = all_freqs.max()
        
        #--> get all frequencies within the given range
        self.freq = all_freqs[np.where((all_freqs >= self.freq_min) & 
                                       (all_freqs <= self.freq_max))]

        if len(self.freq) == 0:
            raise OccamInputError('No frequencies in user-defined interval '
                           '[{0}, {1}]'.format(self.freq_min, self.freq_max))

        #check, if frequency list is longer than given max value
        if self.freq_num is not None:
            if int(self.freq_num) < self.freq.shape[0]:
                print ('Number of frequencies exceeds freq_num ' 
                        '{0} > {1} '.format(self.freq.shape[0], self.freq_num)+
                        'Trimming frequencies to {0}'.format(self.freq_num))
                        
                excess = self.freq.shape[0]/float(self.freq_num)
                if excess < 2:
                    offset = 0
                else:
                    stepsize = (self.freq.shape[0]-1)/self.freq_num
                    offset = stepsize/2.
                indices = np.array(np.around(np.linspace(offset,
                                   self.freq.shape[0]-1-offset, 
                                   self.freq_num),0), dtype='int')
                if indices[0] > (self.freq.shape[0]-1-indices[-1]):
                    indices -= 1
                self.freq = self.freq[indices]

    def _fill_data(self):
        """
        Read all Edi files. 
        Create a profile
        rotate impedance and tipper
        Extract frequencies. 

        Collect all information sorted according to occam specifications.

        Data of Z given in muV/m/nT = km/s
        Error is assumed to be 1 stddev.
        """ 
        
        #create a profile line, this sorts the stations by offset and rotates
        #data.
        self.generate_profile()
        self.plot_profile()

        #--> get frequencies to invert for
        self._get_frequencies()
        
        #set zero array size the first row will be the data and second the error
        asize = (2, self.freq.shape[0])
        
        #make a list of dictionaries for each station.
        self.data=[{'station':station,
                    'offset':offset,
                    'te_phase':np.zeros(asize),
                    'tm_phase':np.zeros(asize),
                    're_tip':np.zeros(asize),
                    'im_tip':np.zeros(asize),
                    'te_res':np.zeros(asize),
                    'tm_res':np.zeros(asize)}
                     for station, offset in zip(self.station_list, 
                                                self.station_locations)]

        #loop over mt object in edi_list and use a counter starting at 1 
        #because that is what occam starts at.
        for s_index, edi in enumerate(self.edi_list):
            rho = edi.Z.resistivity
            phi = edi.Z.phase
            rho_err = edi.Z.resistivity_err
            station_freqs = edi.Z.freq
            tipper = edi.Tipper.tipper
            tipper_err = edi.Tipper.tippererr
            
            self.data[s_index]['station'] = edi.station
            self.data[s_index]['offset'] = edi.offset

            for freq_num, frequency in enumerate(self.freq):
                #skip, if the listed frequency is not available for the station
                if not (frequency in station_freqs):
                    continue

                #find the respective frequency index for the station     
                f_index = np.abs(station_freqs-frequency).argmin()

                #--> get te resistivity
                self.data[s_index]['te_res'][0, freq_num] = rho[f_index, 0, 1]
                #compute error                
                if rho[f_index, 0, 1] != 0.0:
                    #--> get error from data
                    if self.res_te_err is None:
                        self.data[s_index]['te_res'][1, freq_num] = \
                            np.abs(rho_err[f_index, 0, 1]/rho[f_index, 0, 1])
                    #--> set generic error floor
                    else:
                        self.data[s_index]['te_res'][1, freq_num] = \
                                                        self.res_te_err/100.
                            
                #--> get tm resistivity
                self.data[s_index]['tm_res'][0, freq_num] =  rho[f_index, 1, 0]
                #compute error
                if rho[f_index, 1, 0] != 0.0:
                    #--> get error from data
                    if self.res_tm_err is None:
                        self.data[s_index]['tm_res'][1, freq_num] = \
                        np.abs(rho_err[f_index, 1, 0]/rho[f_index, 1, 0])
                    #--> set generic error floor
                    else:
                        self.data[s_index]['tm_res'][1, freq_num] = \
                            self.res_tm_err/100.
                            
                #--> get te phase
                phase_te = phi[f_index, 0, 1]
                #be sure the phase is in the first quadrant
                if phase_te > 180:
                    phase_te -= 180
                self.data[s_index]['te_phase'][0, freq_num] =  phase_te
                #compute error
                #if phi[f_index, 0, 1] != 0.0:
                #--> get error from data
                if self.phase_te_err is None:
                    self.data[s_index]['te_phase'][1, freq_num] = \
                    np.degrees(np.arcsin(.5*
                               self.data[s_index]['te_res'][0, freq_num]))
                #--> set generic error floor
                else:
                    self.data[s_index]['te_phase'][1, freq_num] = \
                        (self.phase_te_err/100.)*57./2.
                            
                #--> get tm phase
                phase_tm = phi[f_index, 1, 0]
                #be sure the phase is in the first quadrant
                if phase_tm > 180:
                    phase_tm -= 180
                self.data[s_index]['tm_phase'][0, freq_num] =  phase_tm
                #compute error
                #if phi[f_index, 1, 0] != 0.0:
                #--> get error from data
                if self.phase_tm_err is None:
                    self.data[s_index]['tm_phase'][1, freq_num] = \
                    np.degrees(np.arcsin(.5*
                               self.data[s_index]['tm_res'][0, freq_num]))
                #--> set generic error floor
                else:
                    self.data[s_index]['tm_phase'][1, freq_num] = \
                        (self.phase_tm_err/100.)*57./2.
               

                                
                #--> get Tipper
                if tipper is not None:
                    self.data[s_index]['re_tip'][0, freq_num] = \
                                                    tipper[f_index, 0, 1].real
                    self.data[s_index]['im_tip'][0, freq_num] = \
                                                    tipper[f_index, 0, 1].imag
                                                    
                    #get error
                    if self.tipper_err is not None:
                        self.data[s_index]['re_tip'][1, freq_num] = \
                                                        self.tipper_err/100.
                        self.data[s_index]['im_tip'][1, freq_num] = \
                                                        self.tipper_err/100.
                    else:
                        self.data[s_index]['re_tip'][1, freq_num] = \
                           tipper[f_index, 0, 1].real/tipper_err[f_index, 0, 1]
                        self.data[s_index]['im_tip'][1, freq_num] = \
                           tipper[f_index, 0, 1].imag/tipper_err[f_index, 0, 1]
                           
    def _get_data_list(self):
        """
        get a list of data to put into data file
        """

        self.data_list = []
        for ss, sdict in enumerate(self.data, 1):
            for ff in range(self.freq.shape[0]):
                for mmode in self.mode_dict[self.model_mode]:
                    #log(te_res)
                    if mmode == 1:
                        if sdict['te_res'][0, ff] != 0.0:
                            dvalue = np.log10(sdict['te_res'][0, ff])
                            derror = sdict['te_res'][1, ff]/np.log(10)
                            dstr = '{0:.4f}'.format(dvalue)
                            derrstr = '{0:.4f}'.format(derror)
                            line = self._data_string.format(ss, ff+1, mmode, 
                                                            dstr, derrstr)
                            self.data_list.append(line)
                    
                    #te_res
                    if mmode == 9:
                        if sdict['te_res'][0, ff] != 0.0:
                            dvalue = sdict['te_res'][0, ff]
                            derror = sdict['te_res'][1, ff]
                            dstr = '{0:.4f}'.format(dvalue)
                            derrstr = '{0:.4f}'.format(derror)
                            line = self._data_string.format(ss, ff+1, mmode, 
                                                            dstr, derrstr)
                            self.data_list.append(line)
                    
                    #te_phase
                    if mmode == 2:
                        if sdict['te_phase'][0, ff] != 0.0:
                            dvalue = sdict['te_phase'][0, ff]
                            derror = sdict['te_phase'][1, ff]
                            dstr = '{0:.4f}'.format(dvalue)
                            derrstr = '{0:.4f}'.format(derror)
                            line = self._data_string.format(ss, ff+1, mmode, 
                                                            dstr, derrstr)
                            self.data_list.append(line)
                   
                   #log(tm_res)
                    if mmode == 5:
                        if sdict['tm_res'][0, ff] != 0.0:
                            dvalue = np.log10(sdict['tm_res'][0, ff])
                            derror = sdict['tm_res'][1, ff]/np.log(10)
                            dstr = '{0:.4f}'.format(dvalue)
                            derrstr = '{0:.4f}'.format(derror)
                            line = self._data_string.format(ss, ff+1, mmode, 
                                                            dstr, derrstr)
                            self.data_list.append(line)
                    
                    #tm_res
                    if mmode == 10:
                        if sdict['tm_res'][0, ff] != 0.0:
                            dvalue = sdict['tm_res'][0, ff]
                            derror = sdict['tm_res'][1, ff]
                            dstr = '{0:.4f}'.format(dvalue)
                            derrstr = '{0:.4f}'.format(derror)
                            line = self._data_string.format(ss, ff+1, mmode, 
                                                            dstr, derrstr)
                            self.data_list.append(line)
                    
                    #tm_phase
                    if mmode == 6:
                        if sdict['tm_phase'][0, ff] != 0.0:
                            dvalue = sdict['tm_phase'][0, ff]
                            derror = sdict['tm_phase'][1, ff]
                            dstr = '{0:.4f}'.format(dvalue)
                            derrstr = '{0:.4f}'.format(derror)
                            line = self._data_string.format(ss, ff+1, mmode, 
                                                            dstr, derrstr)
                            self.data_list.append(line)
                    
                    #Re_tip
                    if mmode == 3:
                        if sdict['re_tip'][0, ff] != 0.0:
                            dvalue = sdict['re_tip'][0, ff]
                            derror = sdict['re_tip'][1, ff]
                            dstr = '{0:.4f}'.format(dvalue)
                            derrstr = '{0:.4f}'.format(derror)
                            line = self._data_string.format(ss, ff+1, mmode, 
                                                            dstr, derrstr)
                            self.data_list.append(line)
                    
                    #Im_tip
                    if mmode == 4:
                        if sdict['im_tip'][0, ff] != 0.0:
                            dvalue = sdict['im_tip'][0, ff]
                            derror = sdict['im_tip'][1, ff]
                            dstr = '{0:.4f}'.format(dvalue)
                            derrstr = '{0:.4f}'.format(derror)
                            line = self._data_string.format(ss, ff+1, mmode, 
                                                            dstr, derrstr)
                            self.data_list.append(line)
                    
                

    def write_data_file(self, data_fn=None):
        """
        write a data file 
        
        """        
        
        if self.data is None:
            self._fill_data()
            self._get_data_list()

        if self.data_list is None:
            self._get_data_list()
            
        if data_fn is not None:
            self.data_fn = data_fn
        if self.data_fn is None:
            if self.save_path is None:
                self.save_path = os.getcwd()
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)
                
            self.data_fn = os.path.join(self.save_path, self.fn_basename)
            
        data_lines = []
        
        #--> header line
        data_lines.append('{0:<18}{1}\n'.format('FORMAT:', self.occam_format))
        
        #--> title line
        data_lines.append('{0:<18}{1}\n'.format('TITLE:', self.title+
                                                ', Profile angle={0:.1f}'.format(self.profile_angle)))
        
        #--> sites
        data_lines.append('{0:<18}{1}\n'.format('SITES:', len(self.data)))
        for sdict in self.data:
            data_lines.append('   {0}\n'.format(sdict['station']))
        
        #--> offsets
        data_lines.append('{0:<18}\n'.format('OFFSETS (M):'))
        for sdict in self.data:
            data_lines.append('   {0:.1f}\n'.format(sdict['offset']))
        #--> frequencies
        data_lines.append('{0:<18}{1}\n'.format('FREQUENCIES:', 
                                                self.freq.shape[0]))
        for ff in self.freq:
            data_lines.append('   {0:.6f}\n'.format(ff))
            
        #--> data
        data_lines.append('{0:<18}{1}\n'.format('DATA BLOCKS:', 
                                                len(self.data_list)))
        data_lines.append(self._data_header)
        data_lines += self.data_list
        
        dfid = file(self.data_fn, 'w')
        dfid.writelines(data_lines)
        dfid.close()
        
        print 'Wrote Occam2D data file to {0}'.format(self.data_fn)
        
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
    
class Response(object):
    """
    deals with .resp files
    
    """
    
    def __init__(self, resp_fn=None, **kwargs):
        self.resp_fn = resp_fn
        
        self.resp = None
        self.occam_dict = {'1':'log_te_res',
                           '2':'te_phase',
                           '3':'re_tip',
                           '4':'im_tip',
                           '5':'log_tm_res',
                           '6':'tm_phase',
                           '9':'te_res',
                           '10':'tm_res'}
        
    def read_response_file(self, resp_fn=None):
        """
        read in response file and put into a list of dictionaries similar 
        to Data
        """
        
        if resp_fn is not None:
            self.resp_fn = resp_fn
            
        if self.resp_fn is None:
            raise OccamInputError('resp_fn is None, please input response file')
            
        if os.path.isfile(self.resp_fn) == False:
            raise OccamInputError('Could not find {0}'.format(self.resp_fn))
         
        r_arr = np.loadtxt(self.resp_fn, dtype=[('station', np.int), 
                                                 ('freq', np.int),
                                                 ('comp', np.int), 
                                                 ('z', np.int), 
                                                 ('data', np.float),
                                                 ('resp', np.float),
                                                 ('err', np.float)])
                                                 
        num_stat = r_arr['station'].max()
        num_freq = r_arr['freq'].max()
        
        #set zero array size the first row will be the data and second the error
        asize = (2, num_freq)
        
        #make a list of dictionaries for each station.
        self.resp = [{'te_phase':np.zeros(asize),
                      'tm_phase':np.zeros(asize),
                      're_tip':np.zeros(asize),
                      'im_tip':np.zeros(asize),
                      'te_res':np.zeros(asize),
                      'tm_res':np.zeros(asize)}
                       for ss in range(num_stat)]

        for line in r_arr:
            #station index -1 cause python starts at 0
            ss = line['station']-1

            #frequency index -1 cause python starts at 0       
            ff = line['freq']-1
            #data key
            key = self.occam_dict[str(line['comp'])]
            #put into array
            if line['comp'] == 1 or line['comp'] == 5:
                self.resp[ss][key[4:]][0, ff] = 10**line['resp'] 
                #error       
                self.resp[ss][key[4:]][1, ff] = line['err']*np.log(10)
            else:
                self.resp[ss][key][0, ff] = line['resp'] 
                #error       
                self.resp[ss][key][1, ff] = line['err']

class Model(Startup):
    """
    Read .iter file, build model from mesh and regularization grid.
    
    Inheret Startup because they are basically the same object
    
    """
    
    def __init__(self, iter_fn=None, model_fn=None, mesh_fn=None, **kwargs):
        Startup.__init__(self, **kwargs)
        self.iter_fn = iter_fn
        self.model_fn = model_fn
        self.mesh_fn = mesh_fn
        self.data_fn = kwargs.pop('data_fn', None)
        self.model_values = kwargs.pop('model_values', None)
        self.res_model = None
        self.plot_x = None
        self.plot_z = None
        self.mesh_x = None
        self.mesh_z = None
  
     
    def read_iter_file(self, iter_fn=None):
        """
        Read an iteration file.
        
        Arguments:
        ----------
            **iter_fn** : string
                        full path to iteration file if iterpath=None.  If 
                        iterpath is input then iterfn is just the name
                        of the file without the full path.

        Returns:
        --------
        
        :Example: ::
            
            >>> import mtpy.modeling.occam2d as occam2d
            >>> itfn = r"/home/Occam2D/Line1/Inv1/Test_15.iter"
            >>> ocm = occam2d.Model(itfn)
            >>> ocm.read2DIter()
            
        """
    
        if iter_fn is not None:
            self.iter_fn == iter_fn
       
        if self.iter_fn is None:
            raise OccamInputError('iter_fn is None, input iteration file')


        #check to see if the file exists
        if os.path.exists(self.iter_fn) == False:
            raise OccamInputError('Can not find {0}'.format(self.iter_fn))
            
        self.save_path = os.path.dirname(self.iter_fn)
    
        #open file, read lines, close file
        ifid = file(self.iter_fn, 'r')
        ilines = ifid.readlines()
        ifid.close()
        
        ii = 0
        #put header info into dictionary with similar keys
        while ilines[ii].lower().find('param') != 0:
            iline = ilines[ii].strip().split(':')
            key = iline[0].strip().lower()
            if key.find('!') != 0:
                key = key.replace(' ', '_').replace('file', 'fn').replace('/','_')
                value = iline[1].strip()
                try:
                    setattr(self, key, float(value))
                except ValueError:
                    setattr(self, key, value)  
            ii += 1
        
        #get number of parameters
        iline = ilines[ii].strip().split(':')
        key = iline[0].strip().lower().replace(' ', '_')
        value = int(iline[1].strip())
        setattr(self, key, value)
        
        self.model_values = np.zeros(self.param_count)
        kk= int(ii+1)
        
        jj = 0
        mv_index = 0
        while jj < len(ilines)-kk:
            iline = np.array(ilines[jj+kk].strip().split(), dtype='float')
            self.model_values[mv_index:mv_index+iline.shape[0]] = iline
            jj += 1
            mv_index += iline.shape[0]
        
        #make sure data file is full path
        if os.path.isfile(self.data_fn) == False:
            self.data_fn = os.path.join(self.save_path, self.data_fn)
        
        #make sure model file is full path
        if os.path.isfile(self.model_fn) == False:
            self.model_fn = os.path.join(self.save_path, self.model_fn)
            
    def write_iter_file(self, iter_fn=None):
        """
        write an iteration file if you need to for some reason, same as 
        startup file
        """
        if iter_fn is not None:
            self.iter_fn = iter_fn
        
        self.write_startup_file(iter_fn)
        
    def build_model(self):
        """
        build the model from the mesh, regularization grid and model file
        
        """
        
        #first read in the iteration file        
        self.read_iter_file()
        
        #read in the regulariztion file
        r1 = Regularization()
        r1.read_regularization_file(self.model_fn)
        r1.model_rows = np.array(r1.model_rows)

        #read in mesh file
        r1.read_mesh_file(r1.mesh_fn)

        #get the binding offset which is the right side of the furthest left
        #block, this helps locate the model in relative space
        bndgoff = r1.binding_offset
        
        #make sure that the number of rows and number of columns are the same
        assert len(r1.model_rows) == len(r1.model_columns)
        
        #initiate the resistivity model to the shape of the FE mesh
        self.res_model = np.zeros((r1.z_nodes.shape[0], r1.x_nodes.shape[0]))
        
        #read in the model and set the regularization block values to map onto
        #the FE mesh so that the model can be plotted as an image or regular 
        #mesh.
        mm = 0
        for ii in range(len(r1.model_rows)):
            #get the number of layers to combine
            #this index will be the first index in the vertical direction
            ny1 = r1.model_rows[:ii, 0].sum()
            #the second index  in the vertical direction
            ny2 = ny1+r1.model_rows[ii][0]
            #make the list of amalgamated columns an array for ease
            lc = np.array(r1.model_columns[ii])
            #loop over the number of amalgamated blocks
            for jj in range(len(r1.model_columns[ii])):
                #get first in index in the horizontal direction
                nx1 = lc[:jj].sum()
                #get second index in horizontal direction
                nx2 = nx1+lc[jj]
                #put the apporpriate resistivity value into all the amalgamated 
                #model blocks of the regularization grid into the forward model
                #grid
                self.res_model[ny1:ny2, nx1:nx2] = self.model_values[mm]
                mm += 1
        
        #make some arrays for plotting the model
        self.plot_x = np.array([r1.x_nodes[:ii+1].sum() 
                                for ii in range(len(r1.x_nodes))])
        self.plot_z = np.array([r1.z_nodes[:ii+1].sum() 
                                for ii in range(len(r1.z_nodes))])
        
        #center the grid onto the station coordinates
        x0 = bndgoff-self.plot_x[r1.model_columns[0][0]]
        self.plot_x += x0
        
        #flip the arrays around for plotting purposes
        #plotx = plotx[::-1] and make the first layer start at zero
        self.plot_z = self.plot_z[::-1]-self.plot_z[0]
        
        #make a mesh grid to plot in the model coordinates
        self.mesh_x, self.mesh_z = np.meshgrid(self.plot_x, self.plot_z)
        
        #flip the resmodel upside down so that the top is the stations
        self.res_model = np.flipud(self.res_model)
        
#==============================================================================
# plot model 
#==============================================================================
class PlotModel(Model):
    """
    plot the 2D model found by Occam2D.  The model is displayed as a meshgrid
    instead of model bricks.  This speeds things up considerably.  
    
    Inherets the Model class to take advantage of the attributes and methods
    already coded.
    
    Arguments:
    -----------
        **iter_fn** : string
                      full path to iteration file.  From here all the 
                      necessary files can be found assuming they are in the 
                      same directory.  If they are not then need to input
                      manually.
    
    
    ======================= ===============================================
    keywords                description
    ======================= ===============================================
    block_font_size         font size of block number is blocknum == 'on'
    blocknum                [ 'on' | 'off' ] to plot regulariztion block 
                            numbers.
    cb_pad                  padding between axes edge and color bar 
    cb_shrink               percentage to shrink the color bar
    climits                 limits of the color scale for resistivity
                            in log scale (min, max)
    cmap                    name of color map for resistivity values
    femesh                  plot the finite element mesh
    femesh_triangles        plot the finite element mesh with each block
                            divided into four triangles
    fig_aspect              aspect ratio between width and height of 
                            resistivity image. 1 for equal axes
    fig_dpi                 resolution of figure in dots-per-inch
    fig_num                 number of figure instance
    fig_size                size of figure in inches (width, height)
    font_size               size of axes tick labels, axes labels is +2
    grid                    [ 'both' | 'major' |'minor' | None ] string 
                            to tell the program to make a grid on the 
                            specified axes.
    meshnum                 [ 'on' | 'off' ] 'on' will plot finite element
                            mesh numbers
    meshnum_font_size       font size of mesh numbers if meshnum == 'on'
    ms                      size of station marker 
    plot_yn                 [ 'y' | 'n']
                            'y' --> to plot on instantiation
                            'n' --> to not plot on instantiation
    regmesh                 [ 'on' | 'off' ] plot the regularization mesh
                            plots as blue lines
    station_color           color of station marker
    station_font_color      color station label
    station_font_pad        padding between station label and marker
    station_font_rotation   angle of station label in degrees 0 is 
                            horizontal
    station_font_size       font size of station label
    station_font_weight     font weight of station label
    station_id              index to take station label from station name
    station_marker          station marker.  if inputing a LaTex marker
                            be sure to input as r"LaTexMarker" otherwise
                            might not plot properly
    subplot_bottom          subplot spacing from bottom  
    subplot_left            subplot spacing from left  
    subplot_right           subplot spacing from right
    subplot_top             subplot spacing from top
    title                   title of plot.  If None then the name of the
                            iteration file and containing folder will be
                            the title with RMS and Roughness.
    xlimits                 limits of plot in x-direction in (km) 
    xminorticks             increment of minor ticks in x direction
    xpad                    padding in x-direction in km
    ylimits                 depth limits of plot positive down (km)
    yminorticks             increment of minor ticks in y-direction
    ypad                    padding in negative y-direction (km)
    yscale                  [ 'km' | 'm' ] scale of plot, if 'm' everything
                            will be scaled accordingly.
    ======================= ===============================================
    
    =================== =======================================================
    Methods             Description
    =================== =======================================================
    plot                plots resistivity model.  
    redraw_plot         call redraw_plot to redraw the figures, 
                        if one of the attributes has been changed
    save_figure         saves the matplotlib.figure instance to desired 
                        location and format
    =================== ======================================================
    
    :Example: ::
        >>> import mtpy.modeling.occam2d as occam2d
        >>> model_plot = occam2d.PlotModel(r"/home/occam/Inv1/mt_01.iter")
        >>> # change the color limits
        >>> model_plot.climits = (1, 4)
        >>> model_plot.redraw_plot()
        >>> #change len of station name
        >>> model_plot.station_id = [2, 5]
        >>> model_plot.redraw_plot()
        
    
    """

    def __init__(self, iter_fn=None, data_fn=None, **kwargs):
        Model.__init__(self, iter_fn, **kwargs)

        self.yscale = kwargs.pop('yscale', 'km')
        
        self.fig_num = kwargs.pop('fig_num', 1)
        self.fig_size = kwargs.pop('fig_size', [6, 6])
        self.fig_dpi = kwargs.pop('dpi', 300)
        self.fig_aspect = kwargs.pop('fig_aspect', 1)
        self.title = kwargs.pop('title', 'on')
        
        self.xpad = kwargs.pop('xpad', 1.0)
        self.ypad = kwargs.pop('ypad', 1.0)
        
        self.ms = kwargs.pop('ms', 10)
        
        self.station_locations = None
        self.station_list = None
        self.station_id = kwargs.pop('station_id', None)
        self.station_font_size = kwargs.pop('station_font_size', 8)
        self.station_font_pad = kwargs.pop('station_font_pad', 1.0)
        self.station_font_weight = kwargs.pop('station_font_weight', 'bold')
        self.station_font_rotation = kwargs.pop('station_font_rotation', 60)
        self.station_font_color = kwargs.pop('station_font_color', 'k')
        self.station_marker = kwargs.pop('station_marker', 
                                         r"$\blacktriangledown$")
        self.station_color = kwargs.pop('station_color', 'k')
        
        self.ylimits = kwargs.pop('ylimits', None)
        self.xlimits = kwargs.pop('xlimits', None)
        
        self.xminorticks = kwargs.pop('xminorticks', 5)
        self.yminorticks = kwargs.pop('yminorticks', 1)
    
        self.climits = kwargs.pop('climits', (0,4))
        self.cmap = kwargs.pop('cmap', 'jet_r')
        self.font_size = kwargs.pop('font_size', 8)
        
        self.femesh = kwargs.pop('femesh', 'off')
        self.femesh_triangles = kwargs.pop('femesh_triangles', 'off')
        self.femesh_lw = kwargs.pop('femesh_lw', .4)
        self.femesh_color = kwargs.pop('femesh_color', 'k')
        self.meshnum = kwargs.pop('meshnum', 'off')
        self.meshnum_font_size = kwargs.pop('meshnum_font_size', 3)
        
        self.regmesh = kwargs.pop('regmesh', 'off')
        self.regmesh_lw = kwargs.pop('regmesh_lw', .4)
        self.regmesh_color = kwargs.pop('regmesh_color', 'b')
        self.blocknum = kwargs.pop('blocknum', 'off')
        self.block_font_size = kwargs.pop('block_font_size', 3)
        self.grid = kwargs.pop('grid', None)
        
        self.cb_shrink = kwargs.pop('cb_shrink', .8)
        self.cb_pad = kwargs.pop('cb_pad', .01)
        
        self.subplot_right = .99
        self.subplot_left = .085
        self.subplot_top = .92
        self.subplot_bottom = .1
        
        self.plot_yn = kwargs.pop('plot_yn', 'y')
        if self.plot_yn == 'y':
            self.plot()

    def plot(self):
        """
        plotModel will plot the model output by occam2d in the iteration file.
        
        
        :Example: ::
            
            >>> import mtpy.modeling.occam2d as occam2d
            >>> itfn = r"/home/Occam2D/Line1/Inv1/Test_15.iter"
            >>> ocm = occam2d.Occam2DModel(itfn)
            >>> ocm.plot2DModel(ms=20,ylimits=(0,.350),yscale='m',spad=.10,\
                                ypad=.125,xpad=.025,climits=(0,2.5),\
                                aspect='equal')
        """   
        #--> read in iteration file and build the model
        self.read_iter_file()
        self.build_model()

        #--> get station locations and names from data file
        d_object = Data()
        d_object.read_data_file(self.data_fn)
        setattr(self, 'station_locations', d_object.station_locations.copy())
        setattr(self, 'station_list', d_object.station_list.copy())
       
        
        #set the scale of the plot
        if self.yscale == 'km':
            df = 1000.
            pf = 1.0
        elif self.yscale == 'm':
            df = 1.
            pf = 1000.
        else:
            df = 1000.
            pf = 1.0
        
        #set some figure properties to use the maiximum space 
        plt.rcParams['font.size'] = self.font_size
        plt.rcParams['figure.subplot.left'] = self.subplot_left
        plt.rcParams['figure.subplot.right'] = self.subplot_right
        plt.rcParams['figure.subplot.bottom'] = self.subplot_bottom
        plt.rcParams['figure.subplot.top'] = self.subplot_top
        
        #station font dictionary
        fdict = {'size':self.station_font_size,
                 'weight':self.station_font_weight,
                 'rotation':self.station_font_rotation,
                 'color':self.station_font_color}
                 
        #plot the model as a mesh
        self.fig = plt.figure(self.fig_num, self.fig_size, dpi=self.fig_dpi)
        plt.clf()
        
        #add a subplot to the figure with the specified aspect ratio
        ax = self.fig.add_subplot(1, 1, 1, aspect=self.fig_aspect)
        
        #plot the model as a pcolormesh so the extents are constrained to 
        #the model coordinates
        ax.pcolormesh(self.mesh_x/df,
                      self.mesh_z/df,
                      self.res_model,
                      cmap=self.cmap,
                      vmin=self.climits[0],
                      vmax=self.climits[1])
        
        #make a colorbar for the resistivity
        cbx = mcb.make_axes(ax, shrink=self.cb_shrink, pad=self.cb_pad)
        cb = mcb.ColorbarBase(cbx[0],
                              cmap=self.cmap,
                              norm=Normalize(vmin=self.climits[0],
                                             vmax=self.climits[1]))
                                           
        cb.set_label('Resistivity ($\Omega \cdot$m)',
                     fontdict={'size':self.font_size+1,'weight':'bold'})
        cb.set_ticks(np.arange(int(self.climits[0]),int(self.climits[1])+1))
        cb.set_ticklabels(['10$^{0}$'.format('{'+str(nn)+'}') for nn in 
                            np.arange(int(self.climits[0]), 
                                      int(self.climits[1])+1)])
        
        #set the offsets of the stations and plot the stations
        #need to figure out a way to set the marker at the surface in all
        #views.
        for offset, name in zip(self.station_locations, self.station_list):
            #plot the station marker
            #plots a V for the station cause when you use scatter the spacing
            #is variable if you change the limits of the y axis, this way it
            #always plots at the surface.
            ax.text(offset/df,
                    self.plot_z.min(),
                    self.station_marker,
                    horizontalalignment='center',
                    verticalalignment='baseline',
                    fontdict={'size':self.ms,'color':self.station_color})
                    
            #put station id onto station marker
            #if there is a station id index
            if self.station_id != None:
                ax.text(offset/df,
                        -self.station_font_pad*pf,
                        name[self.station_id[0]:self.station_id[1]],
                        horizontalalignment='center',
                        verticalalignment='baseline',
                        fontdict=fdict)
            #otherwise put on the full station name found form data file
            else:
                ax.text(offset/df,
                        -self.station_font_pad*pf,
                        name,
                        horizontalalignment='center',
                        verticalalignment='baseline',
                        fontdict=fdict)
        
        #set the initial limits of the plot to be square about the profile line  
        if self.ylimits == None:  
            ax.set_ylim(abs(self.station_locations.max()-
                            self.station_locations.min())/df,
                        -self.ypad*pf)
        else:
            ax.set_ylim(self.ylimits[1]*pf,
                        (self.ylimits[0]-self.ypad)*pf)
        if self.xlimits == None:
            ax.set_xlim(self.station_locations.min()/df-(self.xpad*pf),
                       self.station_locations.max()/df+(self.xpad*pf))
        else:
            ax.set_xlim(self.xlimits[0]*pf, self.xlimits[1]*pf)
            
        #set the axis properties
        ax.xaxis.set_minor_locator(MultipleLocator(self.xminorticks*pf))
        ax.yaxis.set_minor_locator(MultipleLocator(self.yminorticks*pf))
        
        #set axes labels
        ax.set_xlabel('Horizontal Distance ({0})'.format(self.yscale),
                      fontdict={'size':self.font_size+2,'weight':'bold'})
        ax.set_ylabel('Depth ({0})'.format(self.yscale),
                      fontdict={'size':self.font_size+2,'weight':'bold'})

        
        #put a grid on if one is desired    
        if self.grid is not None:
            ax.grid(alpha=.3, which=self.grid, lw=.35)
        
        #set title as rms and roughness
        if type(self.title) is str:
            if self.title == 'on':
                titlestr = os.path.join(os.path.basename(
                                        os.path.dirname(self.iter_fn)),
                                        os.path.basename(self.iter_fn))
                ax.set_title('{0}: RMS={1:.2f}, Roughness={2:.0f}'.format(
                             titlestr,self.misfit_value, self.roughness_value),
                             fontdict={'size':self.font_size+1,
                                       'weight':'bold'})
            else:
                ax.set_title('{0}; RMS={1:.2f}, Roughness={2:.0f}'.format(
                            self.title, self.misfit_value, 
                            self.roughness_value),
                            fontdict={'size':self.font_size+1,
                                      'weight':'bold'})
        else:
            print 'RMS {0:.2f}, Roughness={1:.0f}'.format(self.misfit_value,
                                                          self.roughness_value) 
        
        #plot forward model mesh
        #making an extended list seperated by None's speeds up the plotting
        #by as much as 99 percent, handy
        if self.femesh == 'on':
            row_line_xlist = []
            row_line_ylist = []
            for xx in self.plot_x/df:
                row_line_xlist.extend([xx,xx])
                row_line_xlist.append(None)
                row_line_ylist.extend([0, self.plot_zy[0]/df])
                row_line_ylist.append(None)
            
            #plot column lines (variables are a little bit of a misnomer)
            ax.plot(row_line_xlist, 
                    row_line_ylist, 
                    color='k', 
                    lw=.5)

            col_line_xlist = []
            col_line_ylist = []            
            for yy in self.plot_z/df:
                col_line_xlist.extend([self.plot_x[0]/df, 
                                      self.plot_x[-1]/df])
                col_line_xlist.append(None)
                col_line_ylist.extend([yy, yy])
                col_line_ylist.append(None)
            
            #plot row lines (variables are a little bit of a misnomer)
            ax.plot(col_line_xlist, 
                    col_line_ylist,
                    color='k',
                    lw=.5)
                        
        if self.femesh_triangles == 'on':
            row_line_xlist = []
            row_line_ylist = []
            for xx in self.plot_x/df:
                row_line_xlist.extend([xx,xx])
                row_line_xlist.append(None)
                row_line_ylist.extend([0, self.plot_z[0]/df])
                row_line_ylist.append(None)
                
            #plot columns
            ax.plot(row_line_xlist, 
                    row_line_ylist, 
                    color='k', 
                    lw=.5)

            col_line_xlist = []
            col_line_ylist = []            
            for yy in self.plot_z/df:
                col_line_xlist.extend([self.plot_x[0]/df, 
                                      self.plot_x[-1]/df])
                col_line_xlist.append(None)
                col_line_ylist.extend([yy, yy])
                col_line_ylist.append(None)
            
            #plot rows
            ax.plot(col_line_xlist, 
                    col_line_ylist,
                    color='k',
                    lw=.5)

            diag_line_xlist = []
            diag_line_ylist = []
            for xi, xx in enumerate(self.plot_x[:-1]/df):
                for yi, yy in enumerate(self.plot_z[:-1]/df):
                    diag_line_xlist.extend([xx, self.plot_x[xi+1]/df])
                    diag_line_xlist.append(None)
                    diag_line_xlist.extend([xx, self.plot_x[xi+1]/df])
                    diag_line_xlist.append(None)
                    
                    diag_line_ylist.extend([yy, self.plot_z[yi+1]/df])
                    diag_line_ylist.append(None)
                    diag_line_ylist.extend([self.plot_z[yi+1]/df, yy])
                    diag_line_ylist.append(None)
            
            #plot diagonal lines.
            ax.plot(diag_line_xlist, 
                    diag_line_ylist,
                    color='k',
                    lw=.5)
        
        #plot the regularization mesh
        if self.regmesh == 'on':
            linelist = []
            for ii in range(len(self.rows)):
                #get the number of layers to combine
                #this index will be the first index in the vertical direction
                ny1 = self.rows[:ii,0].sum()
                
                #the second index  in the vertical direction
                ny2 = ny1+self.rows[ii][0]
                
                #make the list of amalgamated columns an array for ease
                lc = np.array(self.cols[ii])
                yline = ax.plot([self.plot_x[0]/df,self.plot_x[-1]/df],
                                [self.plot_z[-ny1]/df,
                                 self.plot_z[-ny1]/df],
                                color='b',
                                lw=.5)
                                 
                linelist.append(yline)

                #loop over the number of amalgamated blocks
                for jj in range(len(self.cols[ii])):
                    #get first in index in the horizontal direction
                    nx1 = lc[:jj].sum()
                    
                    #get second index in horizontal direction
                    nx2 = nx1+lc[jj]
                    try:
                        if ny1 == 0:
                            ny1 = 1
                        xline = ax.plot([self.plot_x[nx1]/df,
                                         self.plot_x[nx1]/df],
                                        [self.plot_z[-ny1]/df,
                                         self.plot_z[-ny2]/df],
                                        color='b',
                                        lw=.5)
                        linelist.append(xline)
                    except IndexError:
                        pass
                    
        ##plot the mesh block numbers
        if self.meshnum == 'on':
            kk = 1
            for yy in self.plot_z[::-1]/df:
                for xx in self.plot_x/df:
                    ax.text(xx, yy, '{0}'.format(kk),
                            fontdict={'size':self.meshnum_font_size})
                    kk+=1
                    
        ##plot regularization block numbers
        if self.blocknum == 'on':
            kk=1
            for ii in range(len(self.rows)):
                #get the number of layers to combine
                #this index will be the first index in the vertical direction
                ny1 = self.rows[:ii,0].sum()
                
                #the second index  in the vertical direction
                ny2 = ny1+self.rows[ii][0]
                #make the list of amalgamated columns an array for ease
                lc = np.array(self.cols[ii])
                #loop over the number of amalgamated blocks
                for jj in range(len(self.cols[ii])):
                    #get first in index in the horizontal direction
                    nx1 = lc[:jj].sum()
                    #get second index in horizontal direction
                    nx2 = nx1+lc[jj]
                    try:
                        if ny1 == 0:
                            ny1 = 1
                        #get center points of the blocks
                        yy = self.plot_z[-ny1]-(self.plot_z[-ny1]-
                                                self.plot_z[-ny2])/2
                        xx = self.plot_x[nx1]-\
                             (self.plot_x[nx1]-self.plot_x[nx2])/2
                        #put the number
                        ax.text(xx/df, yy/df, '{0}'.format(kk),
                                fontdict={'size':self.block_font_size},
                                horizontalalignment='center',
                                verticalalignment='center')
                        kk+=1
                    except IndexError:
                        pass
           
        plt.show()
        
        #make attributes that can be manipulated
        self.ax = ax
        self.cbax = cb
        
    def redraw_plot(self):
        """
        redraw plot if parameters were changed
        
        use this function if you updated some attributes and want to re-plot.
        
        :Example: ::
            
            >>> # change the color and marker of the xy components
            >>> import mtpy.modeling.occam2d as occam2d
            >>> ocd = occam2d.Occam2DData(r"/home/occam2d/Data.dat")
            >>> p1 = ocd.plotAllResponses()
            >>> #change line width
            >>> p1.lw = 2
            >>> p1.redraw_plot()
        """
        
        plt.close(self.fig)
        self.plot()
        
    def save_figure(self, save_fn, file_format='pdf', orientation='portrait', 
                  fig_dpi=None, close_fig='y'):
        """
        save_plot will save the figure to save_fn.
        
        Arguments:
        -----------
        
            **save_fn** : string
                          full path to save figure to, can be input as
                          * directory path -> the directory path to save to
                            in which the file will be saved as 
                            save_fn/station_name_PhaseTensor.file_format
                            
                          * full path -> file will be save to the given 
                            path.  If you use this option then the format
                            will be assumed to be provided by the path
                            
            **file_format** : [ pdf | eps | jpg | png | svg ]
                              file type of saved figure pdf,svg,eps... 
                              
            **orientation** : [ landscape | portrait ]
                              orientation in which the file will be saved
                              *default* is portrait
                              
            **fig_dpi** : int
                          The resolution in dots-per-inch the file will be
                          saved.  If None then the dpi will be that at 
                          which the figure was made.  I don't think that 
                          it can be larger than dpi of the figure.
                          
            **close_plot** : [ y | n ]
                             * 'y' will close the plot after saving.
                             * 'n' will leave plot open
                          
        :Example: ::
            
            >>> # to save plot as jpg
            >>> import mtpy.modeling.occam2d as occam2d
            >>> dfn = r"/home/occam2d/Inv1/data.dat"
            >>> ocd = occam2d.Occam2DData(dfn)
            >>> ps1 = ocd.plotPseudoSection()
            >>> ps1.save_plot(r'/home/MT/figures', file_format='jpg')
            
        """

        if fig_dpi == None:
            fig_dpi = self.fig_dpi
            
        if os.path.isdir(save_fn) == False:
            file_format = save_fn[-3:]
            self.fig.savefig(save_fn, dpi=fig_dpi, format=file_format,
                             orientation=orientation, bbox_inches='tight')
            
        else:
            save_fn = os.path.join(save_fn, 'OccamModel.'+
                                    file_format)
            self.fig.savefig(save_fn, dpi=fig_dpi, format=file_format,
                        orientation=orientation, bbox_inches='tight')
        
        if close_fig == 'y':
            plt.clf()
            plt.close(self.fig)
        
        else:
            pass
        
        self.fig_fn = save_fn
        print 'Saved figure to: '+self.fig_fn
        
    def update_plot(self):
        """
        update any parameters that where changed using the built-in draw from
        canvas.  
        
        Use this if you change an of the .fig or axes properties
        
        :Example: ::
            
            >>> # to change the grid lines to only be on the major ticks
            >>> import mtpy.modeling.occam2d as occam2d
            >>> dfn = r"/home/occam2d/Inv1/data.dat"
            >>> ocd = occam2d.Occam2DData(dfn)
            >>> ps1 = ocd.plotAllResponses()
            >>> [ax.grid(True, which='major') for ax in [ps1.axrte,ps1.axtep]]
            >>> ps1.update_plot()
        
        """

        self.fig.canvas.draw()
                          
    def __str__(self):
        """
        rewrite the string builtin to give a useful message
        """
        
        return ("Plots the resistivity found by Occam2D")

#==============================================================================
# plot L2 curve of iteration vs rms
#==============================================================================
class PlotL2():
    """
    plot L2 curve of iteration vs rms and roughness
    
    Arguments:
    ----------
        **rms_arr** : structured array with keys:
                      * 'iteration' --> for iteration number (int)
                      * 'rms' --> for rms (float)
                      * 'roughness' --> for roughness (float)
                      
    ======================= ===================================================
    Keywords/attributes     Description
    ======================= ===================================================
    ax1                     matplotlib.axes instance for rms vs iteration
    ax2                     matplotlib.axes instance for roughness vs rms
    fig                     matplotlib.figure instance
    fig_dpi                 resolution of figure in dots-per-inch
    fig_num                 number of figure instance
    fig_size                size of figure in inches (width, height)
    font_size               size of axes tick labels, axes labels is +2
    plot_yn                 [ 'y' | 'n']
                            'y' --> to plot on instantiation
                            'n' --> to not plot on instantiation
    rms_arr                 structure np.array as described above
    rms_color               color of rms marker and line
    rms_lw                  line width of rms line
    rms_marker              marker for rms values
    rms_marker_size         size of marker for rms values
    rms_mean_color          color of mean line
    rms_median_color        color of median line
    rough_color             color of roughness line and marker
    rough_font_size         font size for iteration number inside roughness 
                            marker
    rough_lw                line width for roughness line 
    rough_marker            marker for roughness
    rough_marker_size       size of marker for roughness
    subplot_bottom          subplot spacing from bottom  
    subplot_left            subplot spacing from left  
    subplot_right           subplot spacing from right
    subplot_top             subplot spacing from top
    ======================= ===================================================
   
    =================== =======================================================
    Methods             Description
    =================== =======================================================
    plot                plots L2 curve.  
    redraw_plot         call redraw_plot to redraw the figures, 
                        if one of the attributes has been changed
    save_figure         saves the matplotlib.figure instance to desired 
                        location and format
    =================== ======================================================
     
    """
    
    def __init__(self, iter_fn_list, **kwargs):
        self.iter_fn_list = iter_fn_list
        self.rms_arr = None
        self.rough_arr = None
        
        
        self.subplot_right = .98
        self.subplot_left = .085
        self.subplot_top = .91
        self.subplot_bottom = .1
        
        self.fig_num = kwargs.pop('fig_num', 1)
        self.fig_size = kwargs.pop('fig_size', [6, 6])
        self.fig_dpi = kwargs.pop('dpi', 300)
        self.font_size = kwargs.pop('font_size', 8)
        
        self.rms_lw = kwargs.pop('rms_lw', 1)
        self.rms_marker = kwargs.pop('rms_marker', 'd')
        self.rms_color = kwargs.pop('rms_color', 'k')
        self.rms_marker_size = kwargs.pop('rms_marker_size', 5)
        self.rms_median_color = kwargs.pop('rms_median_color', 'red')
        self.rms_mean_color = kwargs.pop('rms_mean_color', 'orange')
        
        self.rough_lw = kwargs.pop('rough_lw', .75)
        self.rough_marker = kwargs.pop('rough_marker', 'o')
        self.rough_color = kwargs.pop('rough_color', 'b')
        self.rough_marker_size = kwargs.pop('rough_marker_size', 7)
        self.rough_font_size = kwargs.pop('rough_font_size', 6)
        
        
        self.plot_yn = kwargs.pop('plot_yn', 'y')
        if self.plot_yn == 'y':
            self.plot()
            
    def _get_values(self):
        """
        get rms and roughness values from iteration files
        """
        self.rms_arr = np.zeros((len(self.iter_fn_list), 2))
        self.rough_arr = np.zeros((len(self.iter_fn_list), 2))
        
        for ii, itfn in enumerate(self.iter_fn_list):
            m_object = Model(itfn)
            m_object.read_iter_file()
            m_index = int(m_object.iteration)
            self.rms_arr[ii, 1] = float(m_object.misfit_value)
            self.rms_arr[ii, 0] = m_index
            self.rough_arr[ii, 1] = float(m_object.roughness_value)
            self.rough_arr[ii, 0] = m_index
            
        #sort by iteration number
#        self.rms_arr = np.sort(self.rms_arr, axis=1)
#        self.rough_arr = np.sort(self.rough_arr, axis=1)
        
        
    def plot(self):
        """
        plot L2 curve
        """
        
        self._get_values()
        nr = self.rms_arr.shape[0]
        med_rms = np.median(self.rms_arr[1:, 1])
        mean_rms = np.mean(self.rms_arr[1:, 1])
        
        #set the dimesions of the figure
        plt.rcParams['font.size'] = self.font_size
        plt.rcParams['figure.subplot.left'] = self.subplot_left
        plt.rcParams['figure.subplot.right'] = self.subplot_right
        plt.rcParams['figure.subplot.bottom'] = self.subplot_bottom
        plt.rcParams['figure.subplot.top'] = self.subplot_top
        
        #make figure instance
        self.fig = plt.figure(self.fig_num,self.fig_size, dpi=self.fig_dpi)
        plt.clf()
        
        #make a subplot for RMS vs Iteration
        self.ax1 = self.fig.add_subplot(1, 1, 1)
        
        #plot the rms vs iteration
        l1, = self.ax1.plot(self.rms_arr[:, 0],
                            self.rms_arr[:, 1],
                            '-k', 
                            lw=1,
                            marker='d',
                            ms=5)
        
        #plot the median of the RMS
        m1, = self.ax1.plot(self.rms_arr[:, 0],
                            np.repeat(med_rms, nr),
                            ls='--',
                            color=self.rms_median_color,
                            lw=self.rms_lw*.75)
        
        #plot the mean of the RMS
        m2, = self.ax1.plot(self.rms_arr[:, 0],
                            np.repeat(mean_rms, nr),
                            ls='--',
                            color=self.rms_mean_color,
                            lw=self.rms_lw*.75)
    
        #make subplot for RMS vs Roughness Plot
        self.ax2 = self.ax1.twiny()
        
        self.ax2.set_xlim(self.rough_arr[1:, 1].min(), 
                          self.rough_arr[1:, 1].max())
            
        self.ax1.set_ylim(np.floor(self.rms_arr[1:,1].min()),
                          self.rms_arr[1:, 1].max())
        
        #plot the rms vs roughness 
        l2, = self.ax2.plot(self.rough_arr[:, 1],
                            self.rms_arr[:, 1],
                            ls='--',
                            color=self.rough_color,
                            lw=self.rough_lw,
                            marker=self.rough_marker,
                            ms=self.rough_marker_size,
                            mfc='white')
       
        #plot the iteration number inside the roughness marker                     
        for rms, ii, rough in zip(self.rms_arr[:, 1], self.rms_arr[:, 0], 
                           self.rough_arr[:, 1]):
            #need this because if the roughness is larger than this number
            #matplotlib puts the text out of bounds and a draw_text_image
            #error is raised and file cannot be saved, also the other 
            #numbers are not put in.
            if rough > 1e8:
                pass
            else:
                self.ax2.text(rough,
                              rms,
                              '{0}'.format(ii),
                              horizontalalignment='center',
                              verticalalignment='center',
                              fontdict={'size':self.rough_font_size,
                                        'weight':'bold',
                                        'color':self.rough_color})
        
        #make a legend
        self.ax1.legend([l1, l2, m1, m2],
                        ['RMS', 'Roughness',
                         'Median_RMS={0:.2f}'.format(med_rms),
                         'Mean_RMS={0:.2f}'.format(mean_rms)],
                         ncol=1,
                         loc='upper right',
                         columnspacing=.25,
                         markerscale=.75,
                         handletextpad=.15)
                    
        #set the axis properties for RMS vs iteration
        self.ax1.yaxis.set_minor_locator(MultipleLocator(.1))
        self.ax1.xaxis.set_minor_locator(MultipleLocator(1))
        self.ax1.xaxis.set_major_locator(MultipleLocator(1))
        self.ax1.set_ylabel('RMS', 
                            fontdict={'size':self.font_size+2,
                                      'weight':'bold'})                                   
        self.ax1.set_xlabel('Iteration',
                            fontdict={'size':self.font_size+2,
                                      'weight':'bold'})
        self.ax1.grid(alpha=.25, which='both', lw=self.rough_lw)
        self.ax2.set_xlabel('Roughness',
                            fontdict={'size':self.font_size+2,
                                      'weight':'bold',
                                      'color':self.rough_color})


        
        for t2 in self.ax2.get_xticklabels():
            t2.set_color(self.rough_color)
            
        plt.show()
            
    def redraw_plot(self):
        """
        redraw plot if parameters were changed
        
        use this function if you updated some attributes and want to re-plot.
        
        :Example: ::
            
            >>> # change the color and marker of the xy components
            >>> import mtpy.modeling.occam2d as occam2d
            >>> ocd = occam2d.Occam2DData(r"/home/occam2d/Data.dat")
            >>> p1 = ocd.plotAllResponses()
            >>> #change line width
            >>> p1.lw = 2
            >>> p1.redraw_plot()
        """
        
        plt.close(self.fig)
        self.plot()
        
    def save_figure(self, save_fn, file_format='pdf', orientation='portrait', 
                  fig_dpi=None, close_fig='y'):
        """
        save_plot will save the figure to save_fn.
        
        Arguments:
        -----------
        
            **save_fn** : string
                          full path to save figure to, can be input as
                          * directory path -> the directory path to save to
                            in which the file will be saved as 
                            save_fn/station_name_PhaseTensor.file_format
                            
                          * full path -> file will be save to the given 
                            path.  If you use this option then the format
                            will be assumed to be provided by the path
                            
            **file_format** : [ pdf | eps | jpg | png | svg ]
                              file type of saved figure pdf,svg,eps... 
                              
            **orientation** : [ landscape | portrait ]
                              orientation in which the file will be saved
                              *default* is portrait
                              
            **fig_dpi** : int
                          The resolution in dots-per-inch the file will be
                          saved.  If None then the dpi will be that at 
                          which the figure was made.  I don't think that 
                          it can be larger than dpi of the figure.
                          
            **close_plot** : [ y | n ]
                             * 'y' will close the plot after saving.
                             * 'n' will leave plot open
                          
        :Example: ::
            
            >>> # to save plot as jpg
            >>> import mtpy.modeling.occam2d as occam2d
            >>> dfn = r"/home/occam2d/Inv1/data.dat"
            >>> ocd = occam2d.Occam2DData(dfn)
            >>> ps1 = ocd.plotPseudoSection()
            >>> ps1.save_plot(r'/home/MT/figures', file_format='jpg')
            
        """

        if fig_dpi == None:
            fig_dpi = self.fig_dpi
            
        if os.path.isdir(save_fn) == False:
            file_format = save_fn[-3:]
            self.fig.savefig(save_fn, dpi=fig_dpi, format=file_format,
                             orientation=orientation, bbox_inches='tight')
            
        else:
            save_fn = os.path.join(save_fn, '_L2.'+
                                    file_format)
            self.fig.savefig(save_fn, dpi=fig_dpi, format=file_format,
                        orientation=orientation, bbox_inches='tight')
        
        if close_fig == 'y':
            plt.clf()
            plt.close(self.fig)
        
        else:
            pass
        
        self.fig_fn = save_fn
        print 'Saved figure to: '+self.fig_fn
        
    def update_plot(self):
        """
        update any parameters that where changed using the built-in draw from
        canvas.  
        
        Use this if you change an of the .fig or axes properties
        
        :Example: ::
            
            >>> # to change the grid lines to only be on the major ticks
            >>> import mtpy.modeling.occam2d as occam2d
            >>> dfn = r"/home/occam2d/Inv1/data.dat"
            >>> ocd = occam2d.Occam2DData(dfn)
            >>> ps1 = ocd.plotAllResponses()
            >>> [ax.grid(True, which='major') for ax in [ps1.axrte,ps1.axtep]]
            >>> ps1.update_plot()
        
        """

        self.fig.canvas.draw()
                          
    def __str__(self):
        """
        rewrite the string builtin to give a useful message
        """
        
        return ("Plots RMS vs Iteration computed by Occam2D")

#==============================================================================
# plot pseudo section of data and model response                
#==============================================================================
class PlotPseudoSection(object):
    """
    plot a pseudo section of the data and response if given
    
        
    Arguments:
    -------------
        **rp_list** : list of dictionaries for each station with keywords:
                
                * *station* : string
                             station name
                
                * *offset* : float
                             relative offset
                
                * *resxy* : np.array(nf,4)
                            TE resistivity and error as row 0 and 1 respectively
                
                * *resyx* : np.array(fn,4)
                            TM resistivity and error as row 0 and 1 respectively
                
                * *phasexy* : np.array(nf,4)
                              TE phase and error as row 0 and 1 respectively
                
                * *phaseyx* : np.array(nf,4)
                              Tm phase and error as row 0 and 1 respectively
                
                * *realtip* : np.array(nf,4)
                              Real Tipper and error as row 0 and 1 respectively
                
                * *imagtip* : np.array(nf,4)
                              Imaginary Tipper and error as row 0 and 1 
                              respectively
                
                Note: that the resistivity will be in log10 space.  Also, there
                are 2 extra rows in the data arrays, this is to put the 
                response from the inversion.  
        
        **period** : np.array of periods to plot that correspond to the index
                     values of each rp_list entry ie. resxy.
    
    ==================== ==================================================
    key words            description
    ==================== ==================================================
    axmpte               matplotlib.axes instance for TE model phase
    axmptm               matplotlib.axes instance for TM model phase
    axmrte               matplotlib.axes instance for TE model app. res 
    axmrtm               matplotlib.axes instance for TM model app. res 
    axpte                matplotlib.axes instance for TE data phase 
    axptm                matplotlib.axes instance for TM data phase
    axrte                matplotlib.axes instance for TE data app. res.
    axrtm                matplotlib.axes instance for TM data app. res.
    cb_pad               padding between colorbar and axes
    cb_shrink            percentage to shrink the colorbar to
    fig                  matplotlib.figure instance
    fig_dpi              resolution of figure in dots per inch
    fig_num              number of figure instance
    fig_size             size of figure in inches (width, height)
    font_size            size of font in points
    label_list            list to label plots
    ml                   factor to label stations if 2 every other station
                         is labeled on the x-axis
    period               np.array of periods to plot
    phase_cmap           color map name of phase
    phase_limits_te      limits for te phase in degrees (min, max)
    phase_limits_tm      limits for tm phase in degrees (min, max)            
    plot_resp            [ 'y' | 'n' ] to plot response
    plot_yn              [ 'y' | 'n' ] 'y' to plot on instantiation

    res_cmap             color map name for resistivity
    res_limits_te        limits for te resistivity in log scale (min, max)
    res_limits_tm        limits for tm resistivity in log scale (min, max)
    rp_list               list of dictionaries as made from read2Dresp
    station_id           index to get station name (min, max)
    station_list          station list got from rp_list
    subplot_bottom       subplot spacing from bottom (relative coordinates) 
    subplot_hspace       vertical spacing between subplots
    subplot_left         subplot spacing from left  
    subplot_right        subplot spacing from right
    subplot_top          subplot spacing from top
    subplot_wspace       horizontal spacing between subplots
    ==================== ==================================================
    
    =================== =======================================================
    Methods             Description
    =================== =======================================================
    plot                plots a pseudo-section of apparent resistiviy and phase
                        of data and model if given.  called on instantiation 
                        if plot_yn is 'y'.
    redraw_plot         call redraw_plot to redraw the figures, 
                        if one of the attributes has been changed
    save_figure         saves the matplotlib.figure instance to desired 
                        location and format
    =================== =======================================================
                    
   :Example: ::
        
        >>> import mtpy.modeling.occam2d as occam2d
        >>> ocd = occam2d.Occam2DData()
        >>> rfile = r"/home/Occam2D/Line1/Inv1/Test_15.resp"
        >>> ocd.data_fn = r"/home/Occam2D/Line1/Inv1/DataRW.dat"
        >>> ps1 = ocd.plot2PseudoSection(resp_fn=rfile) 
    
    """
    
    def __init__(self, data_fn, resp_fn=None, **kwargs):
        
        self.data_fn = data_fn
        self.resp_fn = resp_fn
        
        self.plot_resp = kwargs.pop('plot_resp', 'y')
        if self.resp_fn is None:
            self.plot_resp = 'n'
        
        self.label_list = [r'$\rho_{TE-Data}$',r'$\rho_{TE-Model}$',
                          r'$\rho_{TM-Data}$',r'$\rho_{TM-Model}$',
                          '$\phi_{TE-Data}$','$\phi_{TE-Model}$',
                          '$\phi_{TM-Data}$','$\phi_{TM-Model}$']
        
        self.phase_limits_te = kwargs.pop('phase_limits_te', (-5, 95))
        self.phase_limits_tm = kwargs.pop('phase_limits_tm', (-5, 95))
        self.res_limits_te = kwargs.pop('res_limits_te', (0,3))
        self.res_limits_tm = kwargs.pop('res_limits_tm', (0,3))
        
        self.phase_cmap = kwargs.pop('phase_cmap', 'jet')
        self.res_cmap = kwargs.pop('res_cmap', 'jet_r')
        
        self.ml = kwargs.pop('ml', 2)
        self.station_id = kwargs.pop('station_id', [0,4])

        self.fig_num = kwargs.pop('fig_num', 1)
        self.fig_size = kwargs.pop('fig_size', [6, 6])
        self.fig_dpi = kwargs.pop('dpi', 300)
        
        self.subplot_wspace = .025
        self.subplot_hspace = .0
        self.subplot_right = .95
        self.subplot_left = .085
        self.subplot_top = .97
        self.subplot_bottom = .1

        self.font_size = kwargs.pop('font_size', 6)
        
        self.plot_type = kwargs.pop('plot_type', '1')
        self.plot_num = kwargs.pop('plot_num', 2)
        self.plot_yn = kwargs.pop('plot_yn', 'y')
        
        self.cb_shrink = .7
        self.cb_pad = .015
        
        self.axrte = None
        self.axrtm = None
        self.axpte = None
        self.axptm = None
        self.axmrte = None
        self.axmrtm = None
        self.axmpte = None
        self.axmptm = None
        
        self.te_res_arr = None
        self.tm_res_arr = None
        self.te_phase_arr = None
        self.tm_phase_arr = None
        
        self.fig = None
        
        if self.plot_yn == 'y':
            self.plot()
                        
    def plot(self):
        """
        plot pseudo section of data and response if given
        
        """
        if self.plot_resp == 'y':
            nr = 2
        else:
            nr = 1
            
        data_obj = Data()
        data_obj.read_data_file(self.data_fn)
        
        if self.resp_fn is not None:
            resp_obj = Response()
            resp_obj.read_response_file(self.resp_fn)
            
        ns = len(data_obj.station_list)
        nf = len(data_obj.period)
        ylimits = (data_obj.period.max(), data_obj.period.min())
        
        #make a grid for pcolormesh so you can have a log scale
        #get things into arrays for plotting
        offset_list = np.zeros(ns+1)
        te_res_arr = np.ones((nf, ns, nr))    
        tm_res_arr = np.ones((nf, ns, nr))    
        te_phase_arr = np.zeros((nf, ns, nr))    
        tm_phase_arr = np.zeros((nf, ns, nr))
    
        for ii, d_dict in enumerate(data_obj.data):
            offset_list[ii] = d_dict['offset']     
            te_res_arr[:, ii, 0] = d_dict['te_res'][0]
            tm_res_arr[:, ii, 0] = d_dict['tm_res'][0]
            te_phase_arr[:, ii, 0] = d_dict['te_phase'][0]
            tm_phase_arr[:, ii, 0] = d_dict['tm_phase'][0]
            
        #read in response data
        if self.plot_resp == 'y':
            for ii, r_dict in enumerate(resp_obj.resp):     
                te_res_arr[:, ii, 1] = r_dict['te_res'][0]
                tm_res_arr[:, ii, 1] = r_dict['tm_res'][0]
                te_phase_arr[:, ii, 1] = r_dict['te_phase'][0]
                tm_phase_arr[:, ii, 1] = r_dict['tm_phase'][0]
         
        #need to make any zeros 1 for taking log10 
        te_res_arr[np.where(te_res_arr == 0)] = 1.0
        tm_res_arr[np.where(tm_res_arr == 0)] = 1.0
                
        self.te_res_arr = te_res_arr
        self.tm_res_arr = tm_res_arr
        self.te_phase_arr = te_phase_arr
        self.tm_phase_arr = tm_phase_arr
         
        #need to extend the last grid cell because meshgrid expects n+1 cells 
        offset_list[-1] = offset_list[-2]*1.15       
        #make a meshgrid for plotting
        #flip frequency so bottom corner is long period
        dgrid, fgrid = np.meshgrid(offset_list, data_obj.period[::-1])
    
        #make list for station labels
        slabel = [data_obj.station_list[ss][self.station_id[0]:self.station_id[1]] 
                    for ss in range(0, ns, self.ml)]

        xloc = offset_list[0]+abs(offset_list[0]-offset_list[1])/5
        yloc = 1.10*data_obj.period[1]
        
        
        plt.rcParams['font.size'] = self.font_size
        plt.rcParams['figure.subplot.bottom'] = self.subplot_bottom
        plt.rcParams['figure.subplot.top'] = self.subplot_top        
        
        self.fig = plt.figure(self.fig_num, self.fig_size, dpi=self.fig_dpi)
        plt.clf()
           
        if self.plot_resp == 'y':
            
            gs1 = gridspec.GridSpec(1, 2,
                                    left=self.subplot_left,
                                    right=self.subplot_right,
                                    wspace=.15)
        
            gs2 = gridspec.GridSpecFromSubplotSpec(2, 2,
                                                   hspace=self.subplot_hspace,
                                                   wspace=self.subplot_wspace,
                                                   subplot_spec=gs1[0])
            gs3 = gridspec.GridSpecFromSubplotSpec(2, 2,
                                                   hspace=self.subplot_hspace,
                                                   wspace=self.subplot_wspace,
                                                   subplot_spec=gs1[1])
            
            #plot TE resistivity data
            
            self.axrte = plt.Subplot(self.fig, gs2[0, 0])
            self.fig.add_subplot(self.axrte)
            self.axrte.pcolormesh(dgrid, 
                                  fgrid, 
                                  np.flipud(np.log10(te_res_arr[:, :, 0])),
                                  cmap=self.res_cmap,
                                  vmin=self.res_limits_te[0],
                                  vmax=self.res_limits_te[1])
            
            #plot TE resistivity model
            self.axmrte = plt.Subplot(self.fig, gs2[0, 1])
            self.fig.add_subplot(self.axmrte)
            self.axmrte.pcolormesh(dgrid,
                                   fgrid,
                                   np.flipud(np.log10(te_res_arr[:, :, 1])),
                                   cmap=self.res_cmap,
                                   vmin=self.res_limits_te[0],
                                   vmax=self.res_limits_te[1])
            
            #plot TM resistivity data
            self.axrtm = plt.Subplot(self.fig, gs3[0, 0])
            self.fig.add_subplot(self.axrtm)   
            self.axrtm.pcolormesh(dgrid,
                                  fgrid,
                                  np.flipud(np.log10(tm_res_arr[:,:,0])),
                                  cmap=self.res_cmap,
                                  vmin=self.res_limits_tm[0],
                                  vmax=self.res_limits_tm[1])
            
            #plot TM resistivity model
            self.axmrtm = plt.Subplot(self.fig, gs3[0, 1])
            self.fig.add_subplot(self.axmrtm)
            self.axmrtm.pcolormesh(dgrid,
                                   fgrid,
                                   np.flipud(np.log10(tm_res_arr[:,:,1])),
                                   cmap=self.res_cmap,
                                   vmin=self.res_limits_tm[0],
                                   vmax=self.res_limits_tm[1])
    
            #plot TE phase data
            self.axpte = plt.Subplot(self.fig, gs2[1, 0])
            self.fig.add_subplot(self.axpte)
            self.axpte.pcolormesh(dgrid,
                                  fgrid,
                                  np.flipud(te_phase_arr[:,:,0]),
                                  cmap=self.phase_cmap,
                                  vmin=self.phase_limits_te[0],
                                  vmax=self.phase_limits_te[1])
            
            #plot TE phase model
            self.axmpte = plt.Subplot(self.fig, gs2[1, 1])
            self.fig.add_subplot(self.axmpte)
            self.axmpte.pcolormesh(dgrid,
                                   fgrid,
                                   np.flipud(te_phase_arr[:,:,1]),
                                   cmap=self.phase_cmap,
                                   vmin=self.phase_limits_te[0],
                                   vmax=self.phase_limits_te[1])
            
            #plot TM phase data 
            self.axptm = plt.Subplot(self.fig, gs3[1, 0])
            self.fig.add_subplot(self.axptm)              
            self.axptm.pcolormesh(dgrid,
                                  fgrid,
                                  np.flipud(tm_phase_arr[:,:,0]),
                                  cmap=self.phase_cmap,
                                  vmin=self.phase_limits_tm[0],
                                  vmax=self.phase_limits_tm[1])
            
            #plot TM phase model
            self.axmptm = plt.Subplot(self.fig, gs3[1, 1])
            self.fig.add_subplot(self.axmptm)
            self.axmptm.pcolormesh(dgrid,
                                   fgrid,
                                   np.flipud(tm_phase_arr[:,:,1]),
                                   cmap=self.phase_cmap,
                                   vmin=self.phase_limits_tm[0],
                                   vmax=self.phase_limits_tm[1])
            
            axlist=[self.axrte, self.axmrte, self.axrtm, self.axmrtm, 
                   self.axpte, self.axmpte, self.axptm, self.axmptm]
            
            #make everthing look tidy
            for xx, ax in enumerate(axlist):
                ax.semilogy()
                ax.set_ylim(ylimits)
                ax.xaxis.set_ticks(offset_list[np.arange(0, ns, self.ml)])
                ax.xaxis.set_ticks(offset_list, minor=True)
                ax.xaxis.set_ticklabels(slabel)
                ax.set_xlim(offset_list.min(),offset_list.max())
                if np.remainder(xx, 2.0) == 1:
                    plt.setp(ax.yaxis.get_ticklabels(), visible=False)
                    cbx = mcb.make_axes(ax, 
                                        shrink=self.cb_shrink, 
                                        pad=self.cb_pad)
                                        
                if xx < 4:
                    plt.setp(ax.xaxis.get_ticklabels(), visible=False)
                    if xx == 1:
                        cb = mcb.ColorbarBase(cbx[0],cmap=self.res_cmap,
                                norm=Normalize(vmin=self.res_limits_te[0],
                                               vmax=self.res_limits_te[1]))
                        cb.set_ticks(np.arange(int(self.res_limits_te[0]),
                                               int(self.res_limits_te[1])+1))
                        cb.set_ticklabels(['10$^{0}$'.format('{'+str(nn)+'}')
                                            for nn in 
                                            np.arange(int(self.res_limits_te[0]), 
                                                      int(self.res_limits_te[1])+1)])
                    if xx == 3:
                        cb = mcb.ColorbarBase(cbx[0],cmap=self.res_cmap,
                                norm=Normalize(vmin=self.res_limits_tm[0],
                                               vmax=self.res_limits_tm[1]))
                        cb.set_label('App. Res. ($\Omega \cdot$m)',
                                     fontdict={'size':self.font_size+1,
                                               'weight':'bold'})
                        cb.set_label('Resistivity ($\Omega \cdot$m)',
                                     fontdict={'size':self.font_size+1,
                                               'weight':'bold'})
                        cb.set_ticks(np.arange(int(self.res_limits_tm[0]),
                                               int(self.res_limits_tm[1])+1))
                        cb.set_ticklabels(['10$^{0}$'.format('{'+str(nn)+'}')
                                            for nn in 
                                            np.arange(int(self.res_limits_tm[0]), 
                                                      int(self.res_limits_tm[1])+1)])
                else:
                    if xx == 5:
                        cb = mcb.ColorbarBase(cbx[0],cmap=self.phase_cmap,
                                norm=Normalize(vmin=self.phase_limits_te[0],
                                               vmax=self.phase_limits_te[1]))

                    if xx == 7:
                        cb = mcb.ColorbarBase(cbx[0],cmap=self.phase_cmap,
                                norm=Normalize(vmin=self.phase_limits_tm[0],
                                               vmax=self.phase_limits_tm[1]))
                        cb.set_label('Phase (deg)', 
                                     fontdict={'size':self.font_size+1,
                                               'weight':'bold'})
                ax.text(xloc, yloc, self.label_list[xx],
                        fontdict={'size':self.font_size+1},
                        bbox={'facecolor':'white'},
                        horizontalalignment='left',
                        verticalalignment='top')
                if xx == 0 or xx == 4:
                    ax.set_ylabel('Period (s)',
                                  fontdict={'size':self.font_size+2, 
                                  'weight':'bold'})
                if xx>3:
                    ax.set_xlabel('Station',fontdict={'size':self.font_size+2,
                                                      'weight':'bold'})
                
                    
            plt.show()
            
        else: 
            gs1 = gridspec.GridSpec(2, 2,
                        left=self.subplot_left,
                        right=self.subplot_right,
                        hspace=self.subplot_hspace,
                        wspace=self.subplot_wspace)
            
            #plot TE resistivity data
            self.axrte = self.fig.add_subplot(gs1[0, 0])
            self.axrte.pcolormesh(dgrid,
                                  fgrid,
                                  np.flipud(np.log10(te_res_arr[:, :, 0])),
                                  cmap=self.res_cmap,
                                  vmin=self.res_limits_te[0],
                                  vmax=self.res_limits_te[1])
            
            #plot TM resistivity data               
            self.axrtm = self.fig.add_subplot(gs1[0, 1])
            self.axrtm.pcolormesh(dgrid,
                                  fgrid,
                                  np.flipud(np.log10(tm_res_arr[:, :, 0])),
                                  cmap=self.res_cmap,
                                  vmin=self.res_limits_tm[0],
                                  vmax=self.res_limits_tm[1])
            
            #plot TE phase data
            self.axpte = self.fig.add_subplot(gs1[1, 0])
            self.axpte.pcolormesh(dgrid,
                                  fgrid,
                                  np.flipud(te_phase_arr[:, :, 0]),
                                  cmap=self.phase_cmap,
                                  vmin=self.phase_limits_te[0],
                                  vmax=self.phase_limits_te[1])
            
            #plot TM phase data               
            self.axptm = self.fig.add_subplot(gs1[1, 1])
            self.axptm.pcolormesh(dgrid,
                                  fgrid,
                                  np.flipud(tm_phase_arr[:,:, 0]),
                                  cmap=self.phase_cmap,
                                  vmin=self.phase_limits_tm[0],
                                  vmax=self.phase_limits_tm[1])
            
            
            axlist=[self.axrte, self.axrtm, self.axpte, self.axptm]
            
            #make everything look tidy
            for xx,ax in enumerate(axlist):
                ax.semilogy()
                ax.set_ylim(ylimits)
                ax.xaxis.set_ticks(offset_list[np.arange(0, ns, self.ml)])
                ax.xaxis.set_ticks(offset_list, minor=True)
                ax.xaxis.set_ticklabels(slabel)
                ax.grid(True, alpha=.25)
                ax.set_xlim(offset_list.min(),offset_list.max())
                cbx = mcb.make_axes(ax, 
                                    shrink=self.cb_shrink, 
                                    pad=self.cb_pad)
                if xx == 0:
                    plt.setp(ax.xaxis.get_ticklabels(), visible=False)
                    cb = mcb.ColorbarBase(cbx[0],cmap=self.res_cmap,
                                    norm=Normalize(vmin=self.res_limits_te[0],
                                                   vmax=self.res_limits_te[1]))
                    cb.set_ticks(np.arange(self.res_limits_te[0], 
                                           self.res_limits_te[1]+1))
                    cb.set_ticklabels(['10$^{0}$'.format('{'+str(nn)+'}')
                                        for nn in 
                                        np.arange(int(self.res_limits_te[0]), 
                                                  int(self.res_limits_te[1])+1)])
                elif xx == 1:
                    plt.setp(ax.xaxis.get_ticklabels(), visible=False)
                    
                    cb = mcb.ColorbarBase(cbx[0],cmap=self.res_cmap,
                                    norm=Normalize(vmin=self.res_limits_tm[0],
                                                   vmax=self.res_limits_tm[1]))
                    cb.set_label('App. Res. ($\Omega \cdot$m)',
                                 fontdict={'size':self.font_size+1,
                                           'weight':'bold'})
                    cb.set_ticks(np.arange(self.res_limits_tm[0], 
                                           self.res_limits_tm[1]+1))
                    cb.set_ticklabels(['10$^{0}$'.format('{'+str(nn)+'}')
                                        for nn in 
                                        np.arange(int(self.res_limits_tm[0]), 
                                                  int(self.res_limits_tm[1])+1)])
                elif xx == 2:
                    cb = mcb.ColorbarBase(cbx[0],cmap=self.phase_cmap,
                                    norm=Normalize(vmin=self.phase_limits_te[0],
                                                   vmax=self.phase_limits_te[1]))
                    cb.set_ticks(np.arange(self.phase_limits_te[0], 
                                           self.phase_limits_te[1]+1, 15))
                elif xx == 3:
                    cb = mcb.ColorbarBase(cbx[0],cmap=self.phase_cmap,
                                    norm=Normalize(vmin=self.phase_limits_tm[0],
                                                   vmax=self.phase_limits_tm[1]))
                    cb.set_label('Phase (deg)',
                                 fontdict={'size':self.font_size+1,
                                           'weight':'bold'})
                    cb.set_ticks(np.arange(self.phase_limits_te[0], 
                                           self.phase_limits_te[1]+1, 15))
                ax.text(xloc, yloc, self.label_list[xx],
                        fontdict={'size':self.font_size+1},
                        bbox={'facecolor':'white'},
                        horizontalalignment='left',
                        verticalalignment='top')
                if xx == 0 or xx == 2:
                    ax.set_ylabel('Period (s)',
                                  fontdict={'size':self.font_size+2,
                                            'weight':'bold'})
                if xx>1:
                    ax.set_xlabel('Station',fontdict={'size':self.font_size+2,
                                                      'weight':'bold'})
                
                    
            plt.show()
            
    def redraw_plot(self):
        """
        redraw plot if parameters were changed
        
        use this function if you updated some attributes and want to re-plot.
        
        :Example: ::
            
            >>> # change the color and marker of the xy components
            >>> import mtpy.modeling.occam2d as occam2d
            >>> ocd = occam2d.Occam2DData(r"/home/occam2d/Data.dat")
            >>> p1 = ocd.plotPseudoSection()
            >>> #change color of te markers to a gray-blue
            >>> p1.res_cmap = 'seismic_r'
            >>> p1.redraw_plot()
        """
        
        plt.close(self.fig)
        self.plot()
        
    def save_figure(self, save_fn, file_format='pdf', orientation='portrait', 
                  fig_dpi=None, close_plot='y'):
        """
        save_plot will save the figure to save_fn.
        
        Arguments:
        -----------
        
            **save_fn** : string
                          full path to save figure to, can be input as
                          * directory path -> the directory path to save to
                            in which the file will be saved as 
                            save_fn/station_name_PhaseTensor.file_format
                            
                          * full path -> file will be save to the given 
                            path.  If you use this option then the format
                            will be assumed to be provided by the path
                            
            **file_format** : [ pdf | eps | jpg | png | svg ]
                              file type of saved figure pdf,svg,eps... 
                              
            **orientation** : [ landscape | portrait ]
                              orientation in which the file will be saved
                              *default* is portrait
                              
            **fig_dpi** : int
                          The resolution in dots-per-inch the file will be
                          saved.  If None then the dpi will be that at 
                          which the figure was made.  I don't think that 
                          it can be larger than dpi of the figure.
                          
            **close_plot** : [ y | n ]
                             * 'y' will close the plot after saving.
                             * 'n' will leave plot open
                          
        :Example: ::
            
            >>> # to save plot as jpg
            >>> import mtpy.modeling.occam2d as occam2d
            >>> dfn = r"/home/occam2d/Inv1/data.dat"
            >>> ocd = occam2d.Occam2DData(dfn)
            >>> ps1 = ocd.plotPseudoSection()
            >>> ps1.save_plot(r'/home/MT/figures', file_format='jpg')
            
        """

        if fig_dpi == None:
            fig_dpi = self.fig_dpi
            
        if os.path.isdir(save_fn) == False:
            file_format = save_fn[-3:]
            self.fig.savefig(save_fn, dpi=fig_dpi, format=file_format,
                             orientation=orientation, bbox_inches='tight')
            
        else:
            save_fn = os.path.join(save_fn, 'OccamPseudoSection.'+
                                    file_format)
            self.fig.savefig(save_fn, dpi=fig_dpi, format=file_format,
                        orientation=orientation, bbox_inches='tight')
        
        if close_plot == 'y':
            plt.clf()
            plt.close(self.fig)
        
        else:
            pass
        
        self.fig_fn = save_fn
        print 'Saved figure to: '+self.fig_fn
        
    def update_plot(self):
        """
        update any parameters that where changed using the built-in draw from
        canvas.  
        
        Use this if you change an of the .fig or axes properties
        
        :Example: ::
            
            >>> # to change the grid lines to only be on the major ticks
            >>> import mtpy.modeling.occam2d as occam2d
            >>> dfn = r"/home/occam2d/Inv1/data.dat"
            >>> ocd = occam2d.Occam2DData(dfn)
            >>> ps1 = ocd.plotPseudoSection()
            >>> [ax.grid(True, which='major') for ax in [ps1.axrte,ps1.axtep]]
            >>> ps1.update_plot()
        
        """

        self.fig.canvas.draw()
                          
    def __str__(self):
        """
        rewrite the string builtin to give a useful message
        """
        
        return ("Plots a pseudo section of TE and TM modes for data and "
                "response if given.") 

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

