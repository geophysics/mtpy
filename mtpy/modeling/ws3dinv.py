# -*- coding: utf-8 -*-
"""
===============
ws3d
===============

    * Deals with input and output files for ws3dinv
    
    
Created on Sun Aug 25 18:41:15 2013

@author: jpeacock-pr
"""

#==============================================================================

import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap,Normalize
import matplotlib.colorbar as mcb
import matplotlib.gridspec as gridspec
import mtpy.core.z as mtz
import mtpy.core.edi as mtedi
import mtpy.imaging.mtplottools as mtplottools
import matplotlib.widgets as widgets
import matplotlib.colors as colors
import matplotlib.cm as cm
import mtpy.utils.winglink as wl

import mtpy.utils.latlongutmconversion as ll2utm

#==============================================================================
#==============================================================================
# Data class
#==============================================================================
class WSData(object):
    """
    includes tools for reading and writing data files.
    
    """
    
    def __init__(self, **kwargs):
        
        self.save_path = kwargs.pop('save_path', None)
        self.units = kwargs.pop('units', 'mv')
        self.ncol = kwargs.pop('ncols', 5)
        self.ptol = kwargs.pop('ptol', 0.15)
        self.z_err = kwargs.pop('z_err', 0.05)
        self.z_err_map = kwargs.pop('z_err_map', [10,1,1,10])
        self.n_z = kwargs.pop('n_z', 8)
        self.period_list = None
        self.edi_list = None
        self.station_locations = None
        self.wl_site_fn = kwargs.pop('wl_site_fn', None)
        self.wl_out_fn = kwargs.pop('wl_out_fn', None)
        self.data_fn = kwargs.pop('data_fn', None)
        self.station_fn = kwargs.pop('station_fn', None)
        self.data = None
        
        self._data_keys = ['station', 'east', 'north', 'z_data', 'z_data_err',
                           'z_model', 'z_model_err']

        
    def write_data_file(self, period_list, edi_list, station_locations, 
                        **kwargs):
        """
        writes a data file for WSINV3D
        
        Inputs:
        --------
            **period_list** :list
                            periods to extract from edifiles, can get them from 
                            using the function getPeriods.
                            
            **edi_list** : list
                        list of full paths to .edi files to use for inversion
                        
            **station_locations**  : np.ndarray(ns, 2) (east, north) or
                                     structured array
                                     np.ndarray(ns, 3, 
                                                dtype=('station', 
                                                       'east',
                                                       'north'))

                                 This can be found from Make3DGrid.  
                                 Locations are in meters in grid
                                 coordinates.  Should be the same order as 
                                 edi_list.
                                
            **wl_site_fn** : string
                        if you used Winglink to make the model then you need to
                        input the sites filename (full path)
                         
            **wl_out_fn** : string
                       if you used Winglink to make the model need to input the
                       winglink .out file (full path)
                        
            **save_path** : string
                           directory or full path to save data file to, default 
                           path is dirname sites_fn.  
                           saves as: savepath/WSDataFile.dat
                           *Need to input if you did not use Winglink*
                           
            **z_err** : float
                      percent error to give to impedance tensor components in 
                      decimal form --> 10% = 0.10
                      *default* is .05
                      
            **ptol** : float
                      percent tolerance to locate frequencies in case edi files 
                      don't have the same frequencies.  
                      Need to add interpolation.
                       *default* is 0.15
                       
            **z_err_map** :  tuple (zxx, zxy, zyx, zyy)
                           multiple to multiply err of zxx,zxy,zyx,zyy by.
                           Note the total error is zerr*zerrmap[ii]
                           
            **ncol** : int
                    number of columns in out_fn, sometimes it outputs different
                    number of columns.
            
        
        Returns:
        --------
            
            **data_fn** : full path to data file, saved in dirname(sites_fn) or 
                         savepath where savepath can be a directory or full 
                         filename
        """
        
        self.period_list = period_list
        self.edi_list = edi_list
        self.station_locations = station_locations
        
        self.save_path = kwargs.pop('save_path', None)
        self.units = kwargs.pop('units', 'mv')
        self.ncol = kwargs.pop('ncols', 5)
        self.ptol = kwargs.pop('ptol', 0.15)
        self.z_err = kwargs.pop('z_err', 0.05)
        self.z_err_map = kwargs.pop('z_err_map', np.array([10, 1, 1, 10]))
        self.wl_site_fn = kwargs.pop('wl_site_fn', None)
        self.wl_out_fn = kwargs.pop('wl_out_fn', None)
        self.station_fn = kwargs.pop('station_fn', None)
        
        
        #get units correctly
        if self.units == 'mv':
            zconv = 1./796.
            
        #define some lengths
        n_stations = len(edi_list)
        n_periods = len(self.period_list)
        
        #make a structured array to keep things in for convenience
        z_shape = (n_periods, 4)
        data_dtype = [('station', '|S10'),
                      ('east', np.float),
                      ('north', np.float),
                      ('z_data', (np.complex, z_shape)),
                      ('z_data_err', (np.complex, z_shape)),
                      ('z_err_map', (np.complex, z_shape)),
                      ('z_model', (np.complex, z_shape)),
                      ('z_model_err', (np.complex, z_shape))]
        self.data = np.zeros(n_stations, dtype=data_dtype)
        
        #create the output filename
        if self.save_path == None:
            if self.wl_out_fn is not None:
                self.save_path = os.path.dirname(self.wl_site_fn)
            else:
                self.save_path = os.getcwd()
            self.data_fn = os.path.join(self.save_path, 'WSDataFile.dat')
        elif self.save_path.find('.') == -1:
            self.data_fn = os.path.join(self.save_path, 'WSDataFile.dat')
        else:
            self.data_fn = self.save_path
        
        #------get station locations-------------------------------------------
        if self.wl_site_fn != None: 
            if self.wl_out_fn is None:
                raise IOError('Need to input an .out file to get station'
                              'locations, this should be output by Winglink')
            
            #get x and y locations on a relative grid
            east_list, north_list, station_list = \
                            wl.get_station_locations(self.wl_site_fn, 
                                                     self.wl_out_fn, 
                                                     ncol=self.ncol)
            self.data['station'] = station_list
            self.data['east'] = east_list
            self.data['north'] = north_list
        
        #if a station location file is input
        if self.station_fn != None:
            self.read_station_file(self.station_fn)
            self.data['station'] = self.station_locations['station']
            self.data['east'] = self.station_locations['east']
            self.data['north']= self.station_locations['north']
        
        #if the user made a grid in python or some other fashion
        if self.station_locations != None:
            try:
                for dd, sd in enumerate(self.station_locations):
                    self.data['east'][dd] = sd['east_c']
                    self.data['north'][dd] = sd['north_c']
                    self.data['station'][dd] = sd['station']
            except (KeyError, ValueError): 
                self.data['east'] = self.station_locations[:, 0]
                self.data['north']= self.station_locations[:, 1]
        
        #--------find frequencies----------------------------------------------
        linelist = []
        for ss, edi in enumerate(self.edi_list):
            if not os.path.isfile(edi):
                raise IOError('Could not find '+edi)

            z1 = mtedi.Edi()
            z1.readfile(edi)
            sdict = {}
            fspot = {}
            print '{0}{1}{0}'.format('-'*20, z1.station) 
            for ff, f1 in enumerate(self.period_list):
                for kk,f2 in enumerate(z1.period):
                    if f2 >= (1-self.ptol)*f1 and f2 <= (1+self.ptol)*f1:
                        self.data[ss]['z_data'][ff, :] = \
                                                   zconv*z1.Z.z[kk].reshape(4,)
                        self.data[ss]['z_data_err'][ff, :] = \
                                    zconv*z1.Z.z[kk].reshape(4,)*self.z_err
                                    
                        self.data[ss]['z_err_map'][ff, :] = self.z_errmap
                        
                        #estimate the determinant error for quality check later
                        zderr = np.array([abs(z1.Z.zerr[kk, nn, mm])/
                                        abs(z1.Z.z[kk, nn, mm])*100 
                                        for nn in range(2) for mm in range(2)])
                        print '   Matched {0:.6g} to {1:.6g}'.format(f1, f2)
                        fspot['{0:.6g}'.format(f1)] = (kk, f2, zderr[0], 
                                                      zderr[1], zderr[2],
                                                      zderr[3])
                        
                        break
            # these are for quality checks after data file has been written
            sdict['fspot'] = fspot
            sdict['station'] = z1.station
            linelist.append(sdict)
        
        #-----Write data file--------------------------------------------------
        ofid = file(self.data_fn, 'w')
        ofid.write('{0:d} {1:d} {2:d}\n'.format(n_stations, n_periods, 
                                                self.n_z))
        
        #write N-S locations
        ofid.write('Station_Location: N-S \n')
        for ii in range(n_stations/self.n_z+1):
            for ll in range(self.n_z):
                index = ii*self.n_z+ll 
                try:
                    ofid.write('{0:+.4e} '.format(self.data['north'][index]))
                except IndexError:
                    pass
            ofid.write('\n')
        
        #write E-W locations
        ofid.write('Station_Location: E-W \n')
        for ii in range(n_stations/self.n_z+1):
            for ll in range(self.n_z):
                index = ii*self.n_z+ll 
                try:
                    ofid.write('{0:+.4e} '.format(self.data['east'][index]))
                except IndexError:
                    pass
            ofid.write('\n')
            
        #write impedance tensor components
        for ii, p1 in enumerate(self.period_list):
            ofid.write('DATA_Period: {0:3.6f}\n'.format(p1))
            for ss in range(n_stations):
                zline = self.data[ss]['z_data'][ii, :]
                for jj in range(self.n_z/2):
                    ofid.write('{0:+.4e} '.format(zline[jj].real))
                    ofid.write('{0:+.4e} '.format(-zline[jj].imag))
                ofid.write('\n')
        
        #write error as a percentage of Z
        for ii, p1 in enumerate(self.period_list):
            ofid.write('ERROR_Period: {0:3.6f}\n'.format(p1))
            for ss in range(n_stations):
                zline = self.data[ss]['z_data_err'][ii, :]
                for jj in range(self.n_z/2):
                    ofid.write('{0:+.4e} '.format(zline[jj].real))
                    ofid.write('{0:+.4e} '.format(zline[jj].imag))
                ofid.write('\n')
                
        #write error maps
        for ii, p1 in enumerate(self.period_list):
            ofid.write('ERMAP_Period: {0:3.6f}\n'.format(p1))
            for ss in range(n_stations):
                for jj in range(self.n_z/2):
                    ofid.write('{0:.5e} '.format(self.z_err_map[jj]))
                    ofid.write('{0:.5e} '.format(self.z_err_map[jj]))
                ofid.write('\n')
        ofid.close()
        print 'Wrote file to: {0}'.format(self.data_fn)
                           
    def write_station_file(self, east, north, station_list, save_path=None):
        """
        write a station file to go with the data file.
        
        the locations are on a relative grid where (0, 0, 0) is the 
        center of the grid.  Also, the stations are assumed to be in the center
        of the cell.
        
        """
        
        if save_path is None:
            save_path = os.path.join(os.path.dirname(self.data_fn), 
                                     'WS_Station_locations.txt')
        else:
            if os.path.isdir(save_path):
                save_path = os.path.join(save_path, 
                                         'WS_Station_locations.txt')
            else:
                pass
        
        self.station_fn = save_path
            
        sfid = file(self.station_fn, 'w')
        sfid.write('{0:<14}{1:^14}{2:^14}\n'.format('station', 'east', 
                                                    'north'))
        for ee, nn, ss in zip(east, north, station_list):
            ee = '{0:+.4e}'.format(ee)
            nn = '{0:+.4e}'.format(nn)
            sfid.write('{0:<14}{1:^14}{2:^14}\n'.format(ss, ee, nn))
        sfid.close()
        
        print 'Wrote station locations to {0}'.format(self.station_fn)
        
    def read_station_file(self, station_fn):
        """
        read in station file written by write_station_file
        
        Arguments:
        ----------
            **station_fn** : string
                             full path to station file
                             
        Outputs:
        ---------
            **station_locations** : structured array with keys
                                     * station station name
                                     * east_c E-W location in center of cell
                                     * north_c N-S location in center of cell
        
        """

        self.station_fn = station_fn
        
        self.station_locations = np.loadtxt(self.station_fn, skiprows=1, 
                                         dtype=[('station', '|S10'),
                                                ('east_c', np.float),
                                                ('north_c', np.float)])

                    
    def read_data_file(self, data_fn, wl_sites_fn=None, station_fn=None):
        """
        read in data file
        
        Arguments:
        -----------
            **data_fn** : string
                          full path to data file
            **sites_fn** : string
                           full path to sites file output by winglink.  This is
                           to match the station name with station number.
            **station_fn** : string
                             full path to station location file
                             
        Outputs:
        --------
            **data** : structure np.ndarray
                      fills the attribute WSData.data with values
                      
            **period_list** : np.ndarray()
                             fills the period list with values.
        """
        
        if self.units == 'mv':
            zconv = 796.
        else:
            zconv = 1
        
        self.data_fn = data_fn
        
        dfid = file(self.data_fn, 'r')
        dlines = dfid.readlines()
    
        #get size number of stations, number of frequencies, 
        # number of Z components    
        n_stations, n_periods, nz = np.array(dlines[0].strip().split(), 
                                             dtype='int')
        nsstart = 2
        
        self.n_z = nz
        #make a structured array to keep things in for convenience
        z_shape = (n_periods, 4)
        data_dtype = [('station', '|S10'),
                      ('east', np.float),
                      ('north', np.float),
                      ('z_data', (np.complex, z_shape)),
                      ('z_data_err', (np.complex, z_shape)),
                      ('z_err_map', (np.complex, z_shape)),
                      ('z_model', (np.complex, z_shape)),
                      ('z_model_err', (np.complex, z_shape))]
        self.data = np.zeros(n_stations, dtype=data_dtype)
        
        findlist = []
        for ii, dline in enumerate(dlines[1:50], 1):
            if dline.find('Station_Location: N-S') == 0:
                findlist.append(ii)
            elif dline.find('Station_Location: E-W') == 0:
                findlist.append(ii)
            elif dline.find('DATA_Period:') == 0:
                findlist.append(ii)
                
        ncol = len(dlines[nsstart].strip().split())
        
        #get site names if entered a sites file
        if wl_sites_fn != None:
            self.wl_site_fn = wl_sites_fn
            slist, station_list = wl.read_sites_file(self.wl_sites_fn)
            self.data['station'] = station_list
        
        elif station_fn != None:
            self.station_fn = station_fn
            self.read_station_file(self.station_fn)
            self.data['station'] = self.station_locations['station']
        else:
            self.data['station'] = np.arange(n_stations)
            
    
        #get N-S locations
        for ii, dline in enumerate(dlines[findlist[0]+1:findlist[1]],0):
            dline = dline.strip().split()
            for jj in range(ncol):
                try:
                    self.data['north'][ii*ncol+jj] = float(dline[jj])
                except IndexError:
                    pass
                except ValueError:
                    break
                
        #get E-W locations
        for ii, dline in enumerate(dlines[findlist[1]+1:findlist[2]],0):
            dline = dline.strip().split()
            for jj in range(self.n_z):
                try:
                    self.data['east'][ii*ncol+jj] = float(dline[jj])
                except IndexError:
                    pass
                except ValueError:
                    break
        #make some empty array to put stuff into
        self.period_list = np.zeros(n_periods)
        
        #get data
        per = 0
        error_find = False
        errmap_find = False
        for ii, dl in enumerate(dlines[findlist[2]:]):
            if dl.lower().find('period') > 0:
                st = 0

                if dl.lower().find('data') == 0:
                    dkey = 'z_data'
                    self.period_list[per] = float(dl.strip().split()[1])
                    
                elif dl.lower().find('error') == 0:
                    dkey = 'z_data_err'
                    if not error_find:
                        error_find = True
                        per = 0
                        
                elif dl.lower().find('ermap') == 0:
                    dkey = 'z_err_map'
                    if not errmap_find:
                        errmap_find = True
                        per = 0
                    
                #print '-'*20+dkey+'-'*20
                per += 1
                
            else:
                #print st, per
                zline = np.array(dl.strip().split(), dtype=np.float)*zconv
                self.data[st][dkey][per-1,:] = np.array([zline[0]-1j*zline[1],
                                                    zline[2]-1j*zline[3],
                                                    zline[4]-1j*zline[5],
                                                    zline[6]-1j*zline[7]])
                st += 1

#==============================================================================
# mesh class
#==============================================================================
class WSMesh(object):
    """
    make and read a FE mesh grid
    """
    
    def __init__(self, edi_list=None, **kwargs):
        
        self.edi_list = edi_list
        
        # size of cells within station area in meters
        self.cell_size_east = kwargs.pop('cell_size_east', 500)
        self.cell_size_north = kwargs.pop('cell_size_north', 500)
        self.first_layer_thickness = kwargs.pop('first_layer_thickness', 10)
        
        #padding cells on either side
        self.pad_east = kwargs.pop('pad_east', 5)
        self.pad_north = kwargs.pop('pad_north', 5)
        self.pad_z = kwargs.pop('pad_z', 5)
        
        #root of padding cells
        self.pad_root_east = kwargs.pop('pad_root_east', 5)
        self.pad_root_north = kwargs.pop('pad_root_north', 5)
        self.pad_root_z = kwargs.pop('pad_root_z', 2)
        
        #padding in vertical direction
        self.pad_pow_z = kwargs.pop('pad_pow_z', (7, 15))
        
        #number of vertical layers
        self.n_layers = kwargs.pop('n_layers', 30)
        
        #--> attributes to be calculated
        #station information
        self.station_locations = None
       
       #grid nodes
        self.nodes_east = None
        self.nodes_north = None
        self.nodes_z = None
        
        #grid locations
        self.grid_east = None
        self.grid_north = None
        self.grid_z = None
        
        #resistivity model
        self.res_model = None
        self.res_list = None
        
        #inital file stuff
        self.initial_fn = None
        self.save_path = None
        self.title = 'Inital Model File made in MTpy'
        
        
    def make_mesh(self):
        """ 
        create finite element mesh according to parameters set.
        
        The mesh is built by first finding the center of the station area.  
        Then cells are added in the north and east direction with width
        cell_size_east and cell_size_north to the extremeties of the station 
        area.  Padding cells are then added to extend the model to reduce 
        edge effects.  The number of cells are pad_east and pad_north and the
        increase in size is by pad_root_east and pad_root_north.  The station
        locations are then computed as the center of the nearest cell as 
        required by the code.
        
        The vertical cells are built to increase in size exponentially with
        depth.  The first cell depth is first_layer_thickness and should be
        about 1/10th the shortest skin depth.  The layers then increase
        exponentially accoring to pad_root_z for n_layers.  Then the model is
        padded with pad_z number of cells to extend the depth of the model.
        
        """
        if self.edi_list is None:
            raise AttributeError('edi_list is None, need to input a list of '
                                 'edi files to read in.')
                                 
        n_stations = len(self.edi_list)
        
        #make a structured array to put station location information into
        self.station_locations = np.zeros(n_stations,
                                          dtype=[('station','|S10'), 
                                                 ('east', np.float),
                                                 ('north', np.float), 
                                                 ('east_c', np.float),
                                                 ('north_c', np.float)])
        #get station locations in meters
        for ii, edi in enumerate(self.edi_list):
            zz = mtedi.Edi()
            zz.readfile(edi)
            zone, east, north = ll2utm.LLtoUTM(23, zz.lat, zz.lon)
            self.station_locations[ii]['station'] = zz.station
            self.station_locations[ii]['east'] = east
            self.station_locations[ii]['north'] = north
         
        #remove the average distance to get coordinates in a relative space
        self.station_locations['east'] -= self.station_locations['east'].mean()
        self.station_locations['north'] -= self.station_locations['north'].mean()
     
        #translate the stations so they are relative to 0,0
        east_center = (self.station_locations['east'].max()-
                        np.abs(self.station_locations['east'].min()))/2
        north_center = (self.station_locations['north'].max()-
                        np.abs(self.station_locations['north'].min()))/2
        
        #remove the average distance to get coordinates in a relative space
        self.station_locations['east'] -= east_center
        self.station_locations['north'] -= north_center
    
        #pickout the furtherst south and west locations 
        #and put that station as the bottom left corner of the main grid
        west = self.station_locations['east'].min()-self.cell_size_east/2
        east = self.station_locations['east'].max()+self.cell_size_east/2
        south = self.station_locations['north'].min()-self.cell_size_north/2
        north = self.station_locations['north'].max()+self.cell_size_north/2
    
        #-------make a grid around the stations from the parameters above------
        #--> make grid in east-west direction
        #cells within station area
        midxgrid = np.arange(start=west, stop=east+self.cell_size_east,
                             step=self.cell_size_east)
        
        #padding cells on the west side
        pad_west = np.round(-self.cell_size_east*\
                             self.pad_root_east**np.arange(start=.5, stop=3,
                             step=3./self.pad_east))+west
        
        #padding cells on east side
        pad_east = np.round(self.cell_size_east*\
                             self.pad_root_east**np.arange(start=.5, stop=3,
                             step=3./self.pad_east))+east
        
        #make the cells going west go in reverse order and append them to the
        #cells going east
        east_gridr = np.append(np.append(pad_west[::-1], midxgrid), pad_east)
        
        #--> make grid in north-south direction 
        #N-S cells with in station area
        midygrid = np.arange(start=south, stop=north+self.cell_size_north, 
                             step=self.cell_size_north)
        
        #padding cells on south side
        south_pad = np.round(-self.cell_size_north*
                              self.pad_root_north**np.arange(start=.5,
                              stop=3, step=3./self.pad_north))+south
        
        #padding cells on north side
        north_pad = np.round(self.cell_size_north*
                              self.pad_root_north**np.arange(start=.5,
                              stop=3, step=3./self.pad_north))+north
        
        #make the cells going west go in reverse order and append them to the
        #cells going east                      
        north_gridr = np.append(np.append(south_pad[::-1], midygrid), north_pad)
        
        
        #--> make depth grid
        #cells down to number of z-layers
        zgrid1 = self.first_layer_thickness*\
                 self.pad_root_z**np.round(np.arange(0,self.pad_pow_z[0],
                         self.pad_pow_z[0]/(self.n_layers-float(self.pad_z))))
                         
        #pad bottom of grid
        zgrid2 = self.first_layer_thickness*\
                 self.pad_root_z**np.round(np.arange(self.pad_pow_z[0],
                                                     self.pad_pow_z[1],
                             (self.pad_pow_z[1]-self.pad_pow_z[0]/self.pad_z)))
        
        zgrid = np.append(zgrid1, zgrid2)
        
        #---Need to make an array of the individual cell dimensions for
        #   wsinv3d
        east_nodes = east_gridr.copy()    
        nx = east_gridr.shape[0]
        east_nodes[:nx/2] = np.array([abs(east_gridr[ii]-east_gridr[ii+1]) 
                                          for ii in range(int(nx/2))])
        east_nodes[nx/2:] = np.array([abs(east_gridr[ii]-east_gridr[ii+1]) 
                                          for ii in range(int(nx/2)-1, nx-1)])
    
        north_nodes = north_gridr.copy()
        ny = north_gridr.shape[0]
        north_nodes[:ny/2] = np.array([abs(north_gridr[ii]-north_gridr[ii+1]) 
                                       for ii in range(int(ny/2))])
        north_nodes[ny/2:] = np.array([abs(north_gridr[ii]-north_gridr[ii+1]) 
                                       for ii in range(int(ny/2)-1, ny-1)])
                                
        #--put the grids into coordinates relative to the center of the grid
        east_grid = east_nodes.copy()
        east_grid[:int(nx/2)] = -np.array([east_nodes[ii:int(nx/2)].sum() 
                                           for ii in range(int(nx/2))])
        east_grid[int(nx/2):] = np.array([east_nodes[int(nx/2):ii+1].sum() 
                                         for ii in range(int(nx/2), nx)])-\
                                         east_nodes[int(nx/2)]
                                
        north_grid = north_nodes.copy()
        north_grid[:int(ny/2)] = -np.array([north_nodes[ii:int(ny/2)].sum() 
                                            for ii in range(int(ny/2))])
        north_grid[int(ny/2):] = np.array([north_nodes[int(ny/2):ii+1].sum() 
                                            for ii in range(int(ny/2),ny)])-\
                                            north_nodes[int(ny/2)]
                                
        #make nodes attributes
        self.nodes_east = east_nodes
        self.nodes_north = north_nodes
        self.nodes_z = zgrid
        
        self.grid_east = east_grid
        self.grid_north = north_grid
        self.grid_z = zgrid
        
        #make sure that the stations are in the center of the cell as requested
        #by the code.
        for ii in range(n_stations):
            #look for the closest grid line
            xx = [nn for nn, xf in enumerate(east_grid) 
                if xf>(self.station_locations[ii]['east']-self.cell_size_east) 
                and xf<(self.station_locations[ii]['east']+self.cell_size_east)]
            
            #shift the station to the center in the east-west direction
            if east_grid[xx[0]] < self.station_locations[ii]['east']:
                self.station_locations[ii]['east_c'] = \
                                        east_grid[xx[0]]+self.cell_size_east/2
            elif east_grid[xx[0]] > self.station_locations[ii]['east']:
                self.station_locations[ii]['east_c'] = \
                                        east_grid[xx[0]]-self.cell_size_east/2
            
            #look for closest grid line
            yy = [mm for mm, yf in enumerate(north_grid) 
                 if yf>(self.station_locations[ii]['north']-self.cell_size_north) 
                 and yf<(self.station_locations[ii]['north']+self.cell_size_north)]
            
            #shift station to center of cell in north-south direction
            if north_grid[yy[0]] < self.station_locations[ii]['north']:
                self.station_locations[ii]['north_c'] = \
                                    north_grid[yy[0]]+self.cell_size_north/2
            elif north_grid[yy[0]] > self.station_locations[ii]['north']:
                self.station_locations[ii]['north_c'] = \
                                    north_grid[yy[0]]-self.cell_size_north/2
            
        #--> print out useful information                    
        print '-'*15
        print '   Number of stations = {0}'.format(len(self.station_locations))
        print '   Dimensions: '
        print '      e-w = {0}'.format(east_grid.shape[0])
        print '      n-s = {0}'.format(north_grid.shape[0])
        print '       z  = {0} (without 7 air layers)'.format(zgrid.shape[0])
        print '   Extensions: '
        print '      e-w = {0:.1f} (m)'.format(east_nodes.__abs__().sum())
        print '      n-s = {0:.1f} (m)'.format(north_nodes.__abs__().sum())
        print '      0-z = {0:.1f} (m)'.format(zgrid.__abs__().sum())
        print '-'*15

    def plot_mesh(self, east_limits=None, north_limits=None, z_limits=None,
                  **kwargs):
        """
        
        Arguments:
        ----------
            **east_limits** : tuple (xmin,xmax)
                             plot min and max distances in meters for the 
                             E-W direction.  If None, the east_limits
                             will be set to furthest stations east and west.
                             *default* is None
                        
            **north_limits** : tuple (ymin,ymax)
                             plot min and max distances in meters for the 
                             N-S direction.  If None, the north_limits
                             will be set to furthest stations north and south.
                             *default* is None
                        
            **z_limits** : tuple (zmin,zmax)
                            plot min and max distances in meters for the 
                            vertical direction.  If None, the z_limits is
                            set to the number of layers.  Z is positive down
                            *default* is None
        """
        
        fig_size = kwargs.pop('fig_size', [6, 6])
        fig_dpi = kwargs.pop('fig_dpi', 300)
        fig_num = kwargs.pop('fig_num', 1)
        
        station_marker = kwargs.pop('station_marker', 'v')
        marker_color = kwargs.pop('station_color', 'b')
        marker_size = kwargs.pop('marker_size', 2)
        
        line_color = kwargs.pop('line_color', 'k')
        line_width = kwargs.pop('line_width', .75)
        
        plt.rcParams['figure.subplot.hspace'] = .3
        plt.rcParams['figure.subplot.wspace'] = .3
        plt.rcParams['font.size'] = 7
        
        fig = plt.figure(fig_num, figsize=fig_size, dpi=fig_dpi)
        plt.clf()
        
        #---plot map view    
        ax1 = fig.add_subplot(1, 2, 1, aspect='equal')
        
        #make sure the station is in the center of the cell
        ax1.scatter(self.station_locations['east_c'],
                    self.station_locations['north_c'], 
                    marker=station_marker,
                    c=marker_color,
                    s=marker_size)
                
        for xp in self.grid_east:
            ax1.plot([xp,xp],
                     [self.grid_north.min(), self.grid_north.max()],
                     color=line_color,
                     lw=line_width)
            
        for yp in self.grid_north:
            ax1.plot([self.grid_east.min(),self.grid_east.max()],
                      [yp,yp],
                      color=line_color,
                      lw=line_width)
        
        if east_limits == None:
            ax1.set_xlim(self.station_locations['east'].min()-\
                            10*self.cell_size_east,
                         self.station_locations['east'].max()+\
                             10*self.cell_size_east)
        else:
            ax1.set_xlim(east_limits)
        
        if north_limits == None:
            ax1.set_ylim(self.station_locations['north'].min()-\
                            10*self.cell_size_north,
                         self.station_locations['north'].max()+\
                             10*self.cell_size_east)
        else:
            ax1.set_ylim(north_limits)
            
        ax1.set_ylabel('Northing (m)', fontdict={'size':9,'weight':'bold'})
        ax1.set_xlabel('Easting (m)', fontdict={'size':9,'weight':'bold'})
        
        ##----plot depth view
        ax2 = fig.add_subplot(1, 2, 2, aspect='auto', sharex=ax1)
                
        for xp in self.grid_east:
            ax2.plot([xp, xp], 
                     [0, self.grid_z.sum()],
                     color=line_color,
                     lw=line_width)
            
        ax2.scatter(self.station_locations['east_c'],
                    [0]*self.station_locations.shape[0],
                    marker=station_marker,
                    c=marker_color,
                    s=marker_size)
            
        for zz, zp in enumerate(self.grid_z):
            ax2.plot([self.grid_east.min(), self.grid_east.max()],
                     [self.grid_z[0:zz].sum(), self.grid_z[0:zz].sum()],
                     color=line_color,
                     lw=line_width)
        
        if z_limits == None:
            ax2.set_ylim(self.grid_z[:self.n_layers].sum(), -200)
        else:
            ax2.set_ylim(z_limits)
            
        if east_limits == None:
            ax1.set_xlim(self.station_locations['east'].min()-\
                            10*self.cell_size_east,
                         self.station_locations['east'].max()+\
                             10*self.cell_size_east)
        else:
            ax1.set_xlim(east_limits)
            
        ax2.set_ylabel('Depth (m)', fontdict={'size':9, 'weight':'bold'})
        ax2.set_xlabel('Easting (m)', fontdict={'size':9, 'weight':'bold'})  
        
        plt.show()
    
    def write_initial_file(self, save_path=None, res_model=None, res_list=100,
                           title=None, nodes_east=None, nodes_north=None, 
                           nodes_z=None):
        """
        will write an initial file for wsinv3d.  At the moment can only make a 
        layered model that can then be manipulated later.  Input for a layered
        model is in layers which is [(layer1,layer2,resistivity index for reslist)]
        
        Note that x is assumed to be S --> N, y is assumed to be W --> E and
        z is positive downwards. 
        
        Also, the xgrid, ygrid and zgrid are assumed to be the relative distance
        between neighboring nodes.  This is needed because wsinv3d builds the 
        model from the bottom NW corner assuming the cell width from the init file.
        
        Therefore the first line or index=0 is the southern most row of cells, so
        if you build a model by hand the the layer block will look upside down if
        you were to picture it in map view. Confusing, perhaps, but that is the 
        way it is.  
        
        Arguments:
        ----------
        
            **nodes_north** : np.array(nx)
                        block dimensions (m) in the N-S direction. 
                        **Note** that the code reads the grid assuming that
                        index=0 is the southern most point.
            
            **nodes_east** : np.array(ny)
                        block dimensions (m) in the E-W direction.  
                        **Note** that the code reads in the grid assuming that
                        index=0 is the western most point.
                        
            **nodes_z** : np.array(nz)
                        block dimensions (m) in the vertical direction.  
                        This is positive downwards.
                        
            **save_path** : string
                          Path to where the initial file will be saved
                          to savepath/init3d
                          
            **res_list** : float or list
                        The start resistivity as a float or a list of
                        resistivities that coorespond to the starting
                        resistivity model **resmodel**.  
                        This must be input if you input **resmodel**
                        
            **title** : string
                        Title that goes into the first line of savepath/init3d
                        
            **res_model** : np.array((nx,ny,nz))
                        Starting resistivity model.  Each cell is allocated an
                        integer value that cooresponds to the index value of
                        **reslist**.  **Note** again that the modeling code 
                        assumes that the first row it reads in is the southern
                        most row and the first column it reads in is the 
                        western most column.  Similarly, the first plane it 
                        reads in is the Earth's surface.
                            
                        
                          
        """
        if nodes_east != None:
            self.nodes_east = nodes_east
        if nodes_north != None:
            self.nodes_north = nodes_north
        if nodes_z != None:
            self.nodes_z = nodes_z
        if title != None:
            self.title = title
            
        self.res_list = res_list
        if res_model != None:
            self.res_model = res_model
        
        #--> get path to save initial file to
        if save_path is None:
            self.save_path = os.getcwd()
            self.initial_fn = os.path.join(save_path, "WSInitialModel")
        elif os.path.isdir(save_path) == True:
            self.save_path = save_path
            self.initial_fn = os.path.join(save_path, "WSInitialModel")
        else:
            self.save_path = os.path.dirname(save_path)
            self.initial_fn= os.path.join(save_path)
        
        #check to see what resistivity in input 
        if type(self.res_list) is not list and \
           type(self.res_list) is not np.ndarray:
            self.res_list = [self.res_list]

        #--> write file
        ifid = file(self.initial_fn, 'w')
        ifid.write('# {0}\n'.format(self.title.upper()))
        ifid.write('{0} {1} {2} {3}\n'.format(self.nodes_north.shape[0],
                                              self.nodes_east.shape[0],
                                              self.nodes_z.shape[0],
                                              len(self.res_list)))
    
        #write S --> N node block
        for ii, nnode in enumerate(self.nodes_north):
            ifid.write('{0:>12}'.format('{:.1f}'.format(abs(nnode))))
            if ii != 0 and np.remainder(ii+1, 5) == 0:
                ifid.write('\n')
            elif ii == self.nodes_north.shape[0]-1:
                ifid.write('\n')
        
        #write W --> E node block        
        for jj, enode in enumerate(self.nodes_east):
            ifid.write('{0:>12}'.format('{:.1f}'.format(abs(enode))))
            if jj != 0 and np.remainder(jj+1, 5) == 0:
                ifid.write('\n')
            elif jj == self.nodes_east.shape[0]-1:
                ifid.write('\n')
    
        #write top --> bottom node block
        for kk, zz in enumerate(self.nodes_z):
            ifid.write('{0:>12}'.format('{:.1f}'.format(abs(zz))))
            if kk != 0 and np.remainder(kk+1, 5) == 0:
                ifid.write('\n')
            elif kk == self.nodes_z.shape[0]-1:
                ifid.write('\n')
    
        #write the resistivity list
        for ff in self.res_list:
            ifid.write('{0:.1f} '.format(ff))
        ifid.write('\n')
        
        if self.res_model == None:
            ifid.close()
        else:
            #get similar layers
            l1 = 0
            layers = []
            for zz in range(self.nodes_z.shape[0]-1):
                if (self.res_model[:, :, zz] == self.res_model[:, :, zz+1]).all() == False:
                    layers.append((l1, zz))
                    l1 = zz+1
            #need to add on the bottom layers
            layers.append((l1, self.nodes_z.shape[0]-1))
            
            #write out the layers from resmodel
            for ll in layers:
                ifid.write('{0} {1}\n'.format(ll[0]+1, ll[1]+1))
                for nn in range(self.nodes_north.shape[0]):
                    for ee in range(self.nodes_east.shape[0]):
                        ifid.write('{0:.0f} '.format(self.res_model[nn, ee, ll[0]]))
                    ifid.write('\n')
            ifid.close()
        
        print 'Wrote file to: {0}'.format(self.initial_fn)
        
    def read_initial_file(self, initial_fn):
        """
        read an initial file and return the pertinent information including
        grid positions in coordinates relative to the center point (0,0) and 
        starting model.
    
        Arguments:
        ----------
        
            **initial_fn** : full path to initializing file.
            
        Outputs:
        --------
            
            **nodes_north** : np.array(nx)
                        array of nodes in S --> N direction
            
            **nodes_east** : np.array(ny) 
                        array of nodes in the W --> E direction
                        
            **nodes_z** : np.array(nz)
                        array of nodes in vertical direction positive downwards
            
            **res_model** : dictionary
                        dictionary of the starting model with keys as layers
                        
            **res_list** : list
                        list of resistivity values in the model
            
            **title** : string
                         title string
                           
        """
        self.initial_fn = initial_fn
        ifid = file(self.initial_fn, 'r')    
        ilines = ifid.readlines()
        ifid.close()
        
        self.title = ilines[0].strip()
    
        #get size of dimensions, remembering that x is N-S, y is E-W, z is + down    
        nsize = ilines[1].strip().split()
        n_north = int(nsize[0])
        n_east = int(nsize[1])
        n_z = int(nsize[2])
    
        #initialize empy arrays to put things into
        self.nodes_north = np.zeros(n_north)
        self.nodes_east = np.zeros(n_east)
        self.nodes_z = np.zeros(n_z)
        self.res_model = np.zeros((n_north, n_east, n_z))
        
        #get the grid line locations
        line_index = 2       #line number in file
        count_n = 0  #number of north nodes found
        while count_n < n_north:
            iline = ilines[line_index].strip().split()
            for north_node in iline:
                self.nodes_north[count_n] = float(north_node)
                count_n += 1
            line_index += 1
        
        count_e = 0  #number of east nodes found
        while count_e < n_east:
            iline = ilines[line_index].strip().split()
            for east_node in iline:
                self.nodes_east[count_e] = float(east_node)
                count_e += 1
            line_index += 1
        
        count_z = 0  #number of vertical nodes
        while count_z < n_z:
            iline = ilines[line_index].strip().split()
            for z_node in iline:
                self.nodes_z[count_z] = float(z_node)
                count_z += 1
            line_index += 1
        
        #put the grids into coordinates relative to the center of the grid
        self.grid_north = self.nodes_north.copy()
        self.grid_north[:int(n_north/2)] =\
                        -np.array([self.nodes_north[ii:int(n_north/2)].sum() 
                                   for ii in range(int(n_north/2))])
        self.grid_north[int(n_north/2):] = \
                        np.array([self.nodes_north[int(n_north/2):ii+1].sum() 
                                 for ii in range(int(n_north/2), n_north)])-\
                                 self.nodes_north[int(n_north/2)]
                                
        self.grid_east = self.nodes_east.copy()
        self.grid_east[:int(n_east/2)] = \
                            -np.array([self.nodes_east[ii:int(n_east/2)].sum() 
                                       for ii in range(int(n_east/2))])
        self.grid_east[int(n_east/2):] = \
                            np.array([self.nodes_east[int(n_east/2):ii+1].sum() 
                                     for ii in range(int(n_east/2),n_east)])-\
                                     self.nodes_east[int(n_east/2)]
                                
        self.grid_z = np.array([self.nodes_z[:ii+1].sum() for ii in range(n_z)])
        
        #get the resistivity values
        self.res_list = [float(rr) for rr in ilines[line_index].strip().split()]
        line_index += 1    
        
        #get model
        try:
            iline = ilines[line_index].strip().split()
            
        except IndexError:
            self.res_model[:, :, :] = self.res_list[0]
            return 
            
        if len(iline) == 0 or len(iline) == 1:
            self.res_model[:, :, :] = self.res_list[0]
            return
        else:
            while line_index < len(ilines):
                iline = ilines[line_index].strip().split()
                if len(iline) == 2:
                    l1 = int(iline[0])-1
                    l2 = int(iline[1])
                    line_index += 1
                    count_n = 0
                elif len(iline) == 0:
                    break
                else:
                    count_e = 0
                    while count_e < n_east:
                        self.res_model[count_e, count_n, l1:l2] =\
                                                            int(iline[count_e])
                        count_e += 1
                    count_n += 1
                    line_index += 1

#==============================================================================
# model class                    
#==============================================================================
class WSModel(object):
    """
    included tools for reading a model and plotting a model.
    
    """
    
    def __init__(self, model_fn=None):
        self.model_fn = model_fn
        self.iteration_number = None
        self.rms = None
        self.lagrange = None
        self.res_model = None
        self.res_list = None
        
        self.nodes_north = None
        self.nodes_east = None
        self.nodes_z = None
        
        self.grid_north = None
        self.grid_east = None
        self.grid_z = None
        
    def read_model_file(self):
        """
        read in a model file as x-north, y-east, z-positive down
        """            
        
        mfid = file(self.model_fn, 'r')
        mlines = mfid.readlines()
        mfid.close()
    
        #get info at the beggining of file
        info = mlines[0].strip().split()
        self.iteration_number = int(info[1])
        self.rms = float(info[3])
        self.lagrange = float(info[5])
        
        #get lengths of things
        n_north, n_east, n_z, n_res = np.array(mlines[1].strip().split(),
                                               dtype=np.int)
        
        #make empty arrays to put stuff into
        self.nodes_north = np.zeros(n_north)
        self.nodes_east = np.zeros(n_east)
        self.nodes_z = np.zeros(n_z)
        self.res_model = np.zeros((n_north, n_east, n_z))
        
        #get the grid line locations
        line_index = 2       #line number in file
        count_n = 0  #number of north nodes found
        while count_n < n_north:
            mline = mlines[line_index].strip().split()
            for north_node in mline:
                self.nodes_north[count_n] = float(north_node)
                count_n += 1
            line_index += 1
        
        count_e = 0  #number of east nodes found
        while count_e < n_east:
            mline = mlines[line_index].strip().split()
            for east_node in mline:
                self.nodes_east[count_e] = float(east_node)
                count_e += 1
            line_index += 1
        
        count_z = 0  #number of vertical nodes
        while count_z < n_z:
            mline = mlines[line_index].strip().split()
            for z_node in mline:
                self.nodes_z[count_z] = float(z_node)
                count_z += 1
            line_index += 1
            
        #put the grids into coordinates relative to the center of the grid
        self.grid_north = self.nodes_north.copy()
        self.grid_north[:int(n_north/2)] =\
                        -np.array([self.nodes_north[ii:int(n_north/2)].sum() 
                                   for ii in range(int(n_north/2))])
        self.grid_north[int(n_north/2):] = \
                        np.array([self.nodes_north[int(n_north/2):ii+1].sum() 
                                 for ii in range(int(n_north/2), n_north)])-\
                                 self.nodes_north[int(n_north/2)]
                                
        self.grid_east = self.nodes_east.copy()
        self.grid_east[:int(n_east/2)] = \
                            -np.array([self.nodes_east[ii:int(n_east/2)].sum() 
                                       for ii in range(int(n_east/2))])
        self.grid_east[int(n_east/2):] = \
                            np.array([self.nodes_east[int(n_east/2):ii+1].sum() 
                                     for ii in range(int(n_east/2),n_east)])-\
                                     self.nodes_east[int(n_east/2)]
                                
        self.grid_z = np.array([self.nodes_z[:ii+1].sum() for ii in range(n_z)])
    
        #--> get resistivity values
        for kk in range(n_z):
            for jj in range(n_east):
                for ii in range(n_north):
                    self.res_model[(n_north-1)-ii, jj, kk] = \
                                             float(mlines[line_index].strip())
                    line_index += 1


#==============================================================================
# Manipulate the model
#==============================================================================
class WS3DModelManipulator(object):
    """
    will plot a model from wsinv3d or init file so the user can manipulate the 
    resistivity values relatively easily.  At the moment only plotted
    in map view.
    
    
    """

    def __init__(self, model_fn=None, initial_fn=None, data_fn=None,
                 res_list=None, mapscale='km', plot_yn='y', xlimits=None, 
                 ylimits=None, cbdict={}):
        
        self.model_fn = model_fn
        self.initial_fn = initial_fn
        self.data_fn = data_fn
        self.new_initial_fn = None
        
        if self.model_fn is not None:
            self.save_path = os.path.dirname(self.model_fn)
        elif self.initial_fn is not None:
            self.save_path = os.path.dirname(self.initial_fn)
        elif self.data_fn is not None:
            self.save_path = os.path.dirname(self.data_fn)
        else:
            self.save_path = None
            
        #grid nodes
        self.nodes_east = None
        self.nodes_north = None
        self.nodes_z = None
        
        #grid locations
        self.grid_east = None
        self.grid_north = None
        self.grid_z = None
        
        #resistivity model
        self.res_model_int = None #model in ints
        self.res_model = None     #model in floats
        
        #station locations in relative coordinates read from data file
        self.station_east = None
        self.station_north = None
        
        #--> set map scale
        self.mapscale = mapscale
        
        self.m_width = 100
        self.m_height = 100
        
        #--> scale the map coordinates
        if self.mapscale=='km':
            self.dscale = 1000.
        if self.mapscale=='m':
            self.dscale = 1.
            
        #figure attributes
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.cb = None
        self.east_line_xlist = None
        self.east_line_ylist = None
        self.north_line_xlist = None
        self.north_line_ylist = None
        
        #make a default resistivity list to change values
        self.res_dict = None
        if res_list is None:
            self.set_res_list(np.array([.3, 1, 10, 50, 100, 500, 1000, 5000],
                                      dtype=np.float))
        
        else:
            self.set_res_list(res_list) 
        
        
        #read in model or initial file
        self.read_file()

        #set initial resistivity value
        self.res_value = self.res_list[0]
        
        #--> set map limits
        self.xlimits = xlimits
        self.ylimits = ylimits
        
        self.cb_dict = cbdict

        self.font_size = 7
        self.dpi = 300
        self.fignum = 1
        self.figsize = [6,6]
        self.cmap = cm.jet_r
        self.depth_index = 0
        
        self.fdict = {'size':self.font_size+2, 'weight':'bold'}
    
        self.cblabeldict = {-5:'$10^{-5}$',
                            -4:'$10^{-4}$',
                            -3:'$10^{-3}$',
                            -2:'$10^{-2}$',
                            -1:'$10^{-1}$',
                             0:'$10^{0}$',
                             1:'$10^{1}$',
                             2:'$10^{2}$',
                             3:'$10^{3}$',
                             4:'$10^{4}$',
                             5:'$10^{5}$',
                             6:'$10^{6}$',
                             7:'$10^{7}$',
                             8:'$10^{8}$'}
        

        
        #plot on initialization
        self.plot_yn = plot_yn
        if self.plot_yn=='y':
            self.plot()
            
    def set_res_list(self, res_list):
        """
        on setting res_list also set the res_dict to correspond
        """
        self.res_list = res_list
        #make a dictionary of values to write to file.
        self.res_dict = dict([(res, ii) 
                              for ii, res in enumerate(self.res_list,1)])
        if self.fig is not None:
            plt.close()
            self.plot()
        
    
    #---read files-------------------------------------------------------------    
    def read_file(self):
        """
        reads in initial file or model file and set attributes:
            -resmodel
            -northrid
            -eastrid
            -zgrid
            -res_list if initial file
            
        """
        att_names = ['nodes_north', 'nodes_east', 'nodes_z', 'grid_east', 
                     'grid_north', 'grid_z', 'res_model', 'res_list']
                 
        #--> read model file
        if self.model_fn is not None and self.initial_fn is None:
            
            wsmodel = WSModel(self.model_fn)
            wsmodel.read_model_file()
            
            for name in att_names:
                if hasattr(wsmodel, name):
                    value = getattr(wsmodel, name)
                    setattr(self, name, value)
            
            self.convert_res_to_model()
         
        #--> read initial file
        elif self.initial_fn is not None and self.model_fn is None:
            wsmesh = WSMesh()
            wsmesh.read_initial_file(self.initial_fn)
            for name in att_names:
                if hasattr(wsmesh, name):
                    value = getattr(wsmesh, name)
                    setattr(self, name, value)
                    
            self.res_model_int = wsmesh.res_model
            if len(wsmesh.res_list) == 1:
                self.set_res_list([.3, 1, 10, 100, 1000])
            else:
                self.set_res_list(wsmesh.res_list)
            
            #need to convert index values to resistivity values
            rdict = dict([(ii,res) for ii,res in enumerate(self.res_list,1)])
            
            for ii in range(len(self.res_list)):
                self.res_model[np.where(self.res_model_int==ii+1)] = rdict[ii+1]
                
        elif self.initial_fn is None and self.model_fn is None:
            print 'Need to input either an initial file or model file to plot'
        else:
            print 'Input just initial file or model file not both.'
         
        #--> read in data file if given
        if self.data_fn is not None:
            wsdata = WSData()
            wsdata.read_data_file(self.data_fn)
            
            #get station locations
            self.station_east = wsdata.data['east']
            self.station_north = wsdata.data['north']
            
        #get cell block sizes
        self.m_height = np.median(self.nodes_north[5:-5])/self.dscale
        self.m_width = np.median(self.nodes_east[5:-5])/self.dscale
            
        #make a copy of original in case there are unwanted changes
        self.res_copy = self.res_model.copy()
            
            
            
    #---plot model-------------------------------------------------------------    
    def plot(self):
        """
        plots the model with:
            -a radio dial for depth slice 
            -radio dial for resistivity value
            
        """
        
        self.cmin = np.floor(np.log10(min(self.res_list)))
        self.cmax = np.ceil(np.log10(max(self.res_list)))
        
        #-->Plot properties
        plt.rcParams['font.size'] = self.font_size
        
        #need to add an extra row and column to east and north to make sure 
        #all is plotted see pcolor for details.
        plot_east = np.append(self.grid_east, self.grid_east[-1]*1.25)/self.dscale
        plot_north = np.append(self.grid_north, self.grid_north[-1]*1.25)/self.dscale
        
        #make a mesh grid for plotting
        #the 'ij' makes sure the resulting grid is in east, north
        self.mesh_east, self.mesh_north = np.meshgrid(plot_east, 
                                                      plot_north,
                                                      indexing='ij')
        
        self.fig = plt.figure(self.fignum, figsize=self.figsize, dpi=self.dpi)
        plt.clf()
        self.ax1 = self.fig.add_subplot(1, 1, 1, aspect='equal')
        
        plot_res = np.log10(self.res_model[:,:,self.depth_index].T)
        
        self.mesh_plot = self.ax1.pcolormesh(self.mesh_east,
                                             self.mesh_north, 
                                             plot_res,
                                             cmap=self.cmap,
                                             vmin=self.cmin,
                                             vmax=self.cmax)
                                             
        #on plus or minus change depth slice
        self.cid_depth = \
                    self.mesh_plot.figure.canvas.mpl_connect('key_press_event',
                                                        self._on_key_callback)
                                    
                       
        #plot the stations
        if self.station_east is not None:
            for ee, nn in zip(self.station_east, self.station_north):
                self.ax1.text(ee/self.dscale, nn/self.dscale,
                              '*',
                              verticalalignment='center',
                              horizontalalignment='center',
                              fontdict={'size':self.font_size-2,
                                        'weight':'bold'})

        #set axis properties
        if self.xlimits is not None:
            self.ax1.set_xlim(self.xlimits)
        else:
            self.ax1.set_xlim(xmin=self.grid_east.min()/self.dscale, 
                              xmax=self.grid_east.max()/self.dscale)
        
        if self.ylimits is not None:
            self.ax1.set_ylim(self.ylimits)
        else:
            self.ax1.set_ylim(ymin=self.grid_north.min()/self.dscale,
                              ymax=self.grid_north.max()/self.dscale)
            
        #self.ax1.xaxis.set_minor_locator(MultipleLocator(100*1./dscale))
        #self.ax1.yaxis.set_minor_locator(MultipleLocator(100*1./dscale))
        
        self.ax1.set_ylabel('Northing ('+self.mapscale+')',
                            fontdict=self.fdict)
        self.ax1.set_xlabel('Easting ('+self.mapscale+')',
                            fontdict=self.fdict)
        
        depth_title = self.grid_z[self.depth_index]/self.dscale
                                                        
        self.ax1.set_title('Depth = {:.3f} '.format(depth_title)+\
                           '('+self.mapscale+')',
                           fontdict=self.fdict)
        
        #plot the grid if desired  
        self.east_line_xlist = []
        self.east_line_ylist = []            
        for xx in self.grid_east:
            self.east_line_xlist.extend([xx/self.dscale, xx/self.dscale])
            self.east_line_xlist.append(None)
            self.east_line_ylist.extend([self.grid_north.min()/self.dscale, 
                                         self.grid_north.max()/self.dscale])
            self.east_line_ylist.append(None)
        self.ax1.plot(self.east_line_xlist,
                      self.east_line_ylist,
                       lw=.25,
                       color='k')

        self.north_line_xlist = []
        self.north_line_ylist = [] 
        for yy in self.grid_north:
            self.north_line_xlist.extend([self.grid_east.min()/self.dscale,
                                          self.grid_east.max()/self.dscale])
            self.north_line_xlist.append(None)
            self.north_line_ylist.extend([yy/self.dscale, yy/self.dscale])
            self.north_line_ylist.append(None)
        self.ax1.plot(self.north_line_xlist,
                      self.north_line_ylist,
                      lw=.25,
                      color='k')
        
        #plot the colorbar
        self.ax2 = mcb.make_axes(self.ax1, orientation='vertical', shrink=.5)
        seg_cmap = cmap_discretize(self.cmap, len(self.res_list))
        self.cb = mcb.ColorbarBase(self.ax2[0],cmap=seg_cmap,
                                   norm=colors.Normalize(vmin=self.cmin,
                                                         vmax=self.cmax))
                                                         
                            
        self.cb.set_label('Resistivity ($\Omega \cdot$m)',
                          fontdict={'size':self.font_size})
        self.cb.set_ticks(np.arange(self.cmin, self.cmax+1))
        self.cb.set_ticklabels([self.cblabeldict[cc] 
                                for cc in np.arange(self.cmin, self.cmax+1)])
                            
        #make a resistivity radio button
        resrb = self.fig.add_axes([.85,.1,.1,.15])
        reslabels = ['{0:.4g}'.format(res) for res in self.res_list]
        self.radio_res = widgets.RadioButtons(resrb, reslabels, 
                                        active=self.res_dict[self.res_value])
        
        #make a rectangular selector
        self.rect_selector = widgets.RectangleSelector(self.ax1, 
                                                       self.rect_onselect,
                                                       drawtype='box',
                                                       useblit=True)

        
        plt.show()
        
        #needs to go after show()
        self.radio_res.on_clicked(self.set_res_value)


    def redraw_plot(self):
        """
        redraws the plot
        """
        
        current_xlimits = self.ax1.get_xlim()
        current_ylimits = self.ax1.get_ylim()
        
        self.ax1.cla()
        
        plot_res = np.log10(self.res_model[:,:,self.depth_index].T)
        
        self.mesh_plot = self.ax1.pcolormesh(self.mesh_east,
                                             self.mesh_north, 
                                             plot_res,
                                             cmap=self.cmap,
                                             vmin=self.cmin,
                                             vmax=self.cmax)
                                             
         #plot the stations
        if self.station_east is not None:
            for ee,nn in zip(self.station_east, self.station_north):
                self.ax1.text(ee/self.dscale, nn/self.dscale,
                              '*',
                              verticalalignment='center',
                              horizontalalignment='center',
                              fontdict={'size':self.font_size-2,
                                        'weight':'bold'})

        #set axis properties
        if self.xlimits is not None:
            self.ax1.set_xlim(self.xlimits)
        else:
            self.ax1.set_xlim(current_xlimits)
        
        if self.ylimits is not None:
            self.ax1.set_ylim(self.ylimits)
        else:
            self.ax1.set_ylim(current_ylimits)
        
        self.ax1.set_ylabel('Northing ('+self.mapscale+')',
                            fontdict=self.fdict)
        self.ax1.set_xlabel('Easting ('+self.mapscale+')',
                            fontdict=self.fdict)
        
        depth_title = self.grid_z[self.depth_index]/self.dscale
                                                        
        self.ax1.set_title('Depth = {:.3f} '.format(depth_title)+\
                           '('+self.mapscale+')',
                           fontdict=self.fdict)
                     
        #plot finite element mesh
        self.ax1.plot(self.east_line_xlist,
                      self.east_line_ylist,
                      lw=.25,
                      color='k')

        
        self.ax1.plot(self.north_line_xlist,
                      self.north_line_ylist,
                      lw=.25,
                      color='k')
        
        #be sure to redraw the canvas                  
        self.fig.canvas.draw()
        
    def set_res_value(self, label):
        self.res_value = float(label)
        print 'set resistivity to ', label
        print self.res_value
        
        
    def _on_key_callback(self,event):
        """
        on pressing a key do something
        
        """
        
        self.event_change_depth = event
        
        #go down a layer on push of +/= keys
        if self.event_change_depth.key == '=':
            self.depth_index += 1
            
            if self.depth_index>len(self.grid_z)-1:
                self.depth_index = len(self.grid_z)-1
                print 'already at deepest depth'
                
            print 'Plotting Depth {0:.3f}'.format(self.grid_z[self.depth_index]/\
                    self.dscale)+'('+self.mapscale+')'
            
            self.redraw_plot()
        #go up a layer on push of - key
        elif self.event_change_depth.key == '-':
            self.depth_index -= 1
            
            if self.depth_index < 0:
                self.depth_index = 0
                
            print 'Plotting Depth {0:.3f} '.format(self.grid_z[self.depth_index]/\
                    self.dscale)+'('+self.mapscale+')'
            
            self.redraw_plot()
        
        #exit plot on press of q
        elif self.event_change_depth.key == 'q':
            self.event_change_depth.canvas.mpl_disconnect(self.cid_depth)
            plt.close(self.event_change_depth.canvas.figure)
            self.rewrite_initial_file()
            
        #copy the layer above
        elif self.event_change_depth.key == 'a':
            try:
                if self.depth_index == 0:
                    print 'No layers above'
                else:
                    self.res_model[:, :, self.depth_index] = \
                                       self.res_model[:, :, self.depth_index-1]
            except IndexError:
                print 'No layers above'
                
            self.redraw_plot()
        
        #copy the layer below
        elif self.event_change_depth.key == 'b':
            try:
                self.res_model[:, :, self.depth_index] = \
                                    self.res_model[:, :, self.depth_index+1]
            except IndexError:
                print 'No more layers below'
                
            self.redraw_plot() 
            
        #undo
        elif self.event_change_depth.key == 'u':
            if type(self.xchange) is int and type(self.ychange) is int:
                self.res_model[self.ychange, self.xchange, self.depth_index] =\
                self.res_copy[self.ychange, self.xchange, self.depth_index]
            else:
                for xx in self.xchange:
                    for yy in self.ychange:
                        self.res_model[yy, xx, self.depth_index] = \
                        self.res_copy[yy, xx, self.depth_index]
            
            self.redraw_plot()
            
    def change_model_res(self, xchange, ychange):
        """
        change resistivity values of resistivity model
        
        """
        if type(xchange) is int and type(ychange) is int:
            self.res_model[ychange, xchange, self.depth_index] = self.res_value
        else:
            for xx in xchange:
                for yy in ychange:
                    self.res_model[yy, xx, self.depth_index] = self.res_value
        
        self.redraw_plot()            
           
    def rect_onselect(self, eclick, erelease):
        """
        on selecting a rectangle change the colors to the resistivity values
        """
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        self.xchange = self._get_east_index(x1, x2)
        self.ychange = self._get_north_index(y1, y2)
        
        #reset values of resistivity
        self.change_model_res(self.xchange, self.ychange)
        
        
    def _get_east_index(self, x1, x2):
        """
        get the index value of the points to be changed
        
        """
        if x1 < x2:
            xchange = np.where((self.grid_east/self.dscale >= x1) & \
                               (self.grid_east/self.dscale <= x2))[0]
            if len(xchange) == 0:
                xchange = np.where(self.grid_east/self.dscale >= x1)[0][0]-1
                return [xchange]
                
        if x1 > x2:
            xchange = np.where((self.grid_east/self.dscale <= x1) & \
                               (self.grid_east/self.dscale >= x2))[0]
            if len(xchange) == 0:
                xchange = np.where(self.grid_east/self.dscale >= x2)[0][0]-1
                return [xchange]

            
        #check the edges to see if the selection should include the square
        xchange = np.append(xchange, xchange[0]-1)
        xchange.sort()

        return xchange
                
    def _get_north_index(self, y1, y2):
        """
        get the index value of the points to be changed in north direction
        
        need to flip the index because the plot is flipped
        
        """
        
        if y1 < y2:
            ychange = np.where((self.grid_north/self.dscale > y1) & \
                               (self.grid_north/self.dscale < y2))[0]
            if len(ychange) == 0:
                ychange = np.where(self.grid_north/self.dscale >= y1)[0][0]-1
                return [ychange]
                
        elif y1 > y2:
            ychange = np.where((self.grid_north/self.dscale < y1) & \
                               (self.grid_north/self.dscale > y2))[0]
            if len(ychange) == 0:
                ychange = np.where(self.grid_north/self.dscale >= y2)[0][0]-1
                return [ychange]
        
        ychange -= 1
        ychange = np.append(ychange, ychange[-1]+1)

        return ychange
        
            
    def convert_model_to_int(self):
        """
        convert the resistivity model that is in ohm-m to integer values
        corresponding to res_list
        
        """
 
        self.res_model_int = np.ones_like(self.res_model)
        
        for key in self.res_dict.keys():
            self.res_model_int[np.where(self.res_model==key)] = \
                                                        self.res_dict[key]
            
    def convert_res_to_model(self):
        """
        converts an output model into an array of segmented valued according
        to res_list.        
        
        """
        
        #make values in model resistivity array a value in res_list
        resm = np.ones_like(self.res_model)
        resm[np.where(self.res_model<self.res_list[0])] = \
                                            self.res_dict[self.res_list[0]]
        resm[np.where(self.res_model)>self.res_list[-1]] = \
                                            self.res_dict[self.res_list[-1]]
        
        for zz in range(self.res_model.shape[2]):
            for yy in range(self.res_model.shape[1]):
                for xx in range(self.res_model.shape[0]):
                    for rr in range(len(self.res_list)-1):
                        if self.res[xx, yy, zz] >= self.res_list[rr] and \
                           self.res[xx, yy, zz] <= self.res_list[rr+1]:
                            resm[xx, yy, zz] = self.res_dict[self.res_list[rr]]
                            break
                        elif self.res[xx, yy, zz] <= self.res_list[0]:
                            resm[xx, yy, zz] = self.res_dict[self.res_list[0]]
                            break
                        elif self.res[xx, yy, zz] >= self.res_list[-1]:
                            resm[xx, yy, zz] = self.res_dict[self.res_list[-1]]
                            break
    
        self.res_model = resm
            
        
    def rewrite_initial_file(self, save_path=None):
        """
        write an initial file for wsinv3d from the model created.
        """
        
        self.convert_model_to_int()
        
        #need to flip the resistivity model so that the first index is the 
        #northern most block in N-S
        self.res_model = self.res_model[::-1, :, :]
        
        if save_path is not None:
            self.save_path = save_path
        
        self.new_initial_fn = os.path.join(self.save_path, 'WSInitialFile_RW')
        wsmesh = WSMesh()
        
        #pass attribute to wsmesh
        att_names = ['nodes_north', 'nodes_east', 'nodes_z', 'grid_east', 
                     'grid_north', 'grid_z', 'res_model', 'res_list']
        for name in att_names:
                if hasattr(self, name):
                    value = getattr(self, name)
                    setattr(wsmesh, name, value)            
        
        wsmesh.write_initial_file(save_path=self.new_initial_fn)
                                              
            
def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.
      
         cmap: colormap instance, eg. cm.jet. 
         N: number of colors.
     
     Example
         x = resize(arange(100), (5,100))
         djet = cmap_discretize(cm.jet, 5)
         imshow(x, cmap=djet)
    """

    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki,key in enumerate(('red','green','blue')):
        cdict[key] = [(indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki])
                       for i in xrange(N+1)]
    # Return colormap object.
    return colors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)

class WSModel(object):
    """
    included tools for reading a model and plotting a model.
    
    """
    
    def __init__(self, model_fn=None):
        self.model_fn = model_fn
        self.iteration_number = None
        self.rms = None
        self.lagrange = None
        self.res_model = None
        self.res_list = None
        
        self.nodes_north = None
        self.nodes_east = None
        self.nodes_z = None
        
        self.grid_north = None
        self.grid_east = None
        self.grid_z = None
        
    def read_model_file(self):
        """
        read in a model file as x-north, y-east, z-positive down
        """            
        
        mfid = file(self.model_fn, 'r')
        mlines = mfid.readlines()
        mfid.close()
    
        #get info at the beggining of file
        info = mlines[0].strip().split()
        self.iteration_number = int(info[1])
        self.rms = float(info[3])
        self.lagrange = float(info[5])
        
        #get lengths of things
        n_north, n_east, n_z, n_res = np.array(mlines[1].strip().split(),
                                               dtype=np.int)
        
        #make empty arrays to put stuff into
        self.nodes_north = np.zeros(n_north)
        self.nodes_east = np.zeros(n_east)
        self.nodes_z = np.zeros(n_z)
        self.res_model = np.zeros((n_north, n_east, n_z))
        
        #get the grid line locations
        line_index = 2       #line number in file
        count_n = 0  #number of north nodes found
        while count_n < n_north:
            mline = mlines[line_index].strip().split()
            for north_node in mline:
                self.nodes_north[count_n] = float(north_node)
                count_n += 1
            line_index += 1
        
        count_e = 0  #number of east nodes found
        while count_e < n_east:
            mline = mlines[line_index].strip().split()
            for east_node in mline:
                self.nodes_east[count_e] = float(east_node)
                count_e += 1
            line_index += 1
        
        count_z = 0  #number of vertical nodes
        while count_z < n_z:
            mline = mlines[line_index].strip().split()
            for z_node in mline:
                self.nodes_z[count_z] = float(z_node)
                count_z += 1
            line_index += 1
            
        #put the grids into coordinates relative to the center of the grid
        self.grid_north = self.nodes_north.copy()
        self.grid_north[:int(n_north/2)] =\
                        -np.array([self.nodes_north[ii:int(n_north/2)].sum() 
                                   for ii in range(int(n_north/2))])
        self.grid_north[int(n_north/2):] = \
                        np.array([self.nodes_north[int(n_north/2):ii+1].sum() 
                                 for ii in range(int(n_north/2), n_north)])-\
                                 self.nodes_north[int(n_north/2)]
                                
        self.grid_east = self.nodes_east.copy()
        self.grid_east[:int(n_east/2)] = \
                            -np.array([self.nodes_east[ii:int(n_east/2)].sum() 
                                       for ii in range(int(n_east/2))])
        self.grid_east[int(n_east/2):] = \
                            np.array([self.nodes_east[int(n_east/2):ii+1].sum() 
                                     for ii in range(int(n_east/2),n_east)])-\
                                     self.nodes_east[int(n_east/2)]
                                
        self.grid_z = np.array([self.nodes_z[:ii+1].sum() for ii in range(n_z)])
    
        #--> get resistivity values
        for kk in range(n_z):
            for jj in range(n_east):
                for ii in range(n_north):
                    self.res_model[(n_north-1)-ii, jj, kk] = \
                                             float(mlines[line_index].strip())
                    line_index += 1

