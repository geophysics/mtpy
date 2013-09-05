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
import mtpy.utils.exceptions as mtex
import mtpy.analysis.pt as mtpt
import mtpy.imaging.mtcolors as mtcl

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
        self.period_list = kwargs.pop('period_list', None)
        self.edi_list = kwargs.pop('edi_list', None)
        self.station_locations = kwargs.pop('station_locations', None)
        
        self.wl_site_fn = kwargs.pop('wl_site_fn', None)
        self.wl_out_fn = kwargs.pop('wl_out_fn', None)
        self.data_fn = kwargs.pop('data_fn', None)
        self.station_fn = kwargs.pop('station_fn', None)
        
        self.data = None
        
        self._data_keys = ['station', 'east', 'north', 'z_data', 'z_data_err']

        
    def write_data_file(self):
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

        #get units correctly
        if self.units == 'mv':
            zconv = 1./796.
            
        #define some lengths
        n_stations = len(self.edi_list)
        n_periods = len(self.period_list)
        
        #make a structured array to keep things in for convenience
        z_shape = (n_periods, 4)
        data_dtype = [('station', '|S10'),
                      ('east', np.float),
                      ('north', np.float),
                      ('z_data', (np.complex, z_shape)),
                      ('z_data_err', (np.complex, z_shape)),
                      ('z_err_map', (np.complex, z_shape))]
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
            stations = WSStation(self.station_fn)
            stations.read_station_file()
            self.data['station'] = stations.names
            self.data['east'] = stations.east
            self.data['north'] = stations.north
        
        #if the user made a grid in python or some other fashion
        if self.station_locations != None:
            try:
                for dd, sd in enumerate(self.station_locations):
                    self.data['east'][dd] = sd['east_c']
                    self.data['north'][dd] = sd['north_c']
                    self.data['station'][dd] = sd['station']
                    
                    stations = WSStation()
                    stations.station_fn = os.path.join(self.save_path, 
                                                    'WS_Station_locations.txt')
                    stations.east = self.data['east']
                    stations.north = self.data['north']
                    stations.names = self.data['station']
                    stations.write_station_file()
                    
            except (KeyError, ValueError): 
                self.data['east'] = self.station_locations[:, 0]
                self.data['north']= self.station_locations[:, 1]
        
        #--------find frequencies----------------------------------------------
        for ss, edi in enumerate(self.edi_list):
            if not os.path.isfile(edi):
                raise IOError('Could not find '+edi)

            z1 = mtedi.Edi()
            z1.readfile(edi)
            print '{0}{1}{0}'.format('-'*20, z1.station) 
            for ff, f1 in enumerate(self.period_list):
                for kk,f2 in enumerate(z1.period):
                    if f2 >= (1-self.ptol)*f1 and f2 <= (1+self.ptol)*f1:
                        self.data[ss]['z_data'][ff, :] = \
                                                   zconv*z1.Z.z[kk].reshape(4,)
                        self.data[ss]['z_data_err'][ff, :] = \
                                    zconv*z1.Z.z[kk].reshape(4,)*self.z_err
                                    
                        self.data[ss]['z_err_map'][ff, :] = self.z_err_map
                        
                        print '   Matched {0:.6g} to {1:.6g}'.format(f1, f2)
                        break
        
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
                      ('z_err_map', (np.complex, z_shape))]
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
            stations = WSStation(self.station_fn)
            stations.read_station_file()
            self.data['station'] = stations.names
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
                if dkey == 'z_err_map':
                    zline = np.array(dl.strip().split(), dtype=np.float)
                    self.data[st][dkey][per-1,:] = np.array([zline[0]-1j*zline[1],
                                                        zline[2]-1j*zline[3],
                                                        zline[4]-1j*zline[5],
                                                        zline[6]-1j*zline[7]])
                else:
                    zline = np.array(dl.strip().split(), dtype=np.float)*zconv
                    self.data[st][dkey][per-1,:] = np.array([zline[0]-1j*zline[1],
                                                        zline[2]-1j*zline[3],
                                                        zline[4]-1j*zline[5],
                                                        zline[6]-1j*zline[7]])
                st += 1

#==============================================================================
# stations
#==============================================================================
class WSStation(object):
    """
    read and write a station file 
    """
    
    def __init__(self, station_fn=None, **kwargs):
        self.station_fn = station_fn
        self.east = kwargs.pop('east', None)
        self.north = kwargs.pop('north', None)
        self.elev = kwargs.pop('elev', None)
        self.names = kwargs.pop('names', None)
        self.save_path = kwargs.pop('save_path', None)
        
    def write_station_file(self, east=None, north=None, station_list=None, 
                           save_path=None):
        """
        write a station file to go with the data file.
        
        the locations are on a relative grid where (0, 0, 0) is the 
        center of the grid.  Also, the stations are assumed to be in the center
        of the cell.
        
        """
        if east is not None:
            self.east = east
        if north is not None:
            self.north = north
        if station_list is not None:
            self.names = station_list
        if save_path is not None:
            self.save_path = save_path
            if os.path.isdir(save_path):
                self.station_fn = os.path.join(save_path, 
                                               'WS_Station_locations.txt')
            else:
                self.station_fn = save_path
        else:
            self.save_path = os.getcwd()
            self.station_fn = os.path.join(save_path, 
                                           'WS_Station_locations.txt')
        
        sfid = file(self.station_fn, 'w')
        sfid.write('{0:<14}{1:^14}{2:^14}\n'.format('station', 'east', 
                                                    'north'))
        for ee, nn, ss in zip(self.east, self.north, self.names):
            ee = '{0:+.4e}'.format(ee)
            nn = '{0:+.4e}'.format(nn)
            sfid.write('{0:<14}{1:^14}{2:^14}\n'.format(ss, ee, nn))
        sfid.close()
        
        print 'Wrote station locations to {0}'.format(self.station_fn)
        
    def read_station_file(self, station_fn=None):
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
        if station_fn is not None:
            self.station_fn = station_fn
            
        self.save_path = os.path.dirname(self.station_fn)
        
        station_locations = np.loadtxt(self.station_fn, skiprows=1, 
                                       dtype=[('station', '|S10'),
                                              ('east_c', np.float),
                                              ('north_c', np.float)])
                                              
        self.east = station_locations['east_c']
        self.north = station_locations['north_c']
        self.names = station_locations['station']
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
        
        #padding cells on either side
        self.pad_east = kwargs.pop('pad_east', 5)
        self.pad_north = kwargs.pop('pad_north', 5)
        self.pad_z = kwargs.pop('pad_z', 5)
        
        #root of padding cells
        self.pad_root_east = kwargs.pop('pad_root_east', 5)
        self.pad_root_north = kwargs.pop('pad_root_north', 5)
        
        self.z1_layer = kwargs.pop('z1_layer', 10)
        self.z_target_depth = kwargs.pop('z_target_depth', 50000)
        self.z_bottom = kwargs.pop('z_bottom', 300000)
        
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
        self.station_fn = None
        self.save_path = kwargs.pop('save_path', None)
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
        log_z = np.logspace(np.log10(self.z1_layer), 
                            np.log10(self.z_target_depth-np.logspace(np.log10(self.z1_layer), 
                            np.log10(self.z_target_depth), 
                            num=self.n_layers)[-2]), 
                            num=self.n_layers-self.pad_z)
        ztarget = np.array([zz-zz%10**np.floor(np.log10(zz)) for zz in 
                           log_z])
        log_zpad = np.logspace(np.log10(self.z_target_depth), 
                            np.log10(self.z_bottom-np.logspace(np.log10(self.z_target_depth), 
                            np.log10(self.z_bottom), 
                            num=self.pad_z)[-2]), 
                            num=self.pad_z)
        zpadding = np.array([zz-zz%10**np.floor(np.log10(zz)) for zz in 
                               log_zpad])
#        #cells down to number of z-layers
#        zgrid1 = self.first_layer_thickness*\
#                 self.pad_root_z**np.round(np.arange(0,self.pad_pow_z[0],
#                         self.pad_pow_z[0]/(self.n_layers-float(self.pad_z))))
#                         
#        #pad bottom of grid
#        zgrid2 = self.first_layer_thickness*\
#                 self.pad_root_z**np.round(np.arange(self.pad_pow_z[0],
#                                                     self.pad_pow_z[1],
#                             (self.pad_pow_z[1]-self.pad_pow_z[0]/self.pad_z)))
        
        z_nodes = np.append(ztarget, zpadding)
        z_grid = np.array([z_nodes[:ii+1].sum() for ii in range(z_nodes.shape[0])])
        
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
        self.nodes_z = z_nodes        
        self.grid_east = east_grid
        self.grid_north = north_grid
        self.grid_z = z_grid
        
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
        print '       z  = {0} (without 7 air layers)'.format(z_grid.shape[0])
        print '   Extensions: '
        print '      e-w = {0:.1f} (m)'.format(east_nodes.__abs__().sum())
        print '      n-s = {0:.1f} (m)'.format(north_nodes.__abs__().sum())
        print '      0-z = {0:.1f} (m)'.format(self.nodes_z.__abs__().sum())
        print '-'*15
        
        #write a station location file for later
        stations = WSStation()
        stations.write_station_file(east=self.station_locations['east_c'],
                                    north=self.station_locations['north_c'],
                                    station_list=self.station_locations['station'],
                                    save_path=self.save_path)
        self.station_fn = stations.station_fn
        

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
        line_width = kwargs.pop('line_width', .5)
        
        plt.rcParams['figure.subplot.hspace'] = .3
        plt.rcParams['figure.subplot.wspace'] = .3
        plt.rcParams['figure.subplot.left'] = .08
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
                
        #plot the grid if desired
        east_line_xlist = []
        east_line_ylist = []            
        for xx in self.grid_east:
            east_line_xlist.extend([xx, xx])
            east_line_xlist.append(None)
            east_line_ylist.extend([self.grid_north.min(), 
                                    self.grid_north.max()])
            east_line_ylist.append(None)
        ax1.plot(east_line_xlist,
                      east_line_ylist,
                      lw=line_width,
                      color=line_color)

        north_line_xlist = []
        north_line_ylist = [] 
        for yy in self.grid_north:
            north_line_xlist.extend([self.grid_east.min(),
                                     self.grid_east.max()])
            north_line_xlist.append(None)
            north_line_ylist.extend([yy, yy])
            north_line_ylist.append(None)
        ax1.plot(north_line_xlist,
                      north_line_ylist,
                      lw=line_width,
                      color=line_color)
        
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
        

        #plot the grid if desired
        east_line_xlist = []
        east_line_ylist = []            
        for xx in self.grid_east:
            east_line_xlist.extend([xx, xx])
            east_line_xlist.append(None)
            east_line_ylist.extend([0, 
                                    self.grid_z.max()])
            east_line_ylist.append(None)
        ax2.plot(east_line_xlist,
                 east_line_ylist,
                 lw=line_width,
                 color=line_color)

        z_line_xlist = []
        z_line_ylist = [] 
        for zz in self.grid_z:
            z_line_xlist.extend([self.grid_east.min(),
                                     self.grid_east.max()])
            z_line_xlist.append(None)
            z_line_ylist.extend([zz, zz])
            z_line_ylist.append(None)
        ax2.plot(z_line_xlist,
                 z_line_ylist,
                 lw=line_width,
                 color=line_color)
                      
        
        #--> plot stations
        ax2.scatter(self.station_locations['east_c'],
                    [0]*self.station_locations.shape[0],
                    marker=station_marker,
                    c=marker_color,
                    s=marker_size)

        
        if z_limits == None:
            ax2.set_ylim(self.z_target_depth, -200)
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
        if save_path is not None:
            self.save_path = save_path
            
        if self.save_path is None:
            self.save_path = os.getcwd()
            self.initial_fn = os.path.join(self.save_path, "WSInitialModel")
        elif os.path.isdir(self.save_path) == True:
            self.initial_fn = os.path.join(self.save_path, "WSInitialModel")
        else:
            self.save_path = os.path.dirname(self.save_path)
            self.initial_fn= os.path.join(self.save_path)
        
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
        
        if os.path.isfile(self.model_fn) == True:
            self.read_model_file()
        
    def read_model_file(self):
        """
        read in a model file as x-north, y-east, z-positive down
        """            
        
        mfid = file(self.model_fn, 'r')
        mlines = mfid.readlines()
        mfid.close()
    
        #get info at the beggining of file
        info = mlines[0].strip().split()
        self.iteration_number = int(info[2])
        self.rms = float(info[5])
        try:
            self.lagrange = float(info[8])
        except IndexError:
            print 'Did not get Lagrange Multiplier'
        
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
#==============================================================================
# response
#==============================================================================
class WSResponse(object):
    """
    class to deal with model responses.
    """
    
    def __init__(self, resp_fn, station_fn=None, wl_station_fn=None):
        self.resp_fn = resp_fn
        self.station_fn = station_fn
        self.wl_sites_fn = wl_station_fn
        
        self.period_list = None
        self.resp = None
        
        self.units = 'mv'
        self._zconv = 796.
        
        if os.path.isfile(self.resp_fn) == True:
            self.read_resp_file()
        
        
    def read_resp_file(self, resp_fn=None, wl_sites_fn=None, station_fn=None):
        """
        read in data file
        
        Arguments:
        -----------
            **resp_fn** : string
                          full path to data file
            **sites_fn** : string
                           full path to sites file output by winglink.  This is
                           to match the station name with station number.
            **station_fn** : string
                             full path to station location file
                             
        Outputs:
        --------
            **resp** : structure np.ndarray
                      fills the attribute WSData.data with values
                      
            **period_list** : np.ndarray()
                             fills the period list with values.
        """
        
        if resp_fn is not None:
            self.resp_fn = resp_fn
            
        if wl_sites_fn is not None:
            self.wl_sites_fn = wl_sites_fn
        if station_fn is not None:
            self.station_fn = station_fn
        
        dfid = file(self.resp_fn, 'r')
        dlines = dfid.readlines()
    
        #get size number of stations, number of frequencies, 
        # number of Z components    
        n_stations, n_periods, nz = np.array(dlines[0].strip().split(), 
                                             dtype='int')
        nsstart = 2
        
        self.n_z = nz
        #make a structured array to keep things in for convenience
        z_shape = (n_periods, 4)
        resp_dtype = [('station', '|S10'),
                      ('east', np.float),
                      ('north', np.float),
                      ('z_resp', (np.complex, z_shape)),
                      ('z_resp_err', (np.complex, z_shape))]
        self.resp = np.zeros(n_stations, dtype=resp_dtype)
        
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
        if self.wl_sites_fn != None:
            slist, station_list = wl.read_sites_file(self.wl_sites_fn)
            self.resp['station'] = station_list
        
        elif self.station_fn != None:
            stations = WSStation(self.station_fn)
            stations.read_station_file()
            self.resp['station'] = stations.names
        else:
            self.resp['station'] = np.arange(n_stations)
            
    
        #get N-S locations
        for ii, dline in enumerate(dlines[findlist[0]+1:findlist[1]],0):
            dline = dline.strip().split()
            for jj in range(ncol):
                try:
                    self.resp['north'][ii*ncol+jj] = float(dline[jj])
                except IndexError:
                    pass
                except ValueError:
                    break
                
        #get E-W locations
        for ii, dline in enumerate(dlines[findlist[1]+1:findlist[2]],0):
            dline = dline.strip().split()
            for jj in range(self.n_z):
                try:
                    self.resp['east'][ii*ncol+jj] = float(dline[jj])
                except IndexError:
                    pass
                except ValueError:
                    break
        #make some empty array to put stuff into
        self.period_list = np.zeros(n_periods)
        
        #get resp
        per = 0
        for ii, dl in enumerate(dlines[findlist[2]:]):
            if dl.lower().find('period') > 0:
                st = 0
                if dl.lower().find('data') == 0:
                    dkey = 'z_resp'
                    self.period_list[per] = float(dl.strip().split()[1])
                per += 1
            
            elif dl.lower().find('#iteration') >= 0:
                break
            else:
                zline = np.array(dl.strip().split(),dtype=np.float)*self._zconv
                self.resp[st][dkey][per-1,:] = np.array([zline[0]-1j*zline[1],
                                                         zline[2]-1j*zline[3],
                                                         zline[4]-1j*zline[5],
                                                         zline[6]-1j*zline[7]])
                st += 1

#==============================================================================
# plot response       
#==============================================================================
class PlotResponse(object):
    """
    plot data and response
    """
    
    def __init__(self, data_fn=None, resp_fn=None, station_fn=None, **kwargs):
        self.data_fn = data_fn
        self.resp_fn = resp_fn
        self.station_fn = station_fn
        
        self.data_object = None
        self.resp_object = []
        
        self.color_mode = kwargs.pop('color_mode', 'color')
        
        self.ms = kwargs.pop('ms', 1.5)
        self.lw = kwargs.pop('lw', .5)
        self.e_capthick = kwargs.pop('e_capthick', .5)
        self.e_capsize = kwargs.pop('e_capsize', 2)

        #color mode
        if self.color_mode == 'color':
            #color for data
            self.cted = kwargs.pop('cted', (0, 0, 1))
            self.ctmd = kwargs.pop('ctmd', (1, 0, 0))
            self.mted = kwargs.pop('mted', 's')
            self.mtmd = kwargs.pop('mtmd', 'o')
            
            #color for occam2d model
            self.ctem = kwargs.pop('ctem', (0, .6, .3))
            self.ctmm = kwargs.pop('ctmm', (.9, 0, .8))
            self.mtem = kwargs.pop('mtem', '+')
            self.mtmm = kwargs.pop('mtmm', '+')
         
        #black and white mode
        elif self.color_mode == 'bw':
            #color for data
            self.cted = kwargs.pop('cted', (0, 0, 0))
            self.ctmd = kwargs.pop('ctmd', (0, 0, 0))
            self.mted = kwargs.pop('mted', '*')
            self.mtmd = kwargs.pop('mtmd', 'v')
            
            #color for occam2d model
            self.ctem = kwargs.pop('ctem', (0.6, 0.6, 0.6))
            self.ctmm = kwargs.pop('ctmm', (0.6, 0.6, 0.6))
            self.mtem = kwargs.pop('mtem', '+')
            self.mtmm = kwargs.pop('mtmm', 'x')
            
        self.phase_limits = kwargs.pop('phase_limits', (-5, 95))
        self.res_limits = kwargs.pop('res_limits', None)

        self.fig_num = kwargs.pop('fig_num', 1)
        self.fig_size = kwargs.pop('fig_size', [6, 6])
        self.fig_dpi = kwargs.pop('dpi', 300)
        
        self.subplot_wspace = .05
        self.subplot_hspace = .0
        self.subplot_right = .98
        self.subplot_left = .08
        self.subplot_top = .93
        self.subplot_bottom = .1
        
        self.legend_loc = 'upper left'
        self.legend_marker_scale = 1
        self.legend_border_axes_pad = .01
        self.legend_label_spacing = 0.07
        self.legend_handle_text_pad = .2
        self.legend_border_pad = .15

        self.font_size = kwargs.pop('font_size', 6)
        
        self.plot_type = kwargs.pop('plot_type', '1')
        self.plot_style = kwargs.pop('plot_style', 1)
        self.plot_component = kwargs.pop('plot_component', 4)
        self.plot_yn = kwargs.pop('plot_yn', 'y')
        
        self.fig_list = []
        
        if self.plot_yn == 'y':
            self.plot()
            
    def plot_errorbar(self, ax, period, data, error, color, marker):
        """
        convinience function to make an error bar instance
        """
        
        errorbar_object = ax.errorbar(period,
                                      data,
                                      marker=marker,
                                      ms=self.ms,
                                      mfc='None',
                                      mec=color,
                                      ls=':',
                                      yerr=error, 
                                      ecolor=color,   
                                      color=color,
                                      picker=2,
                                      lw=self.lw,
                                      elinewidth=self.lw,
                                      capsize=self.e_capsize,
                                      capthick=self.e_capthick)
        return errorbar_object
    
    def plot(self):
        """
        plot
        """
        
        self.data_object = WSData()
        self.data_object.read_data_file(self.data_fn, 
                                        station_fn=self.station_fn)
                                        
        #get shape of impedance tensors
        ns = self.data_object.data['station'].shape[0]
        nf = len(self.data_object.period_list)
    
        #read in response files
        if self.resp_fn != None:
            self.resp_object = []
            if type(self.resp_fn) is not list:
                self.resp_object = [WSResponse(self.resp_fn, 
                                               station_fn=self.station_fn)]
            else:
                for rfile in self.resp_fn:
                    self.resp_object.append(WSResponse(rfile, 
                                                       station_fn=self.station_fn))

        #get number of response files
        nr = len(self.resp_object)
        
        if type(self.plot_type) is list:
            ns = len(self.plot_type)
          
        #--> set default font size                           
        plt.rcParams['font.size'] = self.font_size

        fontdict = {'size':self.font_size+2, 'weight':'bold'}    
        gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1.5], hspace=.1)
        
        ax_list = []
        line_list = []
        label_list = []
        
        
        if self.plot_type != '1':
            pstation_list = []
            if type(self.plot_type) is not list:
                self.plot_type = [self.plot_type]
            for ii, station in enumerate(self.data_object.data['station']):
                if type(station) is not int:
                    for pstation in self.plot_type:
                        if station.find(str(pstation)) >= 0:
                            pstation_list.append(ii)
                else:
                    for pstation in self.plot_type:
                        if station == int(pstation):
                            pstation_list.append(ii)
        else:
            pstation_list = np.arange(ns)
        
        for jj in pstation_list:
            data_z = self.data_object.data[jj]['z_data'].reshape(nf, 2, 2)
            data_z_err = (self.data_object.data[jj]['z_err_map']*\
                         self.data_object.data[jj]['z_data_err']).reshape(nf, 2, 2)
            period = self.data_object.period_list
            station = self.data_object.data['station'][jj]
            print 'Plotting: {0}'.format(station)
            
            #check for masked points
            data_z[np.where(data_z == 7.95204E5-7.95204E5j)] = 0.0+0.0j
            data_z_err[np.where(data_z_err == 7.95204E5-7.95204E5j)] =\
                                                                1.0+1.0j
            
            #convert to apparent resistivity and phase
            z_object =  mtz.Z(z_array=data_z, zerr_array=data_z_err)
            z_object.freq = 1./period
    
            rp = mtplottools.ResPhase(z_object)
            
            #find locations where points have been masked
            nzxx = np.where(rp.resxx!=0)[0]
            nzxy = np.where(rp.resxy!=0)[0]
            nzyx = np.where(rp.resyx!=0)[0]
            nzyy = np.where(rp.resyy!=0)[0]
            
            if self.resp_fn != None:
                plotr = True
            else:
                plotr = False
            
            #make figure 
            fig = plt.figure(self.fig_num+jj, self.fig_size, dpi=self.fig_dpi)
            plt.clf()
            fig.suptitle(str(station), fontdict=fontdict)
            
            #set the grid of subplots
            gs = gridspec.GridSpec(2, 4,
                                   wspace=self.subplot_wspace,
                                   left=self.subplot_left,
                                   top=self.subplot_top,
                                   bottom=self.subplot_bottom, 
                                   right=self.subplot_right, 
                                   hspace=self.subplot_hspace,
                                   height_ratios=[2, 1.5])
            #---------plot the apparent resistivity-----------------------------------
            #plot each component in its own subplot
            if self.plot_style == 1:
                if self.plot_component == 2:
                    axrxy = fig.add_subplot(gs[0, 0:2])
                    axryx = fig.add_subplot(gs[0, 2:], sharex=axrxy)
                    
                    axpxy = fig.add_subplot(gs[1, 0:2])
                    axpyx = fig.add_subplot(gs[1, 2:], sharex=axrxy)
                    
                    #plot resistivity
                    erxy = self.plot_errorbar(axrxy, 
                                              period[nzxy], 
                                              rp.resxy[nzxy], 
                                              rp.resxy_err[nzxy],
                                              self.cted, self.mted)
                    eryx = self.plot_errorbar(axryx, 
                                              period[nzyx], 
                                              rp.resyx[nzyx], 
                                              rp.resyx_err[nzyx],
                                              self.ctmd, self.mtmd)
                    #plot phase                         
                    erxy = self.plot_errorbar(axpxy, 
                                              period[nzxy], 
                                              rp.phasexy[nzxy], 
                                              rp.phasexy_err[nzxy],
                                              self.cted, self.mted)
                    eryx = self.plot_errorbar(axpyx, 
                                              period[nzyx], 
                                              rp.phaseyx[nzyx], 
                                              rp.phaseyx_err[nzyx],
                                              self.ctmd, self.mtmd)
                                              
                    ax_list = [axrxy, axryx, axpxy, axpyx]
                    line_list = [[erxy[0]], [eryx[0]]]
                    label_list = [['$Z_{xy}$'], ['$Z_{yx}$']]
                                                           
                elif self.plot_component == 4:
                    axrxx = fig.add_subplot(gs[0, 0])
                    axrxy = fig.add_subplot(gs[0, 1], sharex=axrxx)
                    axryx = fig.add_subplot(gs[0, 2], sharex=axrxx)
                    axryy = fig.add_subplot(gs[0, 3], sharex=axrxx)
                    
                    axpxx = fig.add_subplot(gs[1, 0])
                    axpxy = fig.add_subplot(gs[1, 1], sharex=axrxx)
                    axpyx = fig.add_subplot(gs[1, 2], sharex=axrxx)
                    axpyy = fig.add_subplot(gs[1, 3], sharex=axrxx)
                    
                    #plot resistivity
                    erxx= self.plot_errorbar(axrxx, 
                                              period[nzxx], 
                                              rp.resxx[nzxx], 
                                              rp.resxx_err[nzxx],
                                              self.cted, self.mted)
                    erxy = self.plot_errorbar(axrxy, 
                                              period[nzxy], 
                                              rp.resxy[nzxy], 
                                              rp.resxy_err[nzxy],
                                              self.cted, self.mted)
                    eryx = self.plot_errorbar(axryx, 
                                              period[nzyx], 
                                              rp.resyx[nzyx], 
                                              rp.resyx_err[nzyx],
                                              self.ctmd, self.mtmd)
                    eryy = self.plot_errorbar(axryy, 
                                              period[nzyy], 
                                              rp.resyy[nzyy], 
                                              rp.resyy_err[nzyy],
                                              self.ctmd, self.mtmd)
                    #plot phase                         
                    erxx= self.plot_errorbar(axpxx, 
                                              period[nzxx], 
                                              rp.phasexx[nzxx], 
                                              rp.phasexx_err[nzxx],
                                              self.cted, self.mted)
                    erxy = self.plot_errorbar(axpxy, 
                                              period[nzxy], 
                                              rp.phasexy[nzxy], 
                                              rp.phasexy_err[nzxy],
                                              self.cted, self.mted)
                    eryx = self.plot_errorbar(axpyx, 
                                              period[nzyx], 
                                              rp.phaseyx[nzyx], 
                                              rp.phaseyx_err[nzyx],
                                              self.ctmd, self.mtmd)
                    eryy = self.plot_errorbar(axpyy, 
                                              period[nzyy], 
                                              rp.phaseyy[nzyy], 
                                              rp.phaseyy_err[nzyy],
                                              self.ctmd, self.mtmd)
                    ax_list = [axrxx, axrxy, axryx, axryy, 
                               axpxx, axpxy, axpyx, axpyy]
                    line_list = [[erxx[0]], [erxy[0]], [eryx[0]], [eryy[0]]]
                    label_list = [['$Z_{xx}$'], ['$Z_{xy}$'], 
                                  ['$Z_{yx}$'], ['$Z_{yy}$']]
                    
                #set axis properties
                for aa, ax in enumerate(ax_list):
                    if len(ax_list) == 4:
                        if aa < 2:
                            plt.setp(ax.get_xticklabels(), visible=False)
                            ax.set_yscale('log')
                            if self.res_limits is not None:
                                ax.set_ylim(self.res_limits)
                        else:
                            ax.set_ylim(self.phase_limits)
                            ax.set_xlabel('Period (s)', fontdict=fontdict)
                        #set axes labels
                        if aa == 0:
                            ax.set_ylabel('App. Res. ($\mathbf{\Omega \cdot m}$)',
                                          fontdict=fontdict)
                        elif aa == 2:
                            ax.set_ylabel('Phase (deg)',
                                          fontdict=fontdict)
                        else:
                            plt.setp(ax.yaxis.get_ticklabels(), visible=False)
                            
                    elif len(ax_list) == 8:
                        if aa < 4:
                            plt.setp(ax.get_xticklabels(), visible=False)
                            ax.set_yscale('log')
                            if self.res_limits is not None:
                                ax.set_ylim(self.res_limits)
                        else:
                            ax.set_ylim(self.phase_limits)
                            ax.set_xlabel('Period (s)', fontdict=fontdict)
                        #set axes labels
                        if aa == 0:
                            ax.set_ylabel('App. Res. ($\mathbf{\Omega \cdot m}$)',
                                          fontdict=fontdict)
                        elif aa == 4:
                            ax.set_ylabel('Phase (deg)',
                                          fontdict=fontdict)
                        else:
                            plt.setp(ax.yaxis.get_ticklabels(), visible=False)

                    ax.set_xscale('log')
                    ax.set_xlim(xmin=10**(np.floor(np.log10(period[0])))*1.01,
                             xmax=10**(np.ceil(np.log10(period[-1])))*.99)
                    ax.grid(True, alpha=.25)
                    
            # plot xy and yx together and xx, yy together
            elif self.plot_style == 2:
                if self.plot_component == 2:
                    axrxy = fig.add_subplot(gs[0, 0:])
                    axpxy = fig.add_subplot(gs[1, 0:], sharex=axrxy)
                    
                    #plot resistivity
                    erxy = self.plot_errorbar(axrxy, 
                                              period[nzxy], 
                                              rp.resxy[nzxy], 
                                              rp.resxy_err[nzxy],
                                              self.cted, self.mted)
                    eryx = self.plot_errorbar(axrxy, 
                                              period[nzyx], 
                                              rp.resyx[nzyx], 
                                              rp.resyx_err[nzyx],
                                              self.ctmd, self.mtmd)
                    #plot phase                         
                    erxy = self.plot_errorbar(axpxy, 
                                              period[nzxy], 
                                              rp.phasexy[nzxy], 
                                              rp.phasexy_err[nzxy],
                                              self.cted, self.mted)
                    eryx = self.plot_errorbar(axpxy, 
                                              period[nzyx], 
                                              rp.phaseyx[nzyx], 
                                              rp.phaseyx_err[nzyx],
                                              self.ctmd, self.mtmd)
                    ax_list = [axrxy, axpxy]
                    line_list = [erxy[0], eryx[0]]
                    label_list = ['$Z_{xy}$', '$Z_{yx}$']
                    
                elif self.plot_component == 4:
                    axrxy = fig.add_subplot(gs[0, 0:2])
                    axpxy = fig.add_subplot(gs[1, 0:2], sharex=axrxy)
                    
                    axrxx = fig.add_subplot(gs[0, 2:], sharex=axrxy)
                    axpxx = fig.add_subplot(gs[1, 2:], sharex=axrxy)
                    
                    #plot resistivity
                    erxx= self.plot_errorbar(axrxx, 
                                              period[nzxx], 
                                              rp.resxx[nzxx], 
                                              rp.resxx_err[nzxx],
                                              self.cted, self.mted)
                    erxy = self.plot_errorbar(axrxy, 
                                              period[nzxy], 
                                              rp.resxy[nzxy], 
                                              rp.resxy_err[nzxy],
                                              self.cted, self.mted)
                    eryx = self.plot_errorbar(axrxy, 
                                              period[nzyx], 
                                              rp.resyx[nzyx], 
                                              rp.resyx_err[nzyx],
                                              self.ctmd, self.mtmd)
                    eryy = self.plot_errorbar(axrxx, 
                                              period[nzyy], 
                                              rp.resyy[nzyy], 
                                              rp.resyy_err[nzyy],
                                              self.ctmd, self.mtmd)
                    #plot phase                         
                    erxx= self.plot_errorbar(axpxx, 
                                              period[nzxx], 
                                              rp.phasexx[nzxx], 
                                              rp.phasexx_err[nzxx],
                                              self.cted, self.mted)
                    erxy = self.plot_errorbar(axpxy, 
                                              period[nzxy], 
                                              rp.phasexy[nzxy], 
                                              rp.phasexy_err[nzxy],
                                              self.cted, self.mted)
                    eryx = self.plot_errorbar(axpxy, 
                                              period[nzyx], 
                                              rp.phaseyx[nzyx], 
                                              rp.phaseyx_err[nzyx],
                                              self.ctmd, self.mtmd)
                    eryy = self.plot_errorbar(axpxx, 
                                              period[nzyy], 
                                              rp.phaseyy[nzyy], 
                                              rp.phaseyy_err[nzyy],
                                              self.ctmd, self.mtmd)
                                              
                    ax_list = [axrxy, axrxx, axpxy, axpxx]
                    line_list = [[erxy[0], eryx[0]], [erxx[0], eryy[0]]]
                    label_list = [['$Z_{xy}$', '$Z_{yx}$'], 
                                  ['$Z_{xx}$', '$Z_{yy}$']]
                #set axis properties
                for aa, ax in enumerate(ax_list):
                    if len(ax_list) == 2:
                        ax.set_xlabel('Period (s)', fontdict=fontdict)
                        if aa == 0:
                            plt.setp(ax.get_xticklabels(), visible=False)
                            ax.set_yscale('log')
                            ax.set_ylabel('App. Res. ($\mathbf{\Omega \cdot m}$)',
                                          fontdict=fontdict)
                            if self.res_limits is not None:
                                ax.set_ylim(self.res_limits)
                        else:
                            ax.set_ylim(self.phase_limits)
                            ax.set_ylabel('Phase (deg)', fontdict=fontdict)
                    elif len(ax_list) == 4:
                        if aa < 2:
                            plt.setp(ax.get_xticklabels(), visible=False)
                            ax.set_yscale('log')
                            if self.res_limits is not None:
                                ax.set_ylim(self.res_limits)
                        else:
                            ax.set_ylim(self.phase_limits)
                            ax.set_xlabel('Period (s)', fontdict=fontdict)
                        if aa == 0:
                            ax.set_ylabel('App. Res. ($\mathbf{\Omega \cdot m}$)',
                                       fontdict=fontdict)
                        elif aa == 2:
                            ax.set_ylabel('Phase (deg)', fontdict=fontdict)
                        else:
                            plt.setp(ax.yaxis.get_ticklabels(), visible=False)

                    ax.set_xscale('log')
                    ax.set_xlim(xmin=10**(np.floor(np.log10(period[0])))*1.01,
                                xmax=10**(np.ceil(np.log10(period[-1])))*.99)
                    ax.grid(True,alpha=.25)

            if plotr == True:
                for rr in range(nr):
                    if self.color_mode == 'color':   
                        cxy = (0,.4+float(rr)/(3*nr),0)
                        cyx = (.7+float(rr)/(4*nr),.13,.63-float(rr)/(4*nr))
                    elif self.color_mode == 'bw':
                        cxy = (1-1.25/(rr+2.),1-1.25/(rr+2.),1-1.25/(rr+2.))                    
                        cyx = (1-1.25/(rr+2.),1-1.25/(rr+2.),1-1.25/(rr+2.))
                    
                    resp_z = self.resp_object[rr].resp['z_resp'][jj].reshape(nf, 2, 2)
                    resp_z_err = (data_z-resp_z)/data_z_err
                    resp_z_object =  mtz.Z(z_array=resp_z, 
                                           zerr_array=resp_z_err, 
                                           freq=1./period)
    
                    rrp = mtplottools.ResPhase(resp_z_object)
    
                    rms = resp_z_err.std()
                    rms_xx = resp_z_err[:, 0, 0].std()
                    rms_xy = resp_z_err[:, 0, 1].std()
                    rms_yx = resp_z_err[:, 1, 0].std()
                    rms_yy = resp_z_err[:, 1, 1].std()
                    print ' --- response {0} ---'.format(rr)
                    print '  RMS = {:.2f}'.format(rms)
                    print '      RMS_xx = {:.2f}'.format(rms_xx)
                    print '      RMS_xy = {:.2f}'.format(rms_xy)
                    print '      RMS_yx = {:.2f}'.format(rms_yx)
                    print '      RMS_yy = {:.2f}'.format(rms_yy)
                    
                    if self.plot_style == 1:
                        if self.plot_component == 2:
                            #plot resistivity
                            rerxy = self.plot_errorbar(axrxy, 
                                                      period[nzxy], 
                                                      rrp.resxy[nzxy], 
                                                      rrp.resxy_err[nzxy],
                                                      cxy, self.mted)
                            reryx = self.plot_errorbar(axryx, 
                                                      period[nzyx], 
                                                      rrp.resyx[nzyx], 
                                                      rrp.resyx_err[nzyx],
                                                      cyx, self.mtmd)
                            #plot phase                         
                            rerxy = self.plot_errorbar(axpxy, 
                                                      period[nzxy], 
                                                      rrp.phasexy[nzxy], 
                                                      rrp.phasexy_err[nzxy],
                                                      cxy, self.mted)
                            reryx = self.plot_errorbar(axpyx, 
                                                      period[nzyx], 
                                                      rrp.phaseyx[nzyx], 
                                                      rrp.phaseyx_err[nzyx],
                                                      cyx, self.mtmd)
                                                      
                            line_list[0] += [rerxy[0]]
                            line_list[1] += [reryx[0]]
                            label_list[0] += ['$Z^m_{xy}$ '+
                                               'rms={0:.2f}'.format(rms_xy)]
                            label_list[1] += ['$Z^m_{yx}$ '+
                                           'rms={0:.2f}'.format(rms_yx)]
                        elif self.plot_component == 4:
                            #plot resistivity
                            rerxx= self.plot_errorbar(axrxx, 
                                                      period[nzxx], 
                                                      rrp.resxx[nzxx], 
                                                      rrp.resxx_err[nzxx],
                                                      cxy, self.mted)
                            rerxy = self.plot_errorbar(axrxy, 
                                                      period[nzxy], 
                                                      rrp.resxy[nzxy], 
                                                      rrp.resxy_err[nzxy],
                                                      cxy, self.mted)
                            reryx = self.plot_errorbar(axryx, 
                                                      period[nzyx], 
                                                      rrp.resyx[nzyx], 
                                                      rrp.resyx_err[nzyx],
                                                      cyx, self.mtmd)
                            reryy = self.plot_errorbar(axryy, 
                                                      period[nzyy], 
                                                      rrp.resyy[nzyy], 
                                                      rrp.resyy_err[nzyy],
                                                      cyx, self.mtmd)
                            #plot phase                         
                            rerxx= self.plot_errorbar(axpxx, 
                                                      period[nzxx], 
                                                      rrp.phasexx[nzxx], 
                                                      rrp.phasexx_err[nzxx],
                                                      cxy, self.mted)
                            rerxy = self.plot_errorbar(axpxy, 
                                                      period[nzxy], 
                                                      rrp.phasexy[nzxy], 
                                                      rrp.phasexy_err[nzxy],
                                                      cxy, self.mted)
                            reryx = self.plot_errorbar(axpyx, 
                                                      period[nzyx], 
                                                      rrp.phaseyx[nzyx], 
                                                      rrp.phaseyx_err[nzyx],
                                                      cyx, self.mtmd)
                            reryy = self.plot_errorbar(axpyy, 
                                                      period[nzyy], 
                                                      rrp.phaseyy[nzyy], 
                                                      rrp.phaseyy_err[nzyy],
                                                      cyx, self.mtmd)
                            line_list[0] += [rerxx[0]]
                            line_list[1] += [rerxy[0]]
                            line_list[2] += [reryx[0]]
                            line_list[3] += [reryy[0]]
                            label_list[0] += ['$Z^m_{xx}$ '+
                                               'rms={0:.2f}'.format(rms_xx)]
                            label_list[1] += ['$Z^m_{xy}$ '+
                                           'rms={0:.2f}'.format(rms_xy)]
                            label_list[2] += ['$Z^m_{yx}$ '+
                                           'rms={0:.2f}'.format(rms_yx)]
                            label_list[3] += ['$Z^m_{yy}$ '+
                                           'rms={0:.2f}'.format(rms_yy)]
                    elif self.plot_style == 2:
                        if self.plot_component == 2:
                            #plot resistivity
                            rerxy = self.plot_errorbar(axrxy, 
                                                      period[nzxy], 
                                                      rrp.resxy[nzxy], 
                                                      rrp.resxy_err[nzxy],
                                                      cxy, self.mted)
                            reryx = self.plot_errorbar(axrxy, 
                                                      period[nzyx], 
                                                      rrp.resyx[nzyx], 
                                                      rrp.resyx_err[nzyx],
                                                      cyx, self.mtmd)
                            #plot phase                         
                            rerxy = self.plot_errorbar(axpxy, 
                                                      period[nzxy], 
                                                      rrp.phasexy[nzxy], 
                                                      rrp.phasexy_err[nzxy],
                                                      cxy, self.mted)
                            reryx = self.plot_errorbar(axpxy, 
                                                      period[nzyx], 
                                                      rrp.phaseyx[nzyx], 
                                                      rrp.phaseyx_err[nzyx],
                                                      cyx, self.mtmd)
                            line_list += [rerxy[0], reryx[0]]
                            label_list += ['$Z^m_{xy}$ '+
                                           'rms={0:.2f}'.format(rms_xy),
                                           '$Z^m_{yx}$ '+
                                           'rms={0:.2f}'.format(rms_yx)]
                        elif self.plot_component == 4:
                            #plot resistivity
                            rerxx= self.plot_errorbar(axrxx, 
                                                      period[nzxx], 
                                                      rrp.resxx[nzxx], 
                                                      rrp.resxx_err[nzxx],
                                                      cxy, self.mted)
                            rerxy = self.plot_errorbar(axrxy, 
                                                      period[nzxy], 
                                                      rrp.resxy[nzxy], 
                                                      rrp.resxy_err[nzxy],
                                                      cxy, self.mted)
                            reryx = self.plot_errorbar(axrxy, 
                                                      period[nzyx], 
                                                      rrp.resyx[nzyx], 
                                                      rrp.resyx_err[nzyx],
                                                      cyx, self.mtmd)
                            reryy = self.plot_errorbar(axrxx, 
                                                      period[nzyy], 
                                                      rrp.resyy[nzyy], 
                                                      rrp.resyy_err[nzyy],
                                                      cyx, self.mtmd)
                            #plot phase                         
                            rerxx= self.plot_errorbar(axpxx, 
                                                      period[nzxx], 
                                                      rrp.phasexx[nzxx], 
                                                      rrp.phasexx_err[nzxx],
                                                      cxy, self.mted)
                            rerxy = self.plot_errorbar(axpxy, 
                                                      period[nzxy], 
                                                      rrp.phasexy[nzxy], 
                                                      rrp.phasexy_err[nzxy],
                                                      cxy, self.mted)
                            reryx = self.plot_errorbar(axpxy, 
                                                      period[nzyx], 
                                                      rrp.phaseyx[nzyx], 
                                                      rrp.phaseyx_err[nzyx],
                                                      cyx, self.mtmd)
                            reryy = self.plot_errorbar(axpxx, 
                                                      period[nzyy], 
                                                      rrp.phaseyy[nzyy], 
                                                      rrp.phaseyy_err[nzyy],
                                                      cyx, self.mtmd)
                            
                            line_list[0] += [rerxy[0], reryx[0]]
                            line_list[1] += [rerxx[0], reryy[0]]
                            label_list[0] += ['$Z^m_{xy}$ '+
                                               'rms={0:.2f}'.format(rms_xy),
                                              '$Z^m_{yx}$ '+
                                              'rms={0:.2f}'.format(rms_yx)]
                            label_list[1] += ['$Z^m_{xx}$ '+
                                               'rms={0:.2f}'.format(rms_xx),
                                              '$Z^m_{yy}$ '+
                                              'rms={0:.2f}'.format(rms_yy)]
                    
                #make legends
                if self.plot_style == 1:
                    for aa, ax in enumerate(ax_list[0:self.plot_component]):
                        ax.legend(line_list[aa],
                                  label_list[aa],
                                  loc=self.legend_loc,
                                  markerscale=self.legend_marker_scale,
                                  borderaxespad=self.legend_border_axes_pad,
                                  labelspacing=self.legend_label_spacing,
                                  handletextpad=self.legend_handle_text_pad,
                                  borderpad=self.legend_border_pad,
                                  prop={'size':max([self.font_size/nr, 5])})
                if self.plot_style == 2:
                    if self.plot_component == 2:
                        axrxy.legend(line_list,
                                      label_list,
                                      loc=self.legend_loc,
                                      markerscale=self.legend_marker_scale,
                                      borderaxespad=self.legend_border_axes_pad,
                                      labelspacing=self.legend_label_spacing,
                                      handletextpad=self.legend_handle_text_pad,
                                      borderpad=self.legend_border_pad,
                                      prop={'size':max([self.font_size/nr, 5])})
                    else:
                        for aa, ax in enumerate(ax_list[0:self.plot_component/2]):
                            ax.legend(line_list[aa],
                                      label_list[aa],
                                      loc=self.legend_loc,
                                      markerscale=self.legend_marker_scale,
                                      borderaxespad=self.legend_border_axes_pad,
                                      labelspacing=self.legend_label_spacing,
                                      handletextpad=self.legend_handle_text_pad,
                                      borderpad=self.legend_border_pad,
                                      prop={'size':max([self.font_size/nr, 5])})
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
        for fig in self.fig_list:
            plt.close(fig)
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
        
        return ("Plots data vs model response computed by WS3DINV")

#==============================================================================
# plot depth slices
#==============================================================================
class PlotDepthSlice(object):
    """
    plot depth slices
    """
    
    def __init__(self, model_fn=None, data_fn=None, station_fn=None, 
                 initial_fn=None, **kwargs):
        self.model_fn = model_fn
        self.data_fn = data_fn
        self.station_fn = station_fn
        self.initial_fn = initial_fn
        
        self.save_path = kwargs.pop('save_path', None)
        if self.model_fn is not None and self.save_path is None:
            self.save_path = os.path.dirname(self.model_fn)
        elif self.initial_fn is not None and self.save_path is None:
            self.save_path = os.path.dirname(self.initial_fn)
            
        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)
                
        self.save_plots = kwargs.pop('save_plots', 'y')
        
        self.depth_index = kwargs.pop('depth_index', None)
        self.map_scale = kwargs.pop('map_scale', 'km')
        #make map scale
        if self.map_scale=='km':
            self.dscale=1000.
        elif self.map_scale=='m':
            self.dscale=1. 
        self.ew_limits = kwargs.pop('ew_limits', None)
        self.ns_limits = kwargs.pop('ns_limits', None)
        
        self.plot_grid = kwargs.pop('plot_grid', 'n')
        
        self.fig_num = kwargs.pop('fig_num', 1)
        self.fig_size = kwargs.pop('fig_size', [6, 6])
        self.fig_dpi = kwargs.pop('dpi', 300)
        self.fig_aspect = kwargs.pop('fig_aspect', 1)
        self.title = kwargs.pop('title', 'on')
        self.fig_list = []
        
        self.xminorticks = kwargs.pop('xminorticks', 1000)
        self.yminorticks = kwargs.pop('yminorticks', 1000)
        
        self.climits = kwargs.pop('climits', (0,4))
        self.cmap = kwargs.pop('cmap', 'jet_r')
        self.font_size = kwargs.pop('font_size', 8)
        
        self.cb_shrink = kwargs.pop('cb_shrink', .8)
        self.cb_pad = kwargs.pop('cb_pad', .01)
        self.cb_orientation = kwargs.pop('cb_orientation', 'horizontal')
        self.cb_location = kwargs.pop('cb_location', None)
        
        self.subplot_right = .99
        self.subplot_left = .085
        self.subplot_top = .92
        self.subplot_bottom = .1
        
        self.res_model = None
        self.grid_east = None
        self.grid_north = None
        self.grid_z  = None
        
        self.nodes_east = None
        self.nodes_north = None
        self.nodes_z = None
        
        self.mesh_east = None
        self.mesh_north = None
        
        self.station_east = None
        self.station_north = None
        self.station_names = None
        
        self.plot_yn = kwargs.pop('plot_yn', 'y')
        if self.plot_yn == 'y':
            self.plot()
            
    def read_files(self):
        """
        read in the files to get appropriate information
        """
        #--> read in model file
        if self.model_fn is not None:
            if os.path.isfile(self.model_fn) == True:
                wsmodel = WSModel(self.model_fn)
                self.res_model = wsmodel.res_model
                self.grid_east = wsmodel.grid_east/self.dscale
                self.grid_north = wsmodel.grid_north/self.dscale
                self.grid_z = wsmodel.grid_z/self.dscale
                self.nodes_east = wsmodel.nodes_east/self.dscale
                self.nodes_north = wsmodel.nodes_north/self.dscale
                self.nodes_z = wsmodel.nodes_z/self.dscale
            else:
                raise mtex.MTpyError_file_handling(
                        '{0} does not exist, check path'.format(self.model_fn))
        
        #--> read in data file to get station locations
        if self.data_fn is not None:
            if os.path.isfile(self.data_fn) == True:
                wsdata = WSData()
                wsdata.read_data_file(self.data_fn)
                self.station_east = wsdata.data['east']/self.dscale
                self.station_north = wsdata.data['north']/self.dscale
                self.station_names = wsdata.data['station']
            else:
                print 'Could not find data file {0}'.format(self.data_fn)
            
        #--> read in station file
        if self.station_fn is not None:
            if os.path.isfile(self.station_fn) == True:
                wsstations = WSStation(self.station_fn)
                wsstations.read_station_file()
                self.station_east = wsstations.east/self.dscale
                self.station_north = wsstations.north/self.dscale
                self.station_names = wsstations.names
            else:
                print 'Could not find station file {0}'.format(self.station_fn)
        
        #--> read in initial file
        if self.initial_fn is not None:
            if os.path.isfile(self.initial_fn) == True:
                wsmesh = WSMesh()
                wsmesh.read_initial_file(self.initial_fn)
                self.grid_east = wsmesh.grid_east/self.dscale
                self.grid_north = wsmesh.grid_north/self.dscale
                self.grid_z = wsmesh.grid_z/self.dscale
                self.nodes_east = wsmesh.nodes_east/self.dscale
                self.nodes_north = wsmesh.nodes_north/self.dscale
                self.nodes_z = wsmesh.nodes_z/self.dscale
                
                #need to convert index values to resistivity values
                rdict = dict([(ii,res) for ii,res in enumerate(wsmesh.res_list,1)])
                
                for ii in range(len(wsmesh.res_list)):
                    self.res_model[np.where(wsmesh.res_model==ii+1)] = \
                                                                    rdict[ii+1]
            else:
                raise mtex.MTpyError_file_handling(
                     '{0} does not exist, check path'.format(self.initial_fn))
        
        if self.initial_fn is None and self.model_fn is None:
            raise mtex.MTpyError_inputarguments('Need to input either a model'
                                                ' file or initial file.')

    def plot(self):
        """
        plot depth slices
        """
        #--> get information from files
        self.read_files()

        fdict = {'size':self.font_size+2, 'weight':'bold'}
        
        cblabeldict={-2:'$10^{-3}$',-1:'$10^{-1}$',0:'$10^{0}$',1:'$10^{1}$',
                     2:'$10^{2}$',3:'$10^{3}$',4:'$10^{4}$',5:'$10^{5}$',
                     6:'$10^{6}$',7:'$10^{7}$',8:'$10^{8}$'}
                     
        #create an list of depth slices to plot
        if self.depth_index == None:
            zrange = range(self.grid_z.shape[0])
        elif type(self.depth_index) is int:
            zrange = [self.depth_index]
        elif type(self.depth_index) is list or \
             type(self.depth_index) is np.ndarray:
            zrange = self.depth_index
        
        #set the limits of the plot
        if self.ew_limits == None:
            if self.station_east is not None:
                xlimits = (np.floor(self.station_east.min()), 
                           np.ceil(self.station_east.max()))
            else:
                xlimits = (self.grid_east[5], self.grid_east[-5])
        else:
            xlimits = self.ew_limits
            
        if self.ns_limits == None:
            if self.station_north is not None:
                ylimits = (np.floor(self.station_north.min()), 
                           np.ceil(self.station_north.max()))
            else:
                ylimits = (self.grid_north[5], self.grid_north[-5])
        else:
            ylimits = self.ns_limits
            
            
        #make a mesh grid of north and east
        self.mesh_east, self.mesh_north = np.meshgrid(self.grid_east, 
                                                      self.grid_north,
                                                      indexing='ij')
        
        plt.rcParams['font.size'] = self.font_size
        
        #--> plot depths into individual figures
        for ii in zrange: 
            fig = plt.figure(ii, figsize=self.fig_size, dpi=self.fig_dpi)
            plt.clf()
            ax1 = fig.add_subplot(1, 1, 1, aspect=self.fig_aspect)
            plot_res = np.log10(self.res_model[:, :, ii].T)
            mesh_plot = ax1.pcolormesh(self.mesh_east,
                                       self.mesh_north, 
                                       plot_res,
                                       cmap=self.cmap,
                                       vmin=self.climits[0],
                                       vmax=self.climits[1])
                           
            #plot the stations
            if self.station_east is not None:
                for ee, nn in zip(self.station_east, self.station_north):
                    ax1.text(ee, nn, '*', 
                             verticalalignment='center',
                             horizontalalignment='center',
                             fontdict={'size':5, 'weight':'bold'})
    
            #set axis properties
            ax1.set_xlim(xlimits)
            ax1.set_ylim(ylimits)
            ax1.xaxis.set_minor_locator(MultipleLocator(self.xminorticks/self.dscale))
            ax1.yaxis.set_minor_locator(MultipleLocator(self.yminorticks/self.dscale))
            ax1.set_ylabel('Northing ('+self.map_scale+')',fontdict=fdict)
            ax1.set_xlabel('Easting ('+self.map_scale+')',fontdict=fdict)
            ax1.set_title('Depth = {0:.3f} ({1})'.format(self.grid_z[ii], 
                                                        self.map_scale),
                                                        fontdict=fdict)
                       
            #plot the grid if desired
            if self.plot_grid == 'y':
                east_line_xlist = []
                east_line_ylist = []            
                for xx in self.grid_east:
                    east_line_xlist.extend([xx, xx])
                    east_line_xlist.append(None)
                    east_line_ylist.extend([self.grid_north.min(), 
                                            self.grid_north.max()])
                    east_line_ylist.append(None)
                ax1.plot(east_line_xlist,
                              east_line_ylist,
                              lw=.25,
                              color='k')
        
                north_line_xlist = []
                north_line_ylist = [] 
                for yy in self.grid_north:
                    north_line_xlist.extend([self.grid_east.min(),
                                             self.grid_east.max()])
                    north_line_xlist.append(None)
                    north_line_ylist.extend([yy, yy])
                    north_line_ylist.append(None)
                ax1.plot(north_line_xlist,
                              north_line_ylist,
                              lw=.25,
                              color='k')
            
                
            #plot the colorbar
            if self.cb_location is None:
                if self.cb_orientation == 'horizontal':
                    self.cb_location = (ax1.axes.figbox.bounds[3]-.225,
                                        ax1.axes.figbox.bounds[1]+.05,.3,.025) 
                                            
                elif self.cb_orientation == 'vertical':
                    self.cb_location = ((ax1.axes.figbox.bounds[2]-.15,
                                        ax1.axes.figbox.bounds[3]-.21,.025,.3))
            
            ax2 = fig.add_axes(self.cb_location)
            
            cb = mcb.ColorbarBase(ax2,
                                  cmap=self.cmap,
                                  norm=Normalize(vmin=self.climits[0],
                                                 vmax=self.climits[1]),
                                  orientation=self.cb_orientation)
                                
            if self.cb_orientation == 'horizontal':
                cb.ax.xaxis.set_label_position('top')
                cb.ax.xaxis.set_label_coords(.5,1.3)
                
                
            elif self.cb_orientation == 'vertical':
                cb.ax.yaxis.set_label_position('right')
                cb.ax.yaxis.set_label_coords(1.25,.5)
                cb.ax.yaxis.tick_left()
                cb.ax.tick_params(axis='y',direction='in')
                                
            cb.set_label('Resistivity ($\Omega \cdot$m)',
                         fontdict={'size':self.font_size+1})
            cb.set_ticks(np.arange(self.climits[0],self.climits[1]+1))
            cb.set_ticklabels([cblabeldict[cc] 
                                for cc in np.arange(self.climits[0],
                                                    self.climits[1]+1)])
            
            self.fig_list.append(fig)
            
            #--> save plots to a common folder
            if self.save_plots == 'y':
                
                fig.savefig(os.path.join(self.save_path,
                            "Depth_{}_{:.4f}.png".format(ii, self.grid_z[ii])),
                            dpi=self.fig_dpi, bbox_inches='tight')
                fig.clear()
                plt.close()
    
            else:
                pass
            
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
        for fig in self.fig_list:
            plt.close(fig)
        self.plot()
        
    def update_plot(self, fig):
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

        fig.canvas.draw()
                          
    def __str__(self):
        """
        rewrite the string builtin to give a useful message
        """
        
        return ("Plots depth slices of model from WS3DINV")
        
#==============================================================================
# plot phase tensors
#==============================================================================
class PlotPTMaps(mtplottools.MTEllipse):
    """
    plot phase tensor maps including residual pt
    """
    
    def __init__(self, data_fn=None, resp_fn=None, station_fn=None, 
                 model_fn=None, initial_fn=None, **kwargs):
        
        self.model_fn = model_fn
        self.data_fn = data_fn
        self.station_fn = station_fn
        self.resp_fn = resp_fn
        self.initial_fn = initial_fn
        
        self.save_path = kwargs.pop('save_path', None)
        if self.model_fn is not None and self.save_path is None:
            self.save_path = os.path.dirname(self.model_fn)
        elif self.initial_fn is not None and self.save_path is None:
            self.save_path = os.path.dirname(self.initial_fn)
            
        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)
                
        self.save_plots = kwargs.pop('save_plots', 'y')
        self.plot_period_list = kwargs.pop('plot_period_list', None)
        
        self.depth_index = kwargs.pop('depth_index', None)
        self.map_scale = kwargs.pop('map_scale', 'km')
        #make map scale
        if self.map_scale=='km':
            self.dscale=1000.
        elif self.map_scale=='m':
            self.dscale=1. 
        self.ew_limits = kwargs.pop('ew_limits', None)
        self.ns_limits = kwargs.pop('ns_limits', None)
        
        self.pad_east = kwargs.pop('pad_east', 2)
        self.pad_north = kwargs.pop('pad_north', 2)
        
        self.plot_grid = kwargs.pop('plot_grid', 'n')
        
        self.fig_num = kwargs.pop('fig_num', 1)
        self.fig_size = kwargs.pop('fig_size', [6, 6])
        self.fig_dpi = kwargs.pop('dpi', 300)
        self.fig_aspect = kwargs.pop('fig_aspect', 1)
        self.title = kwargs.pop('title', 'on')
        self.fig_list = []
        
        self.xminorticks = kwargs.pop('xminorticks', 1000)
        self.yminorticks = kwargs.pop('yminorticks', 1000)
        
        self.residual_cmap = kwargs.pop('residual_cmap', 'mt_wh2or')
        self.font_size = kwargs.pop('font_size', 7)
        
        self.cb_tick_step = kwargs.pop('cb_tick_step', 45)
        self.cb_residual_tick_step = kwargs.pop('cb_residual_tick_step', 3)
        self.cb_pad = kwargs.pop('cb_pad', .14)
        
        self.res_limits = kwargs.pop('res_limits', (0,4))
        self.res_cmap = kwargs.pop('res_cmap', 'jet_r')
        
        #--> set the ellipse properties -------------------
        self._ellipse_dict = kwargs.pop('ellipse_dict', {})
        self._read_ellipse_dict()
        
        self.subplot_right = .99
        self.subplot_left = .085
        self.subplot_top = .92
        self.subplot_bottom = .1
        self.subplot_hspace = .2
        self.subplot_wspace = .05
        
        self.res_model = None
        self.grid_east = None
        self.grid_north = None
        self.grid_z  = None
        
        self.nodes_east = None
        self.nodes_north = None
        self.nodes_z = None
        
        self.mesh_east = None
        self.mesh_north = None
        
        self.station_east = None
        self.station_north = None
        self.station_names = None
        
        self.data = None
        self.resp = None
        self.period_list = None
        
        self.plot_yn = kwargs.pop('plot_yn', 'y')
        if self.plot_yn == 'y':
            self.plot()
            
    def _get_pt(self):
        """
        get phase tensors
        """

        #--> read in data file 
        if self.data_fn is None:
            raise mtex.MTpyError_inputarguments('Need to input a data file')
        wsdata = WSData()
        wsdata.read_data_file(self.data_fn, station_fn=self.station_fn)
        self.data = wsdata.data['z_data']
        self.period_list = wsdata.period_list
        self.station_east = wsdata.data['east']/self.dscale
        self.station_north = wsdata.data['north']/self.dscale
        self.station_names = wsdata.data['station']
        
        if self.plot_period_list is None:
            self.plot_period_list = self.period_list
        else:
            if type(self.plot_period_list) is list:
                #check if entries are index values or actual periods
                if type(self.plot_period_list[0]) is int:
                    self.plot_period_list = [self.period_list[ii]
                                             for ii in self.plot_period_list]
                else:
                    pass
            elif type(self.plot_period_list) is int:
                self.plot_period_list = self.period_list[self.plot_period_list]
                
        #--> read model file 
        if self.model_fn is not None:
            wsmodel = WSModel(self.model_fn)
            self.res_model = wsmodel.res_model
            self.grid_east = wsmodel.grid_east/self.dscale
            self.grid_north = wsmodel.grid_north/self.dscale
            self.grid_z = wsmodel.grid_z/self.dscale
            self.mesh_east, self.mesh_north = np.meshgrid(self.grid_east, 
                                                          self.grid_north,
                                                          indexing='ij')
            
        #--> read response file
        if self.resp_fn is not None:
            wsresp = WSResponse(self.resp_fn)
            self.resp = wsresp.resp['z_resp']
            
        
        
     
    def plot(self):
        """
        plot phase tensor maps for data and or response, each figure is of a
        different period.  If response is input a third column is added which is 
        the residual phase tensor showing where the model is not fitting the data 
        well.  The data is plotted in km in units of ohm-m.
        
        Inputs:
            data_fn = full path to data file
            resp_fn = full path to response file, if none just plots data
            sites_fn = full path to sites file
            periodlst = indicies of periods you want to plot
            esize = size of ellipses as:
                    0 = phase tensor ellipse
                    1 = phase tensor residual
                    2 = resistivity tensor ellipse
                    3 = resistivity tensor residual
            ecolor = 'phimin' for coloring with phimin or 'beta' for beta coloring
            colormm = list of min and max coloring for plot, list as follows:
                    0 = phase tensor min and max for ecolor in degrees
                    1 = phase tensor residual min and max [0,1]
                    2 = resistivity tensor coloring as resistivity on log scale
                    3 = resistivity tensor residual coloring as resistivity on 
                        linear scale
            xpad = padding of map from stations at extremities (km)
            units = 'mv' to convert to Ohm-m 
            dpi = dots per inch of figure
        """
                
        self._get_pt()
        plt.rcParams['font.size'] = self.font_size
        plt.rcParams['figure.subplot.left'] = self.subplot_left
        plt.rcParams['figure.subplot.right'] = self.subplot_right
        plt.rcParams['figure.subplot.bottom'] = self.subplot_bottom
        plt.rcParams['figure.subplot.top'] = self.subplot_top
        
        gs = gridspec.GridSpec(1, 3, hspace=self.subplot_hspace,
                               wspace=self.subplot_wspace)
                               
        font_dict = {'size':self.font_size+2, 'weight':'bold'}
        n_stations = self.data.shape[0]
        #set some local parameters
        ckmin = float(self.ellipse_range[0])
        ckmax = float(self.ellipse_range[1])
        try:
            ckstep = float(self.ellipse_range[2])
        except IndexError:
            if self.ellipse_cmap == 'mt_seg_bl2wh2rd':
                raise ValueError('Need to input range as (min, max, step)')
            else:
                ckstep = 3
        nseg = float((ckmax-ckmin)/(2*ckstep))
        
        if self.ew_limits == None:
            if self.station_east is not None:
                self.ew_limits = (np.floor(self.station_east.min())-self.pad_east, 
                                  np.ceil(self.station_east.max())+self.pad_east)
            else:
                self.ew_limits = (self.grid_east[5], self.grid_east[-5])

        if self.ns_limits == None:
            if self.station_north is not None:
                self.ns_limits = (np.floor(self.station_north.min())-self.pad_north, 
                                  np.ceil(self.station_north.max())+self.pad_north)
            else:
                self.ns_limits = (self.grid_north[5], self.grid_north[-5])
                               
        for ff, per in enumerate(self.plot_period_list):
            print 'Plotting Period: {0:.5g}'.format(per)
            fig = plt.figure('{0:.5g}'.format(per), figsize=self.fig_size,
                             dpi=self.fig_dpi)
            fig.clf()
                             
            if self.resp_fn is not None:
                axd = fig.add_subplot(gs[0, 0], aspect='equal')
                axm = fig.add_subplot(gs[0, 1], aspect='equal')
                axr = fig.add_subplot(gs[0, 2], aspect='equal')
                ax_list = [axd, axm, axr]
            
            else:
                axd = fig.add_subplot(gs[0, :], aspect='equal')
                ax_list = [axd]
            
            #plot model below the phase tensors
            if self.model_fn is not None:
                approx_depth = 500*np.sqrt(per*50)/self.dscale
                d_index = np.where(self.grid_z >= approx_depth)[0][0]
                print approx_depth, self.grid_z[d_index]
                for ax in ax_list:
                    plot_res = np.log10(self.res_model[:, :, d_index].T)
                    ax.pcolormesh(self.mesh_east,
                                   self.mesh_north, 
                                   plot_res,
                                   cmap=self.res_cmap,
                                   vmin=self.res_limits[0],
                                   vmax=self.res_limits[1])
                    
                    bb = ax.axes.get_position().bounds
                    cb_position = (bb[2]/5+bb[0], 
                                   bb[0]-self.cb_pad, .6*bb[2], .02)
                    
                
            #--> get phase tensors
            pt = mtpt.PhaseTensor(z_array=self.data[:, ff, :].reshape(n_stations,2,2))
            if self.resp is not None:
                mpt = mtpt.PhaseTensor(z_array=self.resp[:, ff, :].reshape(n_stations,2,2))
                rpt = mtpt.ResidualPhaseTensor(pt_object1=pt, pt_object2=mpt)
                rpt = rpt.residual_pt
                rcarray = np.sqrt(abs(rpt.phimin[0]*rpt.phimax[0]))
                rcmin = np.floor(rcarray.min())
                rcmax = np.floor(rcarray.max())
            
            #--> get color array
            if self.ellipse_cmap == 'mt_seg_bl2wh2rd':
                bounds = np.arange(ckmin, ckmax+ckstep, ckstep)
                nseg = float((ckmax-ckmin)/(2*ckstep))
    
            #get the properties to color the ellipses by
            if self.ellipse_colorby == 'phiminang' or \
               self.ellipse_colorby == 'phimin':
                colorarray = pt.phimin[0]
                if self.resp is not None:
                    mcarray = mpt.phimin[0]
                                        
            elif self.ellipse_colorby == 'phidet':
                 colorarray = np.sqrt(abs(pt.det[0]))*(180/np.pi)
                 if self.resp is not None:
                    mcarray = np.sqrt(abs(mpt.det[0]))*(180/np.pi)
                 
                
            elif self.ellipse_colorby == 'skew' or\
                 self.ellipse_colorby == 'skew_seg':
                colorarray = pt.beta[0]
                if self.resp is not None:
                    mcarray = mpt.beta[0]
                
            elif self.ellipse_colorby == 'ellipticity':
                colorarray = pt.ellipticity[0]
                if self.resp is not None:
                    mcarray = mpt.ellipticity[0]
                
            else:
                raise NameError(self.ellipse_colorby+' is not supported')
        
            
            #--> plot phase tensor ellipses for each stations             
            for jj in range(n_stations):
                #-----------plot data phase tensors---------------
                eheight = pt.phimin[0][jj]/pt.phimax[0].max()*self.ellipse_size
                ewidth = pt.phimax[0][jj]/pt.phimax[0].max()*self.ellipse_size
                
                ellipse = Ellipse((self.station_east[jj],
                                   self.station_north[jj]),
                                   width=ewidth,
                                   height=eheight,
                                   angle=90-pt.azimuth[0][jj])
                
                #get ellipse color
                if self.ellipse_cmap.find('seg')>0:
                    ellipse.set_facecolor(mtcl.get_plot_color(colorarray[jj],
                                                         self.ellipse_colorby,
                                                         self.ellipse_cmap,
                                                         ckmin,
                                                         ckmax,
                                                         bounds=bounds))
                else:
                    ellipse.set_facecolor(mtcl.get_plot_color(colorarray[jj],
                                                         self.ellipse_colorby,
                                                         self.ellipse_cmap,
                                                         ckmin,
                                                         ckmax))
                
                axd.add_artist(ellipse)
                if self.resp is not None:
                    #-----------plot response phase tensors---------------
                    eheight = mpt.phimin[0][jj]/mpt.phimax[0].max()*\
                              self.ellipse_size
                    ewidth = mpt.phimax[0][jj]/mpt.phimax[0].max()*\
                              self.ellipse_size
                
                    ellipsem = Ellipse((self.station_east[jj],
                                       self.station_north[jj]),
                                       width=ewidth,
                                       height=eheight,
                                       angle=90-mpt.azimuth[0][jj])
                    
                    #get ellipse color
                    if self.ellipse_cmap.find('seg')>0:
                        ellipsem.set_facecolor(mtcl.get_plot_color(mcarray[jj],
                                                             self.ellipse_colorby,
                                                             self.ellipse_cmap,
                                                             ckmin,
                                                             ckmax,
                                                             bounds=bounds))
                    else:
                        ellipsem.set_facecolor(mtcl.get_plot_color(mcarray[jj],
                                                         self.ellipse_colorby,
                                                         self.ellipse_cmap,
                                                         ckmin,
                                                         ckmax))
                    
                    axm.add_artist(ellipsem)
                    #-----------plot residual phase tensors---------------
                    eheight = rpt.phimin[0][jj]/rpt.phimax[0].max()*\
                                self.ellipse_size
                    ewidth = rpt.phimax[0][jj]/rpt.phimax[0].max()*\
                                self.ellipse_size
                
                    ellipser = Ellipse((self.station_east[jj],
                                       self.station_north[jj]),
                                       width=ewidth,
                                       height=eheight,
                                       angle=rpt.azimuth[0][jj])
                    
                    #get ellipse color
                    if self.ellipse_cmap.find('seg')>0:
                        ellipser.set_facecolor(mtcl.get_plot_color(rcarray[jj],
                                                     self.ellipse_colorby,
                                                     self.residual_cmap,
                                                     rcmin,
                                                     rcmax,
                                                     bounds=bounds))
                    else:
                        ellipser.set_facecolor(mtcl.get_plot_color(rcarray[jj],
                                                     self.ellipse_colorby,
                                                     self.residual_cmap,
                                                     rcmin,
                                                     rcmax))
                    
                    axr.add_artist(ellipser)
                
            #--> set axes properties
            # data
            axd.set_xlim(self.ew_limits)
            axd.set_ylim(self.ns_limits)
            axd.set_xlabel('Easting ({0})'.format(self.map_scale), 
                           fontdict=font_dict)
            axd.set_ylabel('Northing ({0})'.format(self.map_scale),
                           fontdict=font_dict)
            #make a colorbar ontop of axis
            bb = axd.axes.get_position().bounds
            cb_location = (3.25*bb[2]/5+bb[0], 
                            bb[3]-self.cb_pad, .295*bb[2], .02)
            cbaxd = fig.add_axes(cb_location)
            cbd = mcb.ColorbarBase(cbaxd, 
                                   cmap=mtcl.cmapdict[self.ellipse_cmap],
                                   norm=Normalize(vmin=ckmin,
                                                  vmax=ckmax),
                                   orientation='horizontal')
            cbd.ax.xaxis.set_label_position('top')
            cbd.ax.xaxis.set_label_coords(.5, 1.75)
            cbd.set_label(mtplottools.ckdict[self.ellipse_colorby])
            cbd.set_ticks(np.arange(ckmin, ckmax+self.cb_tick_step, 
                                    self.cb_tick_step))
                                    
            axd.text(self.ew_limits[0]*.95,
                     self.ns_limits[1]*.95,
                     'Data',
                     horizontalalignment='left',
                     verticalalignment='top',
                     bbox={'facecolor':'white'},
                     fontdict={'size':self.font_size+1})
                    
            #Model and residual
            if self.resp is not None:
                for aa, ax in enumerate([axm, axr]):
                    ax.set_xlim(self.ew_limits)
                    ax.set_ylim(self.ns_limits)
                    ax.set_xlabel('Easting ({0})'.format(self.map_scale), 
                                   fontdict=font_dict)
                    plt.setp(ax.yaxis.get_ticklabels(), visible=False)
                    #make a colorbar ontop of axis
                    bb = ax.axes.get_position().bounds
                    cb_location = (3.25*bb[2]/5+bb[0], 
                                    bb[3]-self.cb_pad, .295*bb[2], .02)
                    cbax = fig.add_axes(cb_location)
                    if aa == 0:
                        cb = mcb.ColorbarBase(cbax, 
                                              cmap=mtcl.cmapdict[self.ellipse_cmap],
                                               norm=Normalize(vmin=ckmin,
                                                              vmax=ckmax),
                                               orientation='horizontal')
                        cb.ax.xaxis.set_label_position('top')
                        cb.ax.xaxis.set_label_coords(.5, 1.75)
                        cb.set_label(mtplottools.ckdict[self.ellipse_colorby])
                        cb.set_ticks(np.arange(ckmin, ckmax+self.cb_tick_step, 
                                    self.cb_tick_step))
                        ax.text(self.ew_limits[0]*.95,
                                self.ns_limits[1]*.95,
                                'Model',
                                horizontalalignment='left',
                                verticalalignment='top',
                                bbox={'facecolor':'white'},
                                 fontdict={'size':self.font_size+1})
                    else:
                        cb = mcb.ColorbarBase(cbax, 
                                              cmap='Oranges',
                                               norm=Normalize(vmin=rcmin,
                                                              vmax=rcmax),
                                               orientation='horizontal')
                        cb.ax.xaxis.set_label_position('top')
                        cb.ax.xaxis.set_label_coords(.5, 1.75)
                        cb.set_label(r"$\sqrt{\Phi_{min} \Phi_{max}}$")
                        cb_ticks = np.arange(rcmin,
                                             rcmax+self.cb_residual_tick_step,
                                             self.cb_residual_tick_step)
                        cb.set_ticks(cb_ticks)
                        ax.text(self.ew_limits[0]*.95,
                                self.ns_limits[1]*.95,
                                'Residual',
                                horizontalalignment='left',
                                verticalalignment='top',
                                bbox={'facecolor':'white'},
                                fontdict={'size':self.font_size+1})
            if self.model_fn is not None:
                for ax in ax_list:
                    ax.tick_params(direction='out')
                    bb = ax.axes.get_position().bounds
                    cb_position = (2.95*bb[2]/5+bb[0], 
                                   bb[1]*1.85+self.cb_pad, .35*bb[2], .02)
                    cbax = fig.add_axes(cb_position)
                    cb = mcb.ColorbarBase(cbax, 
                                          cmap=self.res_cmap,
                                          norm=Normalize(vmin=self.res_limits[0],
                                                         vmax=self.res_limits[1]),
                                          orientation='horizontal')
                    cb.ax.xaxis.set_label_position('top')
                    cb.ax.xaxis.set_label_coords(.5, 1.5)
                    cb.set_label('Resistivity ($\Omega \cdot$m)')
                    cb_ticks = np.arange(np.floor(self.res_limits[0]), 
                                         np.ceil(self.res_limits[1]+1), 1)
                    cb.set_ticks(cb_ticks)
                    cb.set_ticklabels([mtplottools.labeldict[ctk] for ctk in cb_ticks])
                    
                            
                
            plt.show()
            
def estimate_skin_depth(resmodel, grid_z, period_list):
    """
    estimate the skin depth from the resistivity model
    
    """
    pass    
            
           
#==============================================================================
# plot slices 
#==============================================================================
class PlotSlices(object):
    """
    plot all slices and be able to scroll through the model
    
    """
    
    def __init__(self, model_fn, data_fn=None, station_fn=None, 
                 initial_fn=None, **kwargs):
        self.model_fn = model_fn
        self.data_fn = data_fn
        self.station_fn = station_fn
        self.initial_fn = initial_fn
        
        self.fig_num = kwargs.pop('fig_num', 1)
        self.fig_size = kwargs.pop('fig_size', [6, 6])
        self.fig_dpi = kwargs.pop('dpi', 300)
        self.fig_aspect = kwargs.pop('fig_aspect', 1)
        self.title = kwargs.pop('title', 'on')
        self.font_size = kwargs.pop('font_size', 7)
        
        self.subplot_wspace = .20
        self.subplot_hspace = .30
        self.subplot_right = .98
        self.subplot_left = .08
        self.subplot_top = .97
        self.subplot_bottom = .1
        
        self.index_vertical = kwargs.pop('index_vertical', 0)
        self.index_east = kwargs.pop('index_east', 0)
        self.index_north = kwargs.pop('index_north', 0)
        
        self.cmap = kwargs.pop('cmap', 'jet_r')
        self.climits = kwargs.pop('climits', (0, 4))
        
        self.map_scale = kwargs.pop('map_scale', 'km')
        #make map scale
        if self.map_scale=='km':
            self.dscale=1000.
        elif self.map_scale=='m':
            self.dscale=1. 
        self.ew_limits = kwargs.pop('ew_limits', None)
        self.ns_limits = kwargs.pop('ns_limits', None)
        self.z_limits = kwargs.pop('z_limits', None)
        
        self.res_model = None
        self.grid_east = None
        self.grid_north = None
        self.grid_z  = None
        
        self.nodes_east = None
        self.nodes_north = None
        self.nodes_z = None
        
        self.mesh_east = None
        self.mesh_north = None
        
        self.station_east = None
        self.station_north = None
        self.station_names = None
        
        self.station_id = kwargs.pop('station_id', None)
        self.station_font_size = kwargs.pop('station_font_size', 8)
        self.station_font_pad = kwargs.pop('station_font_pad', 1.0)
        self.station_font_weight = kwargs.pop('station_font_weight', 'bold')
        self.station_font_rotation = kwargs.pop('station_font_rotation', 60)
        self.station_font_color = kwargs.pop('station_font_color', 'k')
        self.station_marker = kwargs.pop('station_marker', 
                                         r"$\blacktriangledown$")
        self.station_color = kwargs.pop('station_color', 'k')
        self.ms = kwargs.pop('ms', 10)
        
        self.plot_yn = kwargs.pop('plot_yn', 'y')
        if self.plot_yn == 'y':
            self.plot()
        
        
    def read_files(self):
        """
        read in the files to get appropriate information
        """
        #--> read in model file
        if self.model_fn is not None:
            if os.path.isfile(self.model_fn) == True:
                wsmodel = WSModel(self.model_fn)
                self.res_model = wsmodel.res_model
                self.grid_east = wsmodel.grid_east/self.dscale
                self.grid_north = wsmodel.grid_north/self.dscale
                self.grid_z = wsmodel.grid_z/self.dscale
                self.nodes_east = wsmodel.nodes_east/self.dscale
                self.nodes_north = wsmodel.nodes_north/self.dscale
                self.nodes_z = wsmodel.nodes_z/self.dscale
            else:
                raise mtex.MTpyError_file_handling(
                        '{0} does not exist, check path'.format(self.model_fn))
        
        #--> read in data file to get station locations
        if self.data_fn is not None:
            if os.path.isfile(self.data_fn) == True:
                wsdata = WSData()
                wsdata.read_data_file(self.data_fn)
                self.station_east = wsdata.data['east']/self.dscale
                self.station_north = wsdata.data['north']/self.dscale
                self.station_names = wsdata.data['station']
            else:
                print 'Could not find data file {0}'.format(self.data_fn)
            
        #--> read in station file
        if self.station_fn is not None:
            if os.path.isfile(self.station_fn) == True:
                wsstations = WSStation(self.station_fn)
                wsstations.read_station_file()
                self.station_east = wsstations.east/self.dscale
                self.station_north = wsstations.north/self.dscale
                self.station_names = wsstations.names
            else:
                print 'Could not find station file {0}'.format(self.station_fn)
        
        #--> read in initial file
        if self.initial_fn is not None:
            if os.path.isfile(self.initial_fn) == True:
                wsmesh = WSMesh()
                wsmesh.read_initial_file(self.initial_fn)
                self.grid_east = wsmesh.grid_east/self.dscale
                self.grid_north = wsmesh.grid_north/self.dscale
                self.grid_z = wsmesh.grid_z/self.dscale
                self.nodes_east = wsmesh.nodes_east/self.dscale
                self.nodes_north = wsmesh.nodes_north/self.dscale
                self.nodes_z = wsmesh.nodes_z/self.dscale
                
                #need to convert index values to resistivity values
                rdict = dict([(ii,res) for ii,res in enumerate(wsmesh.res_list,1)])
                
                for ii in range(len(wsmesh.res_list)):
                    self.res_model[np.where(wsmesh.res_model==ii+1)] = \
                                                                    rdict[ii+1]
            else:
                raise mtex.MTpyError_file_handling(
                     '{0} does not exist, check path'.format(self.initial_fn))
        
        if self.initial_fn is None and self.model_fn is None:
            raise mtex.MTpyError_inputarguments('Need to input either a model'
                                                ' file or initial file.')
        
    def plot(self):
        """
        plot:
            east vs. vertical,
            north vs. vertical,
            east vs. north
            
        
        """
        
        self.read_files()
        
        self.get_station_grid_locations()
        
        self.font_dict = {'size':self.font_size+2, 'weight':'bold'}
        #set the limits of the plot
        if self.ew_limits == None:
            if self.station_east is not None:
                self.ew_limits = (np.floor(self.station_east.min()), 
                                  np.ceil(self.station_east.max()))
            else:
                self.ew_limits = (self.grid_east[5], self.grid_east[-5])

        if self.ns_limits == None:
            if self.station_north is not None:
                self.ns_limits = (np.floor(self.station_north.min()), 
                                  np.ceil(self.station_north.max()))
            else:
                self.ns_limits = (self.grid_north[5], self.grid_north[-5])
        
        if self.z_limits == None:
                self.z_limits = (self.grid_z[0]-5000/self.dscale, 
                                 self.grid_z[-5])
            
        
        self.fig = plt.figure(self.fig_num, figsize=self.fig_size,
                              dpi=self.fig_dpi)
        plt.clf()
        gs = gridspec.GridSpec(2, 2,
                               wspace=self.subplot_wspace,
                               left=self.subplot_left,
                               top=self.subplot_top,
                               bottom=self.subplot_bottom, 
                               right=self.subplot_right, 
                               hspace=self.subplot_hspace)        
        
        #make subplots
        self.ax_ez = self.fig.add_subplot(gs[0, 0], aspect=self.fig_aspect)
        self.ax_nz = self.fig.add_subplot(gs[1, 1], aspect=self.fig_aspect)
        self.ax_en = self.fig.add_subplot(gs[1, 0], aspect=self.fig_aspect)
        self.ax_map = self.fig.add_subplot(gs[0, 1])
        
        #make grid meshes being sure the indexing is correct
        self.mesh_ez_east, self.mesh_ez_vertical = np.meshgrid(self.grid_east,
                                                               self.grid_z,
                                                               indexing='ij') 
        self.mesh_nz_north, self.mesh_nz_vertical = np.meshgrid(self.grid_north,
                                                                self.grid_z,
                                                                indexing='ij') 
        self.mesh_en_east, self.mesh_en_north = np.meshgrid(self.grid_east, 
                                                            self.grid_north,
                                                            indexing='ij')
                                                            
        #--> plot east vs vertical
        self._update_ax_ez()
        
        #--> plot north vs vertical
        self._update_ax_nz()
                              
        #--> plot east vs north
        self._update_ax_en()
                                 
        #--> plot the grid as a map view 
        self._update_map()
        
        #plot color bar
        cbx = mcb.make_axes(self.ax_map, fraction=.15, shrink=.75, pad = .1)
        cb = mcb.ColorbarBase(cbx[0],
                              cmap=self.cmap,
                              norm=Normalize(vmin=self.climits[0],
                                             vmax=self.climits[1]))

   
        cb.ax.yaxis.set_label_position('right')
        cb.ax.yaxis.set_label_coords(1.25,.5)
        cb.ax.yaxis.tick_left()
        cb.ax.tick_params(axis='y',direction='in')
                            
        cb.set_label('Resistivity ($\Omega \cdot$m)',
                     fontdict={'size':self.font_size+1})
                     
        cb.set_ticks(np.arange(np.ceil(self.climits[0]),
                               np.floor(self.climits[1]+1)))
        cblabeldict={-2:'$10^{-3}$',-1:'$10^{-1}$',0:'$10^{0}$',1:'$10^{1}$',
                     2:'$10^{2}$',3:'$10^{3}$',4:'$10^{4}$',5:'$10^{5}$',
                     6:'$10^{6}$',7:'$10^{7}$',8:'$10^{8}$'}
        cb.set_ticklabels([cblabeldict[cc] 
                            for cc in np.arange(np.ceil(self.climits[0]),
                                                np.floor(self.climits[1]+1))])
                   
        plt.show()
        
        self.key_press = self.fig.canvas.mpl_connect('key_press_event',
                                                     self.on_key_press)


    def on_key_press(self, event):
        """
        on a key press change the slices
        
        """                                                            

        key_press = event.key
        
        if key_press == 'n':
            if self.index_north == self.grid_north.shape[0]:
                print 'Already at northern most grid cell'
            else:
                self.index_north += 1
                if self.index_north > self.grid_north.shape[0]:
                    self.index_north = self.grid_north.shape[0]
            self._update_ax_ez()
            self._update_map()
       
        if key_press == 'm':
            if self.index_north == 0:
                print 'Already at southern most grid cell'
            else:
                self.index_north -= 1 
                if self.index_north < 0:
                    self.index_north = 0
            self._update_ax_ez()
            self._update_map()
                    
        if key_press == 'e':
            if self.index_east == self.grid_east.shape[0]:
                print 'Already at eastern most grid cell'
            else:
                self.index_east += 1
                if self.index_east > self.grid_east.shape[0]:
                    self.index_east = self.grid_east.shape[0]
            self._update_ax_nz()
            self._update_map()
       
        if key_press == 'w':
            if self.index_east == 0:
                print 'Already at western most grid cell'
            else:
                self.index_east -= 1 
                if self.index_east < 0:
                    self.index_east = 0
            self._update_ax_nz()
            self._update_map()
                    
        if key_press == 'd':
            if self.index_vertical == self.grid_z.shape[0]:
                print 'Already at deepest grid cell'
            else:
                self.index_vertical += 1
                if self.index_vertical > self.grid_z.shape[0]:
                    self.index_vertical = self.grid_z.shape[0]
            self._update_ax_en()
            print 'Depth = {0:.5g} ({1})'.format(self.grid_z[self.index_vertical],
                                                 self.map_scale)
       
        if key_press == 'u':
            if self.index_vertical == 0:
                print 'Already at surface grid cell'
            else:
                self.index_vertical -= 1 
                if self.index_vertical < 0:
                    self.index_vertical = 0
            self._update_ax_en()
            print 'Depth = {0:.5gf} ({1})'.format(self.grid_z[self.index_vertical],
                                                 self.map_scale)
                    
    def _update_ax_ez(self):
        """
        update east vs vertical plot
        """
        self.ax_ez.cla()
        plot_ez = np.log10(self.res_model[self.index_north, :, :]) 
        self.ax_ez.pcolormesh(self.mesh_ez_east,
                              self.mesh_ez_vertical, 
                              plot_ez,
                              cmap=self.cmap,
                              vmin=self.climits[0],
                              vmax=self.climits[1])
        #plot stations
        for sx in self.station_dict_north[self.grid_north[self.index_north]]:
            self.ax_ez.text(sx,
                            0,
                            self.station_marker,
                            horizontalalignment='center',
                            verticalalignment='baseline',
                            fontdict={'size':self.ms,
                                      'color':self.station_color})
                                      
        self.ax_ez.set_xlim(self.ew_limits)
        self.ax_ez.set_ylim(self.z_limits[1], self.z_limits[0])
        self.ax_ez.set_ylabel('Depth ({0})'.format(self.map_scale),
                              fontdict=self.font_dict)
        self.ax_ez.set_xlabel('Easting ({0})'.format(self.map_scale),
                              fontdict=self.font_dict)
        self.fig.canvas.draw()
        self._update_map()
        
    def _update_ax_nz(self):
        """
        update east vs vertical plot
        """
        self.ax_nz.cla()
        plot_nz = np.log10(self.res_model[:, self.index_east, :]) 
        self.ax_nz.pcolormesh(self.mesh_nz_north,
                              self.mesh_nz_vertical, 
                              plot_nz,
                              cmap=self.cmap,
                              vmin=self.climits[0],
                              vmax=self.climits[1])
        #plot stations
        for sy in self.station_dict_east[self.grid_east[self.index_east]]:
            self.ax_nz.text(sy,
                            0,
                            self.station_marker,
                            horizontalalignment='center',
                            verticalalignment='baseline',
                            fontdict={'size':self.ms,
                                      'color':self.station_color})
        self.ax_nz.set_xlim(self.ns_limits)
        self.ax_nz.set_ylim(self.z_limits[1], self.z_limits[0])
        self.ax_nz.set_xlabel('Northing ({0})'.format(self.map_scale),
                              fontdict=self.font_dict)
        self.ax_nz.set_ylabel('Depth ({0})'.format(self.map_scale),
                              fontdict=self.font_dict)
        self.fig.canvas.draw()
        self._update_map()
        
    def _update_ax_en(self):
        """
        update east vs vertical plot
        """
        
        self.ax_en.cla()
        plot_en = np.log10(self.res_model[:, :, self.index_vertical].T) 
        self.ax_en.pcolormesh(self.mesh_en_east,
                              self.mesh_en_north, 
                              plot_en,
                              cmap=self.cmap,
                              vmin=self.climits[0],
                              vmax=self.climits[1])
        self.ax_en.set_xlim(self.ew_limits)
        self.ax_en.set_ylim(self.ns_limits)
        self.ax_en.set_ylabel('Northing ({0})'.format(self.map_scale),
                              fontdict=self.font_dict)
        self.ax_en.set_xlabel('Easting ({0})'.format(self.map_scale),
                              fontdict=self.font_dict)
        #--> plot the stations
        if self.station_east is not None:
            for ee, nn in zip(self.station_east, self.station_north):
                self.ax_en.text(ee, nn, '*', 
                                 verticalalignment='center',
                                 horizontalalignment='center',
                                 fontdict={'size':5, 'weight':'bold'})

        self.fig.canvas.draw()
        self._update_map()
        
    def _update_map(self):
        self.ax_map.cla()
        self.east_line_xlist = []
        self.east_line_ylist = []            
        for xx in self.grid_east:
            self.east_line_xlist.extend([xx, xx])
            self.east_line_xlist.append(None)
            self.east_line_ylist.extend([self.grid_north.min(), 
                                         self.grid_north.max()])
            self.east_line_ylist.append(None)
        self.ax_map.plot(self.east_line_xlist,
                         self.east_line_ylist,
                         lw=.25,
                         color='k')

        self.north_line_xlist = []
        self.north_line_ylist = [] 
        for yy in self.grid_north:
            self.north_line_xlist.extend([self.grid_east.min(),
                                          self.grid_east.max()])
            self.north_line_xlist.append(None)
            self.north_line_ylist.extend([yy, yy])
            self.north_line_ylist.append(None)
        self.ax_map.plot(self.north_line_xlist,
                         self.north_line_ylist,
                         lw=.25,
                         color='k')
        #--> e-w indication line 
        self.ax_map.plot([self.grid_east.min(), 
                          self.grid_east.max()],
                         [self.grid_north[self.index_north], 
                          self.grid_north[self.index_north]],
                         lw=1,
                         color='g')
                         
        #--> e-w indication line 
        self.ax_map.plot([self.grid_east[self.index_east], 
                          self.grid_east[self.index_east]],
                         [self.grid_north.min(), 
                          self.grid_north.max()],
                         lw=1,
                         color='b')
         #--> plot the stations
        if self.station_east is not None:
            for ee, nn in zip(self.station_east, self.station_north):
                self.ax_map.text(ee, nn, '*', 
                                 verticalalignment='center',
                                 horizontalalignment='center',
                                 fontdict={'size':5, 'weight':'bold'})
                                 
        self.ax_map.set_xlim(self.ew_limits)
        self.ax_map.set_ylim(self.ns_limits)
        self.ax_map.set_ylabel('Northing ({0})'.format(self.map_scale),
                              fontdict=self.font_dict)
        self.ax_map.set_xlabel('Easting ({0})'.format(self.map_scale),
                              fontdict=self.font_dict)
        
        #plot stations                      
        self.ax_map.text(self.ew_limits[0]*.95, self.ns_limits[1]*.95,
                 '{0:.5g} ({1})'.format(self.grid_z[self.index_vertical],
                                        self.map_scale),
                 horizontalalignment='left',
                 verticalalignment='top',
                 bbox={'facecolor': 'white'},
                 fontdict=self.font_dict)
        
        
        self.fig.canvas.draw()
        
    def get_station_grid_locations(self):
        """
        get the grid line on which a station resides for plotting
        
        """
        self.station_dict_east = dict([(gx, []) for gx in self.grid_east])
        self.station_dict_north = dict([(gy, []) for gy in self.grid_north])
        if self.station_east is not None:
            for ss, sx in enumerate(self.station_east):
                gx = np.where(self.grid_east <= sx)[0][-1]
                self.station_dict_east[self.grid_east[gx]].append(self.station_north[ss])
            
            for ss, sy in enumerate(self.station_north):
                gy = np.where(self.grid_north <= sy)[0][-1]
                self.station_dict_north[self.grid_north[gy]].append(self.station_east[ss])
        else:
            return 
                  
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
            
    def save_figure(self, save_fn=None, fig_dpi=None, file_format='pdf', 
                    orientation='landscape', close_fig='y'):
        """
        save_figure will save the figure to save_fn.
        
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
            save_fn = os.path.join(save_fn, '_E{0}_N{1}_Z{2}.{3}'.format(
                                    self.index_east, self.index_north,
                                    self.index_vertical, file_format))
            self.fig.savefig(save_fn, dpi=fig_dpi, format=file_format,
                        orientation=orientation, bbox_inches='tight')
        
        if close_fig == 'y':
            plt.clf()
            plt.close(self.fig)
        
        else:
            pass
        
        self.fig_fn = save_fn
        print 'Saved figure to: '+self.fig_fn
        
        
        
        
        
        
        
        
        