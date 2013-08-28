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
        linelst = []
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
            linelst.append(sdict)
        
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
        
#        #write out places where errors are larger than error tolerance
#        errfid = file(os.path.join(os.path.dirname(self.data_fn),
#                                   'DataErrorLocations.txt'), 'w')
#        errfid.write('Errors larger than error tolerance of: \n')
#        errfid.write('Zxx={0} Zxy={1} Zyx={2} Zyy={3} \n'.format(zerrmap[0]*zerr,
#                     zerrmap[1]*zerr, zerrmap[2]*zerr, zerrmap[3]*zerr))
#        errfid.write('-'*20+'\n')
#        errfid.write('station  T=period(s) Zij err=percentage \n')
#        for pfdict in linelst:
#            for kk, ff in enumerate(pfdict['fspot']):
#                if pfdict['fspot'][ff][2]>zerr*100*zerrmap[0]:
#                    errfid.write(pfdict['station']+'  T='+ff+\
#                            ' Zxx err={0:.3f} \n'.format(pfdict['fspot'][ff][2])) 
#                if pfdict['fspot'][ff][3]>zerr*100*zerrmap[1]:
#                    errfid.write(pfdict['station']+'  T='+ff+\
#                            ' Zxy err={0:.3f} \n'.format(pfdict['fspot'][ff][3])) 
#                if pfdict['fspot'][ff][4]>zerr*100*zerrmap[2]:
#                    errfid.write(pfdict['station']+'  T='+ff+\
#                            ' Zyx err={0:.3f} \n'.format(pfdict['fspot'][ff][4]))
#                if pfdict['fspot'][ff][5]>zerr*100*zerrmap[3]:
#                    errfid.write(pfdict['station']+'  T='+ff+\
#                            ' Zyy err={0:.3f} \n'.format(pfdict['fspot'][ff][5])) 
#        errfid.close()
#        print 'Wrote errors lager than tolerance to: '
#        print os.path.join(os.path.dirname(self.data_fn),
#                           'DataErrorLocations.txt')
                           
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
        
        findlst = []
        for ii, dline in enumerate(dlines[1:50], 1):
            if dline.find('Station_Location: N-S') == 0:
                findlst.append(ii)
            elif dline.find('Station_Location: E-W') == 0:
                findlst.append(ii)
            elif dline.find('DATA_Period:') == 0:
                findlst.append(ii)
                
        ncol = len(dlines[nsstart].strip().split())
        
        #get site names if entered a sites file
        if wl_sites_fn != None:
            self.wl_site_fn = wl_sites_fn
            slst, station_list = wl.read_sites_file(self.wl_sites_fn)
            self.data['station'] = station_list
        
        elif station_fn != None:
            self.station_fn = station_fn
            self.read_station_file(self.station_fn)
            self.data['station'] = self.station_locations['station']
        else:
            self.data['station'] = np.arange(n_stations)
            
    
        #get N-S locations
        for ii, dline in enumerate(dlines[findlst[0]+1:findlst[1]],0):
            dline = dline.strip().split()
            for jj in range(ncol):
                try:
                    self.data['north'][ii*ncol+jj] = float(dline[jj])
                except IndexError:
                    pass
                except ValueError:
                    break
                
        #get E-W locations
        for ii, dline in enumerate(dlines[findlst[1]+1:findlst[2]],0):
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
        for ii, dl in enumerate(dlines[findlst[2]:]):
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

class WSMeshGrid(object):
    """
    make and read a FE mesh grid
    """
    
    def __init__(self):
        pass
    
    def make3DGrid(edilst, xspacing=500, yspacing=500, z1layer=10, xpad=5,
                   ypad=5, zpad=5, xpadroot=5, ypadroot=5, zpadroot=2, 
                   zpadpow=(5,15), nz=30, plotyn='y', plotxlimits=None, 
                    plotylimits=None, plotzlimits=None):
        """
        makes a grid from the edifiles to go into wsinv3d.  The defaults usually
        work relatively well, but it might take some effort to get a desired grid.
    
        Inputs:
        --------
            **edilst** : list
                         list of full paths to the .edi files to be included in 
                         the inversion.
            
            **xspacing** : float
                           spacing of cells in the east-west direction in meters.
                           *default* is 500 (m)
                           
            **yspacing** : float
                           spacing of cells in the north-south direction in meters.
                           *default* is 500 (m)
                           
            **z1layer** : float
                          the depth of the first layer in the model in meters.  
                          This is usually about 1/10th of your shallowest skin 
                          depth.
                          *default* is 10 (m)
                          
            **xpad** : int
                       number of cells to pad on either side in the east-west 
                       direction.  The width of these cells grows exponentially 
                       to the edge.
                       *default* is 5
                          
            **ypad** : int
                       number of cells to pad on either side in the north-south 
                       direction.  The width of these cells grows exponentially 
                       to the edge.
                       *default* is 5
                          
            **zpad** : int
                       number of cells to pad on either side in the vertical 
                       direction.  This is to pad beneath the depth of 
                       investigation and grows faster exponentially than the zone 
                       of study.  The purpose is to decrease the number of cells
                       in the model.
                       *default* is 5
                       
            **xpadroot** : float
                           the root number that is multiplied to itself for 
                           calculating the width of the padding cells in the 
                           east-west direction.
                           *default* is 5
                       
            **ypadroot** : float
                           the root number that is multiplied to itself for 
                           calculating the width of the padding cells in the 
                           north-south direction.
                           *default* is 5
                           
            **zpadroot** : float
                           the root number that is multiplied to itself for 
                           calculating the width of the padding cells in the 
                           vertical direction.
                           *default* is 2
                           
            **zpadpow** : tuple (min,max)
                          the power to which zpadroot is raised for the padding
                          cells in the vertical direction.  Input as a tuple with
                          minimum power and maximum power.
                          *default* is (5,15)
                          
            **nz** : int
                     number of layers in the vertical direction.  Remember that 
                     the inversion code automatically adds 7 air layers to the 
                     model which need to be used when estimating the memory that
                     it is going to take to run the model.
                     *default* is 30
                     
            **plotyn** : [ 'y' | 'n' ]
                         if plotyn=='y' then a plot showing map view (east:north)
                         and a cross sectional view (east:vertical) plane                     
                         
                         * 'y' to plot the grid with station locations
                         
                         * 'n' to suppress the plotting.
                        
            **plotxlimits** : tuple (xmin,xmax)
                             plot min and max distances in meters for the east-west 
                             direction.  If not input, the xlimits will be set to 
                             the furthest stations east and west.
                             *default* is None
                        
            **plotylimits** : tuple (ymin,ymax)
                             plot min and max distances in meters for the east-west 
                             direction. If not input, the ylimits will be set to 
                             the furthest stations north and south.
                             *default* is None
                        
            **plotzlimits** : tuple (zmin,zmax)
                             plot min and max distances in meters for the east-west 
                             direction.  If not input, the zlimits will be set to 
                             the nz layer and 0.
                             *default* is None
                             
        Returns:
        --------
            **xgrid** : np.array
                        array of the east-west cell locations  
                        
            **ygrid** : np.array
                        array of the north-south cell locations
                        
            **zgrid** : np.array
                        array of the vertical cell locations 
                        
            **locations** : np.array (ns,2)
                            array of station locations placed in the center of 
                            the cells. 
                            * column 1 is for east-west locations
                            * column 2 is for the north-south location
                            
            **slst** : list
                       list of dictionaries for each station with keys:
                           * *'station'* for the station name
                           * *'east'* for easting in model coordinates
                           * *'east_c'* for easting in model coordinates to place 
                                        the station at the center of the cell 
                           * *'north'* for northing in model coordinates
                           * *'north_c'* for northing in model coordinates to place 
                                        the station at the center of the cell 
                            
                           
        :Example: ::
            
            >>> import mtpy.modeling.ws3dtools as ws
            >>> import os
            >>> edipath=r"/home/edifiles"
            >>> edilst=[os.path.join(edipath,edi) for os.listdir(edipath)]
            >>> xg,yg,zg,loc,statlst=ws.make3DGrid(edilst,plotzlimits=(-2000,200))
        
        """
        ns = len(edilst)
        slst = np.zeros(ns, dtype=[('station','|S10'), ('east', np.float),
                                   ('north', np.float), ('east_c', np.float),
                                   ('north_c', np.float)])
        for ii,edi in enumerate(edilst):
            zz = mtedi.Edi()
            zz.readfile(edi)
            zone, east, north = ll2utm.LLtoUTM(23, zz.lat, zz.lon)
            slst[ii]['station'] = zz.station
            slst[ii]['east'] = east
            slst[ii]['north'] = north
        
        #estimate the mean distance to  get into relative coordinates
        xmean = slst['east'].mean()
        ymean = slst['north'].mean()
         
        #remove the average distance to get coordinates in a relative space
        slst['east'] -= xmean
        slst['north'] -= ymean
     
        #translate the stations so they are relative to 0,0
        xcenter = (slst['east'].max()-np.abs(slst['east'].min()))/2
        ycenter = (slst['north'].max()-np.abs(slst['north'].min()))/2
        
        #remove the average distance to get coordinates in a relative space
        slst['east'] -= xcenter
        slst['north'] -= ycenter
    
        #pickout the furtherst south and west locations 
        #and put that station as the bottom left corner of the main grid
        xleft = slst['east'].min()-xspacing/2
        xright = slst['east'].max()+xspacing/2
        ybottom = slst['north'].min()-yspacing/2
        ytop = slst['north'].max()+yspacing/2
    
        #---make a grid around the stations from the parameters above---
        #make grid in east-west direction
        midxgrid = np.arange(start=xleft,stop=xright+xspacing,
                             step=xspacing)
        xpadleft = np.round(-xspacing*xpadroot**np.arange(start=.5,
                                                          stop=3,
                                                          step=3./xpad))+xleft
        xpadright = np.round(xspacing*xpadroot**np.arange(start=.5,
                                                          stop=3,
                                                          step=3./xpad))+xright
        xgridr = np.append(np.append(xpadleft[::-1], midxgrid), xpadright)
        
        #make grid in north-south direction 
        midygrid = np.arange(start= ybottom, stop=ytop+yspacing, step=yspacing)
        ypadbottom = np.round(-yspacing*ypadroot**np.arange(start=.5,
                                                            stop=3,
                                                            step=3./ypad))+ybottom
        ypadtop = np.round(yspacing*ypadroot**np.arange(start=.5,
                                                        stop=3,
                                                        step=3./ypad))+ytop
        ygridr = np.append(np.append(ypadbottom[::-1], midygrid), ypadtop)
        
        
        #make depth grid
        zgrid1 = z1layer*zpadroot**np.round(np.arange(0,zpadpow[0],
                                               zpadpow[0]/(nz-float(zpad))))
        zgrid2 = z1layer*zpadroot**np.round(np.arange(zpadpow[0],zpadpow[1],
                                             (zpadpow[1]-zpadpow[0])/(zpad)))
        
        zgrid = np.append(zgrid1, zgrid2)
        
        #--Need to make an array of the individual cell dimensions for the wsinv3d
        xnodes = xgridr.copy()    
        nx = xgridr.shape[0]
        xnodes[:nx/2] = np.array([abs(xgridr[ii]-xgridr[ii+1]) 
                                for ii in range(int(nx/2))])
        xnodes[nx/2:] = np.array([abs(xgridr[ii]-xgridr[ii+1]) 
                                for ii in range(int(nx/2)-1,nx-1)])
    
        ynodes = ygridr.copy()
        ny = ygridr.shape[0]
        ynodes[:ny/2] = np.array([abs(ygridr[ii]-ygridr[ii+1]) 
                                for ii in range(int(ny/2))])
        ynodes[ny/2:] = np.array([abs(ygridr[ii]-ygridr[ii+1]) 
                                for ii in range(int(ny/2)-1,ny-1)])
                                
        #--put the grids into coordinates relative to the center of the grid
        xgrid = xnodes.copy()
        xgrid[:int(nx/2)] = -np.array([xnodes[ii:int(nx/2)].sum() 
                                        for ii in range(int(nx/2))])
        xgrid[int(nx/2):] = np.array([xnodes[int(nx/2):ii+1].sum() 
                                for ii in range(int(nx/2),nx)])-xnodes[int(nx/2)]
                                
        ygrid = ynodes.copy()
        ygrid[:int(ny/2)] = -np.array([ynodes[ii:int(ny/2)].sum() 
                                        for ii in range(int(ny/2))])
        ygrid[int(ny/2):] = np.array([ynodes[int(ny/2):ii+1].sum() 
                                for ii in range(int(ny/2),ny)])-ynodes[int(ny/2)]
                                
                                
        #make sure that the stations are in the center of the cell as requested by
        #the code.
        for ii in range(ns):
            #look for the closest grid line
            xx = [nn for nn,xf in enumerate(xgrid) if xf>(slst[ii]['east']-xspacing) 
                and xf<(slst[ii]['east']+xspacing)]
            
            #shift the station to the center in the east-west direction
            if xgrid[xx[0]] < slst[ii]['east']:
                slst[ii]['east_c'] = xgrid[xx[0]]+xspacing/2
            elif xgrid[xx[0]] > slst[ii]['east']:
                slst[ii]['east_c'] = xgrid[xx[0]]-xspacing/2
            
            #look for closest grid line
            yy = [mm for mm,yf in enumerate(ygrid) 
                  if yf >(slst[ii]['north']-yspacing) 
                  and yf<(slst[ii]['north']+yspacing)]
            
            #shift station to center of cell in north-south direction
            if ygrid[yy[0]] < slst[ii]['north']:
                slst[ii]['north_c'] = ygrid[yy[0]]+yspacing/2
            elif ygrid[yy[0]] > slst[ii]['north']:
                slst[ii]['north_c'] = ygrid[yy[0]]-yspacing/2
                
            
        #=Plot the data if desired=========================
        if plotyn == 'y':
            fig = plt.figure(1,figsize=[6,6],dpi=300)
            plt.clf()
            
            #---plot map view    
            ax1 = fig.add_subplot(1,2,1,aspect='equal')
            
            #make sure the station is in the center of the cell
            ax1.scatter(slst['east_c'], slst['north_c'], marker='v')
                    
            for xp in xgrid:
                ax1.plot([xp,xp],[ygrid.min(),ygrid.max()],color='k')
                
            for yp in ygrid:
                ax1.plot([xgrid.min(),xgrid.max()],[yp,yp],color='k')
            
            if plotxlimits == None:
                ax1.set_xlim(slst['east'].min()-10*xspacing,
                             slst['east'].max()+10*xspacing)
            else:
                ax1.set_xlim(plotxlimits)
            
            if plotylimits == None:
                ax1.set_ylim(slst['north'].min()-50*yspacing,
                             slst['north'].max()+50*yspacing)
            else:
                ax1.set_ylim(plotylimits)
                
            ax1.set_ylabel('Northing (m)',fontdict={'size':10,'weight':'bold'})
            ax1.set_xlabel('Easting (m)',fontdict={'size':10,'weight':'bold'})
            
            ##----plot depth view
            ax2 = fig.add_subplot(1,2,2,aspect='auto')
                    
            for xp in xgrid:
                ax2.plot([xp,xp],[-zgrid.sum(),0],color='k')
                
            ax2.scatter(slst['east_c'], [0]*ns, marker='v')
                
            for zz,zp in enumerate(zgrid):
                ax2.plot([xgrid.min(),xgrid.max()],[-zgrid[0:zz].sum(),
                          -zgrid[0:zz].sum()],color='k')
            
            if plotzlimits == None:
                ax2.set_ylim(-zgrid1.max(),200)
            else:
                ax2.set_ylim(plotzlimits)
                
            if plotxlimits == None:
                ax2.set_xlim(slst['east'].min()-xspacing,
                             slst['east'].max()+xspacing)
            else:
                ax2.set_xlim(plotxlimits)
                
            ax2.set_ylabel('Depth (m)', fontdict={'size':10, 'weight':'bold'})
            ax2.set_xlabel('Easting (m)', fontdict={'size':10, 'weight':'bold'})  
            
            plt.show()
        
    
        
        
        print '-'*15
        print '   Number of stations = {0}'.format(len(slst))
        print '   Dimensions: '
        print '      e-w = {0}'.format(xgrid.shape[0])
        print '      n-s = {0}'.format(ygrid.shape[0])
        print '       z  = {0} (without 7 air layers)'.format(zgrid.shape[0])
        print '   Extensions: '
        print '      e-w = {0:.1f} (m)'.format(xnodes.__abs__().sum())
        print '      n-s = {0:.1f} (m)'.format(ynodes.__abs__().sum())
        print '      0-z = {0:.1f} (m)'.format(zgrid.__abs__().sum())
        print '-'*15
        
        loc = np.array([slst['east_c'], slst['north_c']])
        return ynodes, xnodes, zgrid, loc.T, slst            

class WSModel(WSData):
    """
    included tools for making a model, reading a model and plotting a model.
    
    """
    pass
    
class WSInputs(object):
    """
    includes tools for writing input files
    """
    
    pass
