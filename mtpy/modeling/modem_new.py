#!/usr/bin/env python
"""
==================
ModEM
==================


# Generate data file for ModEM
# by Paul Soeffky 2013
# revised by LK 2014

"""

import os
import mtpy.core.edi as mtedi
import mtpy.core.z as mtz
import mtpy.core.mt as mt
import numpy as np
import mtpy.utils.latlongutmconversion as utm2ll
import mtpy.modeling.ws3dinv as ws
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Ellipse
from matplotlib.colors import Normalize
import matplotlib.colorbar as mcb
import matplotlib.gridspec as gridspec
import mtpy.imaging.mtplottools as mtplottools
import matplotlib.widgets as widgets
import matplotlib.colors as colors
import matplotlib.cm as cm
import mtpy.modeling.winglink as wl
import mtpy.utils.exceptions as mtex
import mtpy.analysis.pt as mtpt
import mtpy.imaging.mtcolors as mtcl

#==============================================================================

class Data(object):
    """
    read and write .dat files for ModEM
    
    
    """
    
    def __init__(self, edi_list=None, **kwargs):
        self.edi_list = edi_list
        
        self.merge_periods = kwargs.pop('merge_periods', False)
        self.merge_threshold = kwargs.pop('merge_threshold', 5.0)
        self.error_floor = kwargs.pop('error_floor', 5.0)
        
        self.wave_sign = kwargs.pop('wave_sign', '+')
        self.units = kwargs.pop('units', '[mV/km]/[nT]')
        self.inv_mode = kwargs.pop('inv_mode', '1')
        self.period_list = kwargs.pop('period_list', None)
        self.period_step = kwargs.pop('period_step', 1)
        self.period_min = kwargs.pop('period_min', None)
        self.period_max = kwargs.pop('period_max', None)
        self.max_num_periods = kwargs.pop('max_num_periods', None)
        self.period_dict = None
        self.data_period_list = None
        
        self.fn_basename = kwargs.pop('fn_basename', 'ModEM_Data.dat')
        self.save_path = kwargs.pop('save_path', os.getcwd())
        
        self.coord_array = None
        self.center_position = (0.0, 0.0, 0.0)
        self.edi_obj_list = None
        self.data_array = None
        
        self.inv_mode_dict = {'1':['Full_Impedance', 'Full_Vertical_Components'],
                              '2':['Full_Impedance'],
                              '3':['Off_Diagonal_Impedance', 
                                   'Full_Vertical_Components'],
                              '4':['Off_Diagonal_Impedance'],
                              '5':['Full_Vertical_Components'],
                              '6':['Full_Interstation_TF'],
                              '7':['Off_Diagonal_Rho_Phase']}
        self.inv_comp_dict = {'Full_Impedance':['zxx', 'zxy', 'zyx', 'zyy'],
                              'Off_Diagonal_Impedance':['zxy', 'zyx'],
                              'Full_Vertical_Components':['tx', 'ty']}
        
                              
        self.header_strings = \
        ['# Created using MTpy error floor {0:.0f}%\n'.format(self.error_floor), 
        '# Period(s) Code GG_Lat GG_Lon X(m) Y(m) Z(m) Component Real Imag Error\n']

        
    def get_station_locations(self):
        """
        get station locations from edi files
        """
        self.edi_obj_list = []
        if self.edi_list is None:
            raise ModEMError('edi_list is None, enter a list of .edi files')

        #--> read in .edi files and get position information as well as center
        #    station. 
        ns = len(self.edi_list)
        self.coord_array = np.zeros(ns, dtype=[('station','|S10'),
                                               ('east', np.float),
                                               ('north', np.float),
                                               ('lat', np.float),
                                               ('lon', np.float),
                                               ('elev', np.float),
                                               ('rel_east', np.float),
                                               ('rel_north', np.float),
                                               ('rel_east_c', np.float),
                                               ('rel_north_c', np.float)])
                                        
        for ii, edi in enumerate(self.edi_list):
            edi_obj = mtedi.Edi(edi)
            self.edi_obj_list.append(edi_obj)
            zone, east, north = utm2ll.LLtoUTM(23, edi_obj.lat, edi_obj.lon)
            self.coord_array[ii]['station'] = edi_obj.station
            self.coord_array[ii]['lat'] = float(edi_obj.lat)
            self.coord_array[ii]['lon'] = float(edi_obj.lon)
            self.coord_array[ii]['east'] = float(east)
            self.coord_array[ii]['north'] = float(north)
            self.coord_array[ii]['elev'] = float(edi_obj.elev)
            
        #--> get center of the grid
        east_0 = self.coord_array['east'].mean()
        north_0 = self.coord_array['north'].mean()
        
        #set the center of the grid, for now leve elevation at 0, but later
        #add in elevation.  Also should find the closest station to center
        #of the grid.
        self.center_position = (east_0, north_0, 0.0)
        
        self.coord_array['rel_east'] = self.coord_array['east']-east_0
        self.coord_array['rel_north'] = self.coord_array['north']-north_0
        
    def get_period_list(self):
        """
        make a period list to invert for
        
        """
        
        if self.period_list is not None:
            return
        
        data_period_list = []
        for edi in self.edi_obj_list:
            data_period_list.extend(list(1./edi.freq))
            
        self.data_period_list = sorted(list(set(data_period_list)), 
                                       reverse=False)
                                       
        if self.period_min is not None:
            if self.period_max is None:
                raise ModEMError('Need to input period_max')
        if self.period_max is not None:
            if self.period_min is None:
                raise ModEMError('Need to input period_min')
        if self.period_min is not None and self.period_max is not None:
            if self.max_num_periods is None:
                raise ModEMError('Need to input number of periods to use')
                
            min_index = np.where(self.data_period_list >= self.period_min)[0][1]
            max_index = np.where(self.data_period_list >= self.period_max)[0][0]
            
            pmin = np.log10(self.data_period_list[min_index])
            pmax = np.log10(self.data_period_list[max_index])
            self.period_list = np.logspace(pmin, pmax, num=self.max_num_periods)
            
        if self.period_list is None:
            raise ModEMError('Need to input period_min, period_max, '
                             'max_num_periods or a period_list')
    def get_data_from_edi(self):
        """
        get data from edi files and put into an array for easy manipulation 
        later, this will be handy if you want to rewrite the data file from
        an existing file
        
        """

        if self.edi_list is None:
            self.get_station_locations()
            self.get_period_list()
            
        if self.edi_obj_list is None:
            self.edi_obj_list = []
            for ii, edi in enumerate(self.edi_list):
                edi_obj = mtedi.Edi(edi)
                self.edi_obj_list.append(edi_obj)
            
        ns = len(self.edi_obj_list)
        nf = len(self.period_list)
        
        self.data_array = np.zeros((ns, nf), dtype=[('zxx', np.complex),
                                                    ('zxy', np.complex),
                                                    ('zyx', np.complex),
                                                    ('zyy', np.complex),
                                                    ('tx', np.complex),
                                                    ('ty', np.complex),
                                                    ('zxx_err', np.float),
                                                    ('zxy_err', np.float),
                                                    ('zyx_err', np.float),
                                                    ('zyy_err', np.float),
                                                    ('tx_err', np.float),
                                                    ('ty_err', np.float)])
                                              
        for ii, edi in enumerate(self.edi_obj_list):
            p_dict = dict([(np.round(per, 5), kk) for kk, per in 
                            enumerate(1./edi.freq)])
            for ff, per in enumerate(self.period_list):
                per = np.round(per, 5)
                jj = None
                try:
                    jj = p_dict[per]
                except KeyError:
                    try:
                        jj = np.where((edi.period*.95 <= per) & 
                                      (edi.period*1.05 >= per))[0][0]
                    except IndexError:
                        print 'Could not find {0:.5e} in {1}'.format(per,
                                                                edi.station)
                if jj is not None:
                    self.data_array[ii][ff]['zxx'] = edi.Z.z[jj, 0, 0]
                    self.data_array[ii][ff]['zxy'] = edi.Z.z[jj, 0, 1]
                    self.data_array[ii][ff]['zyx'] = edi.Z.z[jj, 1, 0]
                    self.data_array[ii][ff]['zyy'] = edi.Z.z[jj, 1, 1]
                    
                    self.data_array[ii][ff]['zxx_err'] = edi.Z.zerr[jj, 0, 0]
                    self.data_array[ii][ff]['zxy_err'] = edi.Z.zerr[jj, 0, 1]
                    self.data_array[ii][ff]['zyx_err'] = edi.Z.zerr[jj, 1, 0]
                    self.data_array[ii][ff]['zyy_err'] = edi.Z.zerr[jj, 1, 1]
                    if edi.Tipper.tipper is not None:
                        self.data_array[ii][ff]['tx'] = \
                                                edi.Tipper.tipper[jj, 0, 0]
                        self.data_array[ii][ff]['ty'] = \
                                                edi.Tipper.tipper[jj, 0, 1]
                        
                        self.data_array[ii][ff]['tx_err'] = \
                                                edi.Tipper.tippererr[jj, 0, 0]
                        self.data_array[ii][ff]['ty_err'] = \
                                                edi.Tipper.tippererr[jj, 0, 1]
                    
        
        
        
        
    def write_data_file(self, save_path=None, fn_basename=None):
        """
        write data file for ModEM
        
        """
        
        if save_path is not None:
            self.save_path = save_path
        if fn_basename is not None:
            self.fn_basename = fn_basename
            
        self.data_fn = os.path.join(self.save_path, self.fn_basename)
        
        if self.coord_array is None:
            self.get_station_locations()
            
        self.get_period_list()
        
        if self.data_array is None:
            self.get_data_from_edi()

        dlines = []           
        for inv_mode in self.inv_mode_dict[self.inv_mode]:
            dlines.append(self.header_strings[0])
            dlines.append(self.header_strings[1])
            dlines.append('> {0}\n'.format(inv_mode))
            dlines.append('> exp({0}i\omega t)\n'.format(self.wave_sign))
            if inv_mode.find('Impedance') > 0:
                dlines.append('> {0}\n'.format(self.units))
            elif inv_mode.find('Vertical') >=0:
                dlines.append('> []\n')
            dlines.append('> 0\n') #oriention, need to add at some point
            dlines.append('> {0: >7.3f} {1: >7.3f}\n'.format(
                          self.center_position[0], self.center_position[1]))
            dlines.append('> {0} {1}\n'.format(self.data_array.shape[1],
                                               self.data_array.shape[0]))
                                               
            for ss in range(self.data_array.shape[0]):
                for ff in range(self.data_array.shape[1]):
                    for comp in self.inv_comp_dict[inv_mode]:
                        zz = self.data_array[ss, ff][comp]
                        if zz != 0.0+0.0j:
                            per = '{0:<12.5e}'.format(self.period_list[ff])
                            sta = '{0:>7}'.format(self.coord_array[ss]['station'])
                            lat = '{0:> 9.3f}'.format(self.coord_array[ss]['lat'])
                            lon = '{0:> 9.3f}'.format(self.coord_array[ss]['lon'])
                            eas = '{0:> 12.3f}'.format(self.coord_array[ss]['rel_east_c'])
                            nor = '{0:> 12.3f}'.format(self.coord_array[ss]['rel_north_c'])
                            ele = '{0:> 12.3f}'.format(0)
                            com = '{0:>4}'.format(comp.upper())
                            rea = '{0:> 14.6e}'.format(zz.real)
                            ima = '{0:> 14.6e}'.format(zz.imag)
                            rel_err = self.data_array[ss, ff][comp+'_err']/\
                                      abs(zz)
                            if rel_err < self.error_floor/100.:
                                rel_err = self.error_floor/100.*abs(zz)
                            rel_err = '{0:> 14.6e}'.format(rel_err)
                            
                            dline = ''.join([per, sta, lat, lon, eas, nor, ele, 
                                             com, rea, ima, rel_err, '\n'])
                            dlines.append(dline)
        
        dfid = file(self.data_fn, 'w')
        dfid.writelines(dlines)
        dfid.close()
        
        print 'Wrote ModEM data file to {0}'.format(self.data_fn)
        
    def convert_ws3dinv_data_file(self, ws_data_fn, station_fn=None, 
                                  save_path=None, fn_basename=None):
        """ 
        convert a ws3dinv data file into ModEM format
        
        """
        
        if os.path.isfile(ws_data_fn) == False:
            raise ws.WSInputError('Did not find {0}, check path'.format(ws_data_fn))
        
        if save_path is not None:
            self.save_path = save_path
        else:
            self.save_path = os.path.dirname(ws_data_fn)
    
        if fn_basename is not None:
            self.fn_basename = fn_basename
            
        #--> get data from data file
        wsd = ws.WSData()
        wsd.read_data_file(ws_data_fn, station_fn=station_fn)
        
        ns = wsd.data['station'].shape[0]
        nf = wsd.period_list.shape[0]
        
        self.period_list = wsd.period_list.copy()
        ws_z = wsd.data['z_data']
        ws_z_err = wsd.data['z_data_err']
        self.data_array = np.zeros((ns, nf), dtype=[('zxx', np.complex),
                                                    ('zxy', np.complex),
                                                    ('zyx', np.complex),
                                                    ('zyy', np.complex),
                                                    ('txy', np.complex),
                                                    ('tyx', np.complex),
                                                    ('zxx_err', np.float),
                                                    ('zxy_err', np.float),
                                                    ('zyx_err', np.float),
                                                    ('zyy_err', np.float),
                                                    ('txy_err', np.float),
                                                    ('tyx_err', np.float)])
                                                    
        #--> fill data array
        for ss in range(ns):
            for ff in range(nf):                                            
                self.data_array[ss, ff]['zxx'] = ws_z[ss][ff, 0, 0]
                self.data_array[ss, ff]['zxx_err'] = ws_z_err[ss][ff, 0, 0]
                
                self.data_array[ss, ff]['zxy'] = ws_z[ss][ff, 0, 1]
                self.data_array[ss, ff]['zxy_err'] = ws_z_err[ss][ff, 0, 1]
                
                self.data_array[ss, ff]['zyx'] = ws_z[ss][ff, 1, 0]
                self.data_array[ss, ff]['zyx_err'] = ws_z_err[ss][ff, 1, 0]
                
                self.data_array[ss, ff]['zyy'] = ws_z[ss][ff, 1, 1]
                self.data_array[ss, ff]['zyy_err'] = ws_z_err[ss][ff, 1, 1]

        #--> fill coord array
        self.coord_array = np.zeros(ns, dtype=[('station','|S10'),
                                               ('east', np.float),
                                               ('north', np.float),
                                               ('lat', np.float),
                                               ('lon', np.float),
                                               ('elev', np.float),
                                               ('rel_east', np.float),
                                               ('rel_north', np.float)])
                                        
        for ii, dd in enumerate(wsd.data):
            self.coord_array[ii]['station'] = dd['station']
            self.coord_array[ii]['lat'] = 0.0
            self.coord_array[ii]['lon'] = 0.0
            self.coord_array[ii]['rel_east'] = dd['east']
            self.coord_array[ii]['rel_north'] = dd['north']
            self.coord_array[ii]['elev'] = 0.0
                

        #--write file
        self.write_data_file()
        
    def read_data_file(self, data_fn=None):
        """
        read ModEM data file
        
        """
        
        if data_fn is not None:
            self.data_fn = data_fn
            
        if self.data_fn is None:
            raise ModEMError('data_fn is None, enter a data file to read.')
        elif os.path.isfile(self.data_fn) is False:
            raise ModEMError('Could not find {0}, check path'.format(self.data_fn))
            
        dfid = file(self.data_fn, 'r')
        dlines = dfid.readlines()
        dfid.close()
        
        header_list = []
        metadata_list = []
        data_list = []
        period_list = []
        station_list = []
        for dline in dlines:
            if dline.find('#') == 0:
                header_list.append(dline.strip())
            elif dline.find('>') == 0:
                metadata_list.append(dline[1:].strip())
            else:
                dline_list = dline.strip().split()
                if len(dline_list) == 11:
                    for ii, d_str in enumerate(dline_list):
                        try:
                            dline_list[ii] = float(d_str.strip())
                        except ValueError:
                            pass
                    period_list.append(dline_list[0])
                    station_list.append(dline_list[1])
                    
                    data_list.append(dline_list)
            
        self.period_list = np.array(sorted(set(period_list)))
        station_list = sorted(set(station_list))
        
        period_dict = dict([(per, ii) for ii, per in enumerate(self.period_list)])
        
        #--> need to sort the data into a useful fashion such that each station
        #    is an mt object
        
        data_dict = {}
        z_dummy = np.zeros((len(self.period_list), 2, 2), dtype='complex')
        t_dummy = np.zeros((len(self.period_list), 1, 2), dtype='complex')
        
        index_dict = {'zxx': (0, 0), 'zxy':(0, 1), 'zyx':(1, 0), 'zyy':(1, 1),
                      'tx':(0, 0), 'ty':(0, 1)} 
                      
        tf_dict = {}
        for station in station_list:
            data_dict[station] = mt.MT()
            data_dict[station].Z = mtz.Z(z_array=z_dummy, 
                                        zerr_array=z_dummy.real, 
                                        freq=1./self.period_list)
            data_dict[station].Tipper = mtz.Tipper(tipper_array=t_dummy, 
                                                   tippererr_array=t_dummy.real, 
                                                   freq=1./self.period_list)
            tf_dict[station] = False
                                                   
        for dd in data_list:
            p_index = period_dict[dd[0]]
            ii, jj = index_dict[dd[7].lower()]
            if tf_dict[dd[1]] == False:
                data_dict[dd[1]].lat = dd[2]
                data_dict[dd[1]].lon = dd[3]
                data_dict[dd[1]].grid_east = dd[4]
                data_dict[dd[1]].grid_north = dd[5]
                data_dict[dd[1]].grid_elev = dd[6]
                data_dict[dd[1]].station = dd[1]
                tf_dict[dd[1]] = True
            if dd[7].find('Z') == 0:
                data_dict[dd[1]].Z.z[p_index, ii, jj] = dd[8]+1j*dd[9]
                data_dict[dd[1]].Z.zerr[p_index, ii, jj] = dd[10]
            elif dd[7].find('T') == 0:
                data_dict[dd[1]].Tipper.tipper[p_index, ii, jj] = dd[8]+1j*dd[9]
                data_dict[dd[1]].Tipper.tippererr[p_index, ii, jj] = dd[10]
        self.mt_dict = data_dict
                
            
            
            
        
        
#==============================================================================
# mesh class
#==============================================================================
class Mesh(object):
    """
    make and read a FE mesh grid
    
    The mesh assumes the coordinate system where:
        x == North
        y == East
        z == + down
        
    All dimensions are in meters.
    
    :Example: ::
    
        >>> import mtpy.modeling.ws3dinv as ws
        >>> import os
        >>> #1) make a list of all .edi files that will be inverted for 
        >>> edi_path = r"/home/EDI_Files"
        >>> edi_list = [os.path.join(edi_path, edi) for edi in edi_path 
        >>> ...         if edi.find('.edi') > 0]
        >>> #2) make a grid from the stations themselves with 200m cell spacing
        >>> wsmesh = ws.WSMesh(edi_list=edi_list, cell_size_east=200, 
        >>> ...                cell_size_north=200)
        >>> wsmesh.make_mesh()
        >>> # check to see if the mesh is what you think it should be
        >>> wsmesh.plot_mesh()
        >>> # all is good write the mesh file
        >>> wsmesh.write_initial_file(save_path=r"/home/ws3dinv/Inv1")
    
    ==================== ======================================================
    Attributes           Description    
    ==================== ======================================================
    cell_size_east       mesh block width in east direction
                         *default* is 500
    cell_size_north      mesh block width in north direction
                         *default* is 500
    edi_list             list of .edi files to invert for
    grid_east            overall distance of grid nodes in east direction 
    grid_north           overall distance of grid nodes in north direction 
    grid_z               overall distance of grid nodes in z direction 
    initial_fn           full path to initial file name
    n_layers             total number of vertical layers in model
    nodes_east           relative distance between nodes in east direction 
    nodes_north          relative distance between nodes in north direction 
    nodes_z              relative distance between nodes in east direction 
    pad_east             number of cells for padding on E and W sides
                         *default* is 5
    pad_north            number of cells for padding on S and N sides
                         *default* is 5
    pad_root_east        padding cells E & W will be pad_root_east**(x)
    pad_root_north       padding cells N & S will be pad_root_north**(x) 
    pad_z                number of cells for padding at bottom
                         *default* is 5
    res_list             list of resistivity values for starting model
    res_model            starting resistivity model
    rotation_angle       Angle to rotate the grid to. Angle is measured
                         positve clockwise assuming North is 0 and east is 90.
                         *default* is None
    save_path            path to save file to  
    station_fn           full path to station file
    station_locations    location of stations
    title                title in initial file
    z1_layer             first layer thickness
    z_bottom             absolute bottom of the model *default* is 300,000 
    z_target_depth       Depth of deepest target, *default* is 50,000
    ==================== ======================================================
    
    ==================== ======================================================
    Methods              Description
    ==================== ======================================================
    make_mesh            makes a mesh from the given specifications
    plot_mesh            plots mesh to make sure everything is good
    write_initial_file   writes an initial model file that includes the mesh
    ==================== ======================================================
    
    
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
        self.station_locations = kwargs.pop('station_locations', None)
       
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
        
        self.grid_center = None
        
        #rotation angle
        self.rotation_angle = kwargs.pop('rotation_angle', None)
        
        #inital file stuff
        self.initial_fn = None
        self.station_fn = None
        self.save_path = kwargs.pop('save_path', None)
        self.title = 'Model File written by MTpy.modeling.modem'
        
        
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
        on a log scale to z_target_depth.  Then the model is
        padded with pad_z number of cells to extend the depth of the model.
        
        padding = np.round(cell_size_east*pad_root_east**np.arange(start=.5,
                           stop=3, step=3./pad_east))+west 
        
        """
        #if station locations are not input read from the edi files
        if self.station_locations is None:
            if self.edi_list is None:
                raise AttributeError('edi_list is None, need to input a list of '
                                     'edi files to read in.')
                                     
            n_stations = len(self.edi_list)
            
            #make a structured array to put station location information into
            self.station_locations = np.zeros(n_stations,
                                              dtype=[('station','|S10'),
                                                     ('lat', np.float),
                                                     ('lon', np.float),
                                                     ('east', np.float),
                                                     ('north', np.float), 
                                                     ('rel_east', np.float),
                                                     ('rel_north', np.float),
                                                     ('rel_east_c', np.float),
                                                     ('rel_north_c', np.float),
                                                     ('elev', np.float)])
            #get station locations in meters
            for ii, edi in enumerate(self.edi_list):
                zz = mtedi.Edi()
                zz.readfile(edi)
                zone, east, north = utm2ll.LLtoUTM(23, zz.lat, zz.lon)
                self.station_locations[ii]['lat'] = zz.lat
                self.station_locations[ii]['lon'] = zz.lon
                self.station_locations[ii]['station'] = zz.station
                self.station_locations[ii]['east'] = east
                self.station_locations[ii]['north'] = north
                self.station_locations[ii]['elev'] = zz.elev
             
            #remove the average distance to get coordinates in a relative space
            self.station_locations['rel_east'] = self.station_locations['east']-\
                                                 self.station_locations['east'].mean()
            self.station_locations['rel_north'] = self.station_locations['north']-\
                                                  self.station_locations['north'].mean()
         
            #translate the stations so they are relative to 0,0
            east_center = (self.station_locations['rel_east'].max()-
                            np.abs(self.station_locations['rel_east'].min()))/2
            north_center = (self.station_locations['rel_north'].max()-
                            np.abs(self.station_locations['rel_north'].min()))/2
            
            self.grid_center = (east_center, north_center)
            #remove the average distance to get coordinates in a relative space
            self.station_locations['rel_east'] -= east_center
            self.station_locations['rel_north'] -= north_center
        
        #pickout the furtherst south and west locations 
        #and put that station as the bottom left corner of the main grid
        west = self.station_locations['rel_east'].min()-self.cell_size_east/2
        east = self.station_locations['rel_east'].max()+self.cell_size_east/2
        south = self.station_locations['rel_north'].min()-self.cell_size_north/2
        north = self.station_locations['rel_north'].max()+self.cell_size_north/2

        #make sure the variable n_stations is initialized        
        try:
            n_stations
        except NameError:
            n_stations = self.station_locations.shape[0]

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
                if xf>(self.station_locations[ii]['rel_east']-self.cell_size_east) 
                and xf<(self.station_locations[ii]['rel_east']+self.cell_size_east)]
            
            #shift the station to the center in the east-west direction
            if east_grid[xx[0]] < self.station_locations[ii]['rel_east']:
                self.station_locations[ii]['rel_east_c'] = \
                                        east_grid[xx[0]]+self.cell_size_east/2
            elif east_grid[xx[0]] > self.station_locations[ii]['rel_east']:
                self.station_locations[ii]['rel_east_c'] = \
                                        east_grid[xx[0]]-self.cell_size_east/2
            
            #look for closest grid line
            yy = [mm for mm, yf in enumerate(north_grid) 
                 if yf>(self.station_locations[ii]['rel_north']-self.cell_size_north) 
                 and yf<(self.station_locations[ii]['rel_north']+self.cell_size_north)]
            
            #shift station to center of cell in north-south direction
            if north_grid[yy[0]] < self.station_locations[ii]['rel_north']:
                self.station_locations[ii]['rel_north_c'] = \
                                    north_grid[yy[0]]+self.cell_size_north/2
            elif north_grid[yy[0]] > self.station_locations[ii]['rel_north']:
                self.station_locations[ii]['rel_north_c'] = \
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
        ax1.scatter(self.station_locations['rel_east_c'],
                    self.station_locations['rel_north_c'], 
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
            ax1.set_xlim(self.station_locations['rel_east'].min()-\
                            10*self.cell_size_east,
                         self.station_locations['rel_east'].max()+\
                             10*self.cell_size_east)
        else:
            ax1.set_xlim(east_limits)
        
        if north_limits == None:
            ax1.set_ylim(self.station_locations['rel_north'].min()-\
                            10*self.cell_size_north,
                         self.station_locations['rel_north'].max()+\
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
        ax2.scatter(self.station_locations['rel_east'],
                    [0]*self.station_locations.shape[0],
                    marker=station_marker,
                    c=marker_color,
                    s=marker_size)

        
        if z_limits == None:
            ax2.set_ylim(self.z_target_depth, -200)
        else:
            ax2.set_ylim(z_limits)
            
        if east_limits == None:
            ax1.set_xlim(self.station_locations['rel_east'].min()-\
                            10*self.cell_size_east,
                         self.station_locations['rel_east'].max()+\
                             10*self.cell_size_east)
        else:
            ax1.set_xlim(east_limits)
            
        ax2.set_ylabel('Depth (m)', fontdict={'size':9, 'weight':'bold'})
        ax2.set_xlabel('Easting (m)', fontdict={'size':9, 'weight':'bold'})  
        
        plt.show()
        
    
    def write_initial_file(self, log_E=True, **kwargs):
        """
        will write an initial file for wsinv3d.  
        
        Note that x is assumed to be S --> N, y is assumed to be W --> E and
        z is positive downwards.  This means that index [0, 0, 0] is the 
        southwest corner of the first layer.  Therefore if you build a model
        by hand the layer block will look as it should in map view. 
        
        Also, the xgrid, ygrid and zgrid are assumed to be the relative 
        distance between neighboring nodes.  This is needed because wsinv3d 
        builds the  model from the bottom SW corner assuming the cell width
        from the init file.
        
           
        
        Key Word Arguments:
        ----------------------
        
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
                        resistivity model **res_model_int**.  
                        This must be input if you input **res_model_int**
                        If res_list is None, then Nr = 0 and the real 
                        resistivity values are used from **res_model**.
                        *default* is 100
                        
            **title** : string
                        Title that goes into the first line of savepath/init3d
                        
            **res_model** : np.array((nx,ny,nz))
                        Starting resistivity model.  Each cell is allocated an
                        integer value that cooresponds to the index value of
                        **res_list**.  .. note:: again that the modeling code 
                        assumes that the first row it reads in is the northern
                        most row and the first column it reads in is the 
                        western most column.  Similarly, the first plane it 
                        reads in is the Earth's surface.
                        
            **res_model_int** : np.array((nx,ny,nz))
                        Starting resistivity model.  Each cell is allocated an
                        linear resistivity value.
                        .. note:: again that the modeling code 
                        assumes that the first row it reads in is the northern
                        most row and the first column it reads in is the 
                        western most column.  Similarly, the first plane it 
                        reads in is the Earth's surface.
                            
                        
                          
        """
        keys = ['nodes_east', 'nodes_north', 'nodes_z', 'title',
                'res_model', 'save_path', 'initial_fn']
        for key in keys:
            try:
                setattr(self, key, kwargs[key])
            except KeyError:
                if self.__dict__[key] is None:
                    pass

        if self.initial_fn is None:
            if self.save_path is None:
                self.save_path = os.getcwd()
                self.initial_fn = os.path.join(self.save_path, "ModEM_Mesh")
            elif os.path.isdir(self.save_path) == True:
                self.initial_fn = os.path.join(self.save_path, "ModEM_Mesh")
            else:
                self.save_path = os.path.dirname(self.save_path)
                self.initial_fn= self.save_path
                
        #--> make a res_model if there is not one given
        if type(self.res_model) is float or type(self.res_model) is int:
            self.res_model = np.zeros((self.nodes_north.shape[0],
                                       self.nodes_east.shape[0],
                                       self.nodes_z.shape[0]))
            self.res_model[:, :, :] = self.res_model
        elif self.res_model is None:
            self.res_model = np.zeros((self.nodes_north.shape[0],
                                       self.nodes_east.shape[0],
                                       self.nodes_z.shape[0]))
            self.res_model[:, :, :] = 100.00

        #--> write file
        ifid = file(self.initial_fn, 'w')
        ifid.write('# {0}\n'.format(self.title.upper()))
        ifid.write('{0} {1} {2} {3} LOGE\n'.format(self.nodes_north.shape[0],
                                                   self.nodes_east.shape[0],
                                                   self.nodes_z.shape[0],
                                                   0))
    
        #write S --> N node block
        for ii, nnode in enumerate(self.nodes_north):
            ifid.write('{0:.1f} '.format(abs(nnode)))
        ifid.write('\n')
        
        #write W --> E node block        
        for jj, enode in enumerate(self.nodes_east):
            ifid.write('{0:.1f} '.format(abs(enode)))
        ifid.write('\n')
    
        #write top --> bottom node block
        for kk, zz in enumerate(self.nodes_z):
            ifid.write('{0:.1f} '.format(abs(zz)))
        ifid.write('\n')
        
        ifid.write('\n')
        #write the resistivity list
        write_res_model = self.res_model[::-1, :, :]
        for zz in range(self.nodes_z.shape[0]):
            for nn in range(self.nodes_north.shape[0]):
                for ee in range(self.nodes_east.shape[0]):
                    if log_E == True:
                        ifid.write('{0: >8.4f}'.format(
                                          np.log(write_res_model[nn, ee, zz])))
                    else:
                        ifid.write('{0: >8.4f}'.format(
                                          write_res_model[nn, ee, zz]))
                ifid.write('\n')
            ifid.write('\n')
        ifid.write('{0:<8.3f} {1:<8.3f} {2:<8.3f}\n'.format(self.grid_center[0],
                                                          self.grid_center[1],
                                                          0. ))
        ifid.write('0\n')
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
        log_yn = nsize[4]
    
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
        
        #get model
        count_z = 0
        while line_index < len(ilines):
            iline = ilines[line_index].strip().split()
            if len(iline) == 0:
                count_n = 0
                count_z += 1 
            elif len(iline) == 0:
                break
            else:
                count_e = 0
                while count_e < n_east:
                    #be sure the indes of res list starts at 0 not 1 as 
                    #in ws3dinv

                    self.res_model[count_n, count_e, count_z] =\
                                    self.res_list[int(iline[count_e])-1]
                    self.res_model_int[count_n, count_e, count_z] =\
                                        int(iline[count_e])
                    count_e += 1
                count_n += 1
                line_index += 1
            # Need to be sure that the resistivity array matches
            # with the grids, such that the first index is the 
            # furthest south, even though ws3dinv outputs as first
            # index as furthest north. 
            self.res_model = self.res_model[::-1, :, :]

#==============================================================================
# plot response       
#==============================================================================
class PlotResponse(object):
    """
    plot data and response
    
    :Example: ::
    
        >>> import mtpy.modeling.ws3dinv as ws
        >>> dfn = r"/home/MT/ws3dinv/Inv1/WSDataFile.dat"
        >>> rfn = r"/home/MT/ws3dinv/Inv1/Test_resp.00"
        >>> sfn = r"/home/MT/ws3dinv/Inv1/WSStationLocations.txt"
        >>> wsrp = ws.PlotResponse(data_fn=dfn, resp_fn=rfn, station_fn=sfn)
        >>> # plot only the TE and TM modes
        >>> wsrp.plot_component = 2
        >>> wsrp.redraw_plot()
    
    ======================== ==================================================
    Attributes               Description    
    ======================== ==================================================
    color_mode               [ 'color' | 'bw' ] color or black and white plots
    cted                     color for data TE mode
    ctem                     color for data TM mode
    ctmd                     color for model TE mode
    ctmm                     color for model TM mode
    data_fn                  full path to data file
    data_object              WSResponse instance
    e_capsize                cap size of error bars in points (*default* is .5)
    e_capthick               cap thickness of error bars in points (*default*
                             is 1)
    fig_dpi                  resolution of figure in dots-per-inch (300)
    fig_list                 list of matplotlib.figure instances for plots
    fig_size                 size of figure in inches (*default* is [6, 6])
    font_size                size of font for tick labels, axes labels are
                             font_size+2 (*default* is 7)
    legend_border_axes_pad   padding between legend box and axes 
    legend_border_pad        padding between border of legend and symbols
    legend_handle_text_pad   padding between text labels and symbols of legend
    legend_label_spacing     padding between labels
    legend_loc               location of legend 
    legend_marker_scale      scale of symbols in legend
    lw                       line width response curves (*default* is .5)
    ms                       size of markers (*default* is 1.5)
    mted                     marker for data TE mode
    mtem                     marker for data TM mode
    mtmd                     marker for model TE mode
    mtmm                     marker for model TM mode 
    phase_limits             limits of phase
    plot_component           [ 2 | 4 ] 2 for TE and TM or 4 for all components
    plot_style               [ 1 | 2 ] 1 to plot each mode in a seperate
                             subplot and 2 to plot xx, xy and yx, yy in same 
                             plots
    plot_type                [ '1' | list of station name ] '1' to plot all 
                             stations in data file or input a list of station
                             names to plot if station_fn is input, otherwise
                             input a list of integers associated with the 
                             index with in the data file, ie 2 for 2nd station
    plot_z                   [ True | False ] *default* is True to plot 
                             impedance, False for plotting resistivity and 
                             phase
    plot_yn                  [ 'n' | 'y' ] to plot on instantiation
    res_limits               limits of resistivity in linear scale
    resp_fn                  full path to response file
    resp_object              WSResponse object for resp_fn, or list of 
                             WSResponse objects if resp_fn is a list of
                             response files
    station_fn               full path to station file written by WSStation
    subplot_bottom           space between axes and bottom of figure
    subplot_hspace           space between subplots in vertical direction
    subplot_left             space between axes and left of figure
    subplot_right            space between axes and right of figure
    subplot_top              space between axes and top of figure
    subplot_wspace           space between subplots in horizontal direction    
    ======================== ==================================================
    """
    
    def __init__(self, data_fn=None, resp_fn=None, **kwargs):
        self.data_fn = data_fn
        self.resp_fn = resp_fn
        
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
            
        self.phase_limits = kwargs.pop('phase_limits', None)
        self.res_limits = kwargs.pop('res_limits', None)

        self.fig_num = kwargs.pop('fig_num', 1)
        self.fig_size = kwargs.pop('fig_size', [6, 6])
        self.fig_dpi = kwargs.pop('dpi', 300)
        
        self.subplot_wspace = .2
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
        self.plot_z = kwargs.pop('plot_z', True)
        self.ylabel_pad = kwargs.pop('ylabel_pad', 1.25)
        
        self.fig_list = []
        
        if self.plot_yn == 'y':
            self.plot()
    
    def plot(self):
        """
        plot
        """
        
        self.data_object = Data()
        self.data_object.read_data_file(self.data_fn)
                                        
        #get shape of impedance tensors
        ns = len(self.data_object.mt_dict.keys())
    
        #read in response files
        if self.resp_fn != None:
            self.resp_object = []
            if type(self.resp_fn) is not list:
                resp_obj = Data()
                resp_obj.read_data_file(self.resp_fn)
                self.resp_object = [resp_obj]
            else:
                for rfile in self.resp_fn:
                    resp_obj = Data()
                    resp_obj.read_data_file(rfile)
                    self.resp_object.append(resp_obj)

        #get number of response files
        nr = len(self.resp_object)
        
        if type(self.plot_type) is list:
            ns = len(self.plot_type)
          
        #--> set default font size                           
        plt.rcParams['font.size'] = self.font_size

        fontdict = {'size':self.font_size+2, 'weight':'bold'} 
        if self.plot_z == True:
            h_ratio = [1,1]
        elif self.plot_z == False:
            h_ratio = [2, 1.5]
        
        ax_list = []
        line_list = []
        label_list = []
        
        
        #--> make key word dictionaries for plotting
        kw_xx = {'color':self.cted,
                 'marker':self.mted,
                 'ms':self.ms,
                 'ls':':',
                 'lw':self.lw,
                 'e_capsize':self.e_capsize,
                 'e_capthick':self.e_capthick}        
       
        kw_yy = {'color':self.ctmd,
                 'marker':self.mtmd,
                 'ms':self.ms,
                 'ls':':',
                 'lw':self.lw,
                 'e_capsize':self.e_capsize,
                 'e_capthick':self.e_capthick}        
        
        if self.plot_type != '1':
            pstation_list = []
            if type(self.plot_type) is not list:
                self.plot_type = [self.plot_type]
            for ii, station in enumerate(self.data_object.mt_dict.keys()):
                if type(station) is not int:
                    for pstation in self.plot_type:
                        if station.find(str(pstation)) >= 0:
                            pstation_list.append(station)
                else:
                    for pstation in self.plot_type:
                        if station == int(pstation):
                            pstation_list.append(ii)
        else:
            pstation_list = np.arange(ns)
        
        for jj, station in enumerate(pstation_list):
            z_obj = self.data_object.mt_dict[station].Z
            t_obj = self.data_object.mt_dict[station].Tipper
            period = self.data_object.period_list
            print 'Plotting: {0}'.format(station)
            
            #convert to apparent resistivity and phase
            rp = mtplottools.ResPhase(z_object=z_obj)
            #find locations where points have been masked
            nzxx = np.nonzero(z_obj.z[:, 0, 0])[0]
            nzxy = np.nonzero(z_obj.z[:, 0, 1])[0]
            nzyx = np.nonzero(z_obj.z[:, 1, 0])[0]
            nzyy = np.nonzero(z_obj.z[:, 1, 1])[0]
            ntx = np.nonzero(t_obj.tipper[:, 0, 0])[0]
            nty = np.nonzero(t_obj.tipper[:, 0, 1])[0]
            
            if self.resp_fn != None:
                plotr = True
            else:
                plotr = False
            
            #make figure 
            fig = plt.figure(station, self.fig_size, dpi=self.fig_dpi)
            plt.clf()
            fig.suptitle(str(station), fontdict=fontdict)
            
            #set the grid of subplots
            plot_tipper = t_obj.tipper.all() == 0.0
            if plot_tipper == False:
                #makes more sense if plot_tipper is True to plot tipper
                plot_tipper = True
            else:
                plot_tipper = False
                
            if plot_tipper == True:
                gs = gridspec.GridSpec(2, 6,
                                   wspace=self.subplot_wspace,
                                   left=self.subplot_left,
                                   top=self.subplot_top,
                                   bottom=self.subplot_bottom, 
                                   right=self.subplot_right, 
                                   hspace=self.subplot_hspace,
                                   height_ratios=h_ratio)
            else:
                gs = gridspec.GridSpec(2, 4,
                                       wspace=self.subplot_wspace,
                                       left=self.subplot_left,
                                       top=self.subplot_top,
                                       bottom=self.subplot_bottom, 
                                       right=self.subplot_right, 
                                       hspace=self.subplot_hspace,
                                       height_ratios=h_ratio)
            #---------plot the apparent resistivity-----------------------------------
            #plot each component in its own subplot
            if self.plot_style == 1:
                #plot xy and yx 
                if self.plot_component == 2:
                    if plot_tipper == False:
                        axrxy = fig.add_subplot(gs[0, 0:2])
                        axryx = fig.add_subplot(gs[0, 2:], sharex=axrxy)
                        
                        axpxy = fig.add_subplot(gs[1, 0:2], sharex=axrxy)
                        axpyx = fig.add_subplot(gs[1, 2:], sharex=axrxy)
                    else:
                        
                        axrxy = fig.add_subplot(gs[0, 0:2])
                        axryx = fig.add_subplot(gs[0, 2:4], sharex=axrxy)
                        
                        axpxy = fig.add_subplot(gs[1, 0:2], sharex=axrxy)
                        axpyx = fig.add_subplot(gs[1, 2:4], sharex=axrxy)
                        
                        axtr = fig.add_subplot(gs[0, 4:], sharex=axrxy)
                        axti = fig.add_subplot(gs[1, 4:], sharex=axrxy)
                        
                        
                    if self.plot_z == False: 
                        #plot resistivity
                        erxy = mtplottools.plot_errorbar(axrxy, 
                                                         period,
                                                         rp.resxy[nzxy],
                                                         rp.resxy_err[nzxy],
                                                         **kw_xx)   

                        eryx = mtplottools.plot_errorbar(axryx, 
                                                         period[nzyx], 
                                                         rp.resyx[nzyx], 
                                                         rp.resyx_err[nzyx],
                                                         **kw_yy)
                        #plot phase                         
                        erxy = mtplottools.plot_errorbar(axpxy, 
                                                         period[nzxy], 
                                                         rp.phasexy[nzxy], 
                                                         rp.phasexy_err[nzxy],
                                                         **kw_xx)
                        eryx = mtplottools.plot_errorbar(axpyx, 
                                                         period[nzyx], 
                                                         rp.phaseyx[nzyx], 
                                                         rp.phaseyx_err[nzyx],
                                                         **kw_yy)
                        
                    elif self.plot_z == True:
                        #plot real
                        erxy = mtplottools.plot_errorbar(axrxy, 
                                                  period[nzxy], 
                                                  z_obj.z[nzxy,0,1].real, 
                                                  z_obj.zerr[nzxy,0,1].real,
                                                  **kw_xx)
                        eryx = mtplottools.plot_errorbar(axryx, 
                                                     period[nzyx], 
                                                     z_obj.z[nzyx,1,0].real, 
                                                     z_obj.zerr[nzyx,1,0].real,
                                                     **kw_yy)
                        #plot phase                         
                        erxy = mtplottools.plot_errorbar(axpxy, 
                                                  period[nzxy], 
                                                  z_obj.z[nzxy,0,1].imag, 
                                                  z_obj.zerr[nzxy,0,1].imag,
                                                  **kw_xx)
                        eryx = mtplottools.plot_errorbar(axpyx, 
                                                  period[nzyx], 
                                                  z_obj.z[nzyx,1,0].imag, 
                                                  z_obj.zerr[nzyx,1,0].imag,
                                                  **kw_yy)
                    #plot tipper
                    if plot_tipper == True:
                        ertx = mtplottools.plot_errorbar(axtr, 
                                                 period,
                                                 t_obj.tipper[ntx, 0, 0].real,
                                                 t_obj.tippererr[ntx, 0, 0],
                                                 **kw_xx)
                        erty = mtplottools.plot_errorbar(axtr, 
                                                 period,
                                                 t_obj.tipper[nty, 0, 1].real,
                                                 t_obj.tippererr[nty, 0, 1],
                                                 **kw_yy)
                                                 
                        ertx = mtplottools.plot_errorbar(axti, 
                                                 period,
                                                 t_obj.tipper[ntx, 0, 0].imag,
                                                 t_obj.tippererr[ntx, 0, 0],
                                                 **kw_xx)
                        erty = mtplottools.plot_errorbar(axti, 
                                                 period,
                                                 t_obj.tipper[nty, 0, 1].imag,
                                                 t_obj.tippererr[nty, 0, 1],
                                                 **kw_yy)
                    
                    if plot_tipper == False:                          
                        ax_list = [axrxy, axryx, axpxy, axpyx]
                        line_list = [[erxy[0]], [eryx[0]]]
                        label_list = [['$Z_{xy}$'], ['$Z_{yx}$']]
                    else:                          
                        ax_list = [axrxy, axryx, axpxy, axpyx, axtr, axti]
                        line_list = [[erxy[0]], [eryx[0]], 
                                     [ertx[0], erty[0]]]
                        label_list = [['$Z_{xy}$'], ['$Z_{yx}$'],
                                       ['$T_{x}$', '$T_{y}$']]
                                                           
                elif self.plot_component == 4:
                    if plot_tipper == False:
                        axrxx = fig.add_subplot(gs[0, 0])
                        axrxy = fig.add_subplot(gs[0, 1], sharex=axrxx)
                        axryx = fig.add_subplot(gs[0, 2], sharex=axrxx)
                        axryy = fig.add_subplot(gs[0, 3], sharex=axrxx)
                        
                        axpxx = fig.add_subplot(gs[1, 0])
                        axpxy = fig.add_subplot(gs[1, 1], sharex=axrxx)
                        axpyx = fig.add_subplot(gs[1, 2], sharex=axrxx)
                        axpyy = fig.add_subplot(gs[1, 3], sharex=axrxx)
                    else:
                        axrxx = fig.add_subplot(gs[0, 0])
                        axrxy = fig.add_subplot(gs[0, 1], sharex=axrxx)
                        axryx = fig.add_subplot(gs[0, 2], sharex=axrxx)
                        axryy = fig.add_subplot(gs[0, 3], sharex=axrxx)
                        
                        axpxx = fig.add_subplot(gs[1, 0])
                        axpxy = fig.add_subplot(gs[1, 1], sharex=axrxx)
                        axpyx = fig.add_subplot(gs[1, 2], sharex=axrxx)
                        axpyy = fig.add_subplot(gs[1, 3], sharex=axrxx)
                        
                        axtxr = fig.add_subplot(gs[0, 4], sharex=axrxx)
                        axtxi = fig.add_subplot(gs[1, 4], sharex=axrxx)
                        axtyr = fig.add_subplot(gs[0, 5], sharex=axrxx)
                        axtyi = fig.add_subplot(gs[1, 5], sharex=axrxx)
                    
                    if self.plot_z == False:
                        #plot resistivity
                        erxx= mtplottools.plot_errorbar(axrxx, 
                                                  period[nzxx], 
                                                  rp.resxx[nzxx], 
                                                  rp.resxx_err[nzxx],
                                                  **kw_xx)
                        erxy = mtplottools.plot_errorbar(axrxy, 
                                                  period[nzxy], 
                                                  rp.resxy[nzxy], 
                                                  rp.resxy_err[nzxy],
                                                  **kw_xx)
                        eryx = mtplottools.plot_errorbar(axryx, 
                                                  period[nzyx], 
                                                  rp.resyx[nzyx], 
                                                  rp.resyx_err[nzyx],
                                                  **kw_yy)
                        eryy = mtplottools.plot_errorbar(axryy, 
                                                  period[nzyy], 
                                                  rp.resyy[nzyy], 
                                                  rp.resyy_err[nzyy],
                                                  **kw_yy)
                        #plot phase                         
                        erxx= mtplottools.plot_errorbar(axpxx, 
                                                  period[nzxx], 
                                                  rp.phasexx[nzxx], 
                                                  rp.phasexx_err[nzxx],
                                                  **kw_xx)
                        erxy = mtplottools.plot_errorbar(axpxy, 
                                                  period[nzxy], 
                                                  rp.phasexy[nzxy], 
                                                  rp.phasexy_err[nzxy],
                                                  **kw_xx)
                        eryx = mtplottools.plot_errorbar(axpyx, 
                                                  period[nzyx], 
                                                  rp.phaseyx[nzyx], 
                                                  rp.phaseyx_err[nzyx],
                                                  **kw_yy)
                        eryy = mtplottools.plot_errorbar(axpyy, 
                                                  period[nzyy], 
                                                  rp.phaseyy[nzyy], 
                                                  rp.phaseyy_err[nzyy],
                                                  **kw_yy)
                    elif self.plot_z == True:
                        #plot real
                        erxx = mtplottools.plot_errorbar(axrxx, 
                                                  period[nzxx], 
                                                  z_obj.z[nzxx,0,0].real, 
                                                  z_obj.zerr[nzxx,0,0].real,
                                                  **kw_xx)
                        erxy = mtplottools.plot_errorbar(axrxy, 
                                                  period[nzxy], 
                                                  z_obj.z[nzxy,0,1].real, 
                                                  z_obj.zerr[nzxy,0,1].real,
                                                  **kw_xx)
                        eryx = mtplottools.plot_errorbar(axryx, 
                                                  period[nzyx], 
                                                  z_obj.z[nzyx,1,0].real, 
                                                  z_obj.zerr[nzyx,1,0].real,
                                                  **kw_yy)
                        eryy = mtplottools.plot_errorbar(axryy, 
                                                  period[nzyy], 
                                                  z_obj.z[nzyy,1,1].real, 
                                                  z_obj.zerr[nzyy,1,1].real,
                                                  **kw_yy)
                        #plot phase                         
                        erxx = mtplottools.plot_errorbar(axpxx, 
                                                  period[nzxx], 
                                                  z_obj.z[nzxx,0,0].imag, 
                                                  z_obj.zerr[nzxx,0,0].imag,
                                                  **kw_xx)
                        erxy = mtplottools.plot_errorbar(axpxy, 
                                                  period[nzxy], 
                                                  z_obj.z[nzxy,0,1].imag, 
                                                  z_obj.zerr[nzxy,0,1].imag,
                                                  **kw_xx)
                        eryx = mtplottools.plot_errorbar(axpyx, 
                                                  period[nzyx], 
                                                  z_obj.z[nzyx,1,0].imag, 
                                                  z_obj.zerr[nzyx,1,0].imag,
                                                  **kw_yy)
                        eryy = mtplottools.plot_errorbar(axpyy, 
                                                  period[nzyy], 
                                                  z_obj.z[nzyy,1,1].imag, 
                                                  z_obj.zerr[nzyy,1,1].imag,
                                                  **kw_yy)
                                                  
                    #plot tipper
                    if plot_tipper == True:
                        ertx = mtplottools.plot_errorbar(axtxr, 
                                                 period,
                                                 t_obj.tipper[ntx, 0, 0].real,
                                                 t_obj.tippererr[ntx, 0, 0],
                                                 **kw_xx)
                        erty = mtplottools.plot_errorbar(axtyr, 
                                                 period,
                                                 t_obj.tipper[nty, 0, 1].real,
                                                 t_obj.tippererr[nty, 0, 0],
                                                 **kw_yy)
                                                 
                        ertx = mtplottools.plot_errorbar(axtxi, 
                                                 period,
                                                 t_obj.tipper[ntx, 0, 0].imag,
                                                 t_obj.tippererr[ntx, 0, 1],
                                                 **kw_xx)
                        erty = mtplottools.plot_errorbar(axtyi, 
                                                 period,
                                                 t_obj.tipper[nty, 0, 1].imag,
                                                 t_obj.tippererr[nty, 0, 1],
                                                 **kw_yy)
                    if plot_tipper == False:                    
                        ax_list = [axrxx, axrxy, axryx, axryy, 
                                   axpxx, axpxy, axpyx, axpyy]
                        line_list = [[erxx[0]], [erxy[0]], [eryx[0]], [eryy[0]]]
                        label_list = [['$Z_{xx}$'], ['$Z_{xy}$'], 
                                      ['$Z_{yx}$'], ['$Z_{yy}$']]
                    else:                    
                        ax_list = [axrxx, axrxy, axryx, axryy, 
                                   axpxx, axpxy, axpyx, axpyy, 
                                   axtxr, axtxi, axtyr, axtyi]
                        line_list = [[erxx[0]], [erxy[0]], 
                                     [eryx[0]], [eryy[0]],
                                     [ertx[0]], [erty[0]]]
                        label_list = [['$Z_{xx}$'], ['$Z_{xy}$'], 
                                      ['$Z_{yx}$'], ['$Z_{yy}$'],
                                      ['$T_{x}$'], ['$T_{y}$']]
                    
                #set axis properties
                for aa, ax in enumerate(ax_list):
                    ax.tick_params(axis='y', pad=self.ylabel_pad)
                    ylabels = ax.get_yticks().tolist()
                    ylabels[-1] = ''
                    ylabels[0] = ''
                    ax.set_yticklabels(ylabels)
                    
                    if len(ax_list) == 4 or len(ax_list) == 6:
                        if aa < 2:
                            plt.setp(ax.get_xticklabels(), visible=False)
                            if self.plot_z == False:
                                ax.set_yscale('log')
                            if self.res_limits is not None:
                                ax.set_ylim(self.res_limits)
                        else:
                            ax.set_ylim(self.phase_limits)
                            ax.set_xlabel('Period (s)', fontdict=fontdict)
                            
                        #set axes labels
                        if aa == 0:
                            if self.plot_z == False:
                                ax.set_ylabel('App. Res. ($\mathbf{\Omega \cdot m}$)',
                                              fontdict=fontdict)
                            elif self.plot_z == True:
                                ax.set_ylabel('Re[Z (mV/km nT)]',
                                              fontdict=fontdict)
                        elif aa == 2:
                            if self.plot_z == False:
                                ax.set_ylabel('Phase (deg)',
                                              fontdict=fontdict)
                            elif self.plot_z == True:
                                ax.set_ylabel('Im[Z (mV/km nT)]',
                                              fontdict=fontdict)
                            
                    elif len(ax_list) == 8 or len(ax_list) == 12:
                        if aa < 4:
                            plt.setp(ax.get_xticklabels(), visible=False)
                            if self.plot_z == False:
                                ax.set_yscale('log')
                            if self.res_limits is not None:
                                ax.set_ylim(self.res_limits)
                        else:
                            if aa == 8 or aa == 10:
                                plt.setp(ax.get_xticklabels(), visible=False)
                            else:
                                ax.set_ylim(self.phase_limits)
                                ax.set_xlabel('Period (s)', fontdict=fontdict)

                        #set axes labels
                        if aa == 0:
                            if self.plot_z == False:
                                ax.set_ylabel('App. Res. ($\mathbf{\Omega \cdot m}$)',
                                              fontdict=fontdict)
                            elif self.plot_z == True:
                                ax.set_ylabel('Re[Z (mV/km nT)]',
                                               fontdict=fontdict)
                        elif aa == 4:
                            if self.plot_z == False:
                                ax.set_ylabel('Phase (deg)',
                                              fontdict=fontdict)
                            elif self.plot_z == True:
                                ax.set_ylabel('Im[Z (mV/km nT)]',
                                              fontdict=fontdict)

                    ax.set_xscale('log')
                    ax.set_xlim(xmin=10**(np.floor(np.log10(period[0])))*1.01,
                             xmax=10**(np.ceil(np.log10(period[-1])))*.99)
                    ax.grid(True, alpha=.25)
                    
            # plot xy and yx together and xx, yy together
            elif self.plot_style == 2:
                if self.plot_component == 2:
                    if plot_tipper == False:
                        axrxy = fig.add_subplot(gs[0, 0:])
                        axpxy = fig.add_subplot(gs[1, 0:], sharex=axrxy)
                    else:
                        axrxy = fig.add_subplot(gs[0, 0:4])
                        axpxy = fig.add_subplot(gs[1, 0:4], sharex=axrxy)
                        axtr = fig.add_subplot(gs[0, 4:], sharex=axrxy)
                        axti = fig.add_subplot(gs[1, 4:], sharex=axrxy)
                        
                    if self.plot_z == False:
                        #plot resistivity
                        erxy = mtplottools.plot_errorbar(axrxy, 
                                                  period[nzxy], 
                                                  rp.resxy[nzxy], 
                                                  rp.resxy_err[nzxy],
                                                  **kw_xx)
                        eryx = mtplottools.plot_errorbar(axrxy, 
                                                  period[nzyx], 
                                                  rp.resyx[nzyx], 
                                                  rp.resyx_err[nzyx],
                                                  **kw_yy)
                        #plot phase                         
                        erxy = mtplottools.plot_errorbar(axpxy, 
                                                  period[nzxy], 
                                                  rp.phasexy[nzxy], 
                                                  rp.phasexy_err[nzxy],
                                                  **kw_xx)
                        eryx = mtplottools.plot_errorbar(axpxy, 
                                                  period[nzyx], 
                                                  rp.phaseyx[nzyx], 
                                                  rp.phaseyx_err[nzyx],
                                                  **kw_yy)
                    elif self.plot_z == True:
                        #plot real
                        erxy = mtplottools.plot_errorbar(axrxy, 
                                                  period[nzxy], 
                                                  z_obj.z[nzxy,0,1].real, 
                                                  z_obj.zerr[nzxy,0,1].real,
                                                  **kw_xx)
                        eryx = mtplottools.plot_errorbar(axrxy, 
                                                  period[nzxy], 
                                                  z_obj.z[nzxy,1,0].real, 
                                                  z_obj.zerr[nzxy,1,0].real,
                                                  **kw_yy)
                        #plot phase                         
                        erxy = mtplottools.plot_errorbar(axpxy, 
                                                  period[nzxy], 
                                                  z_obj.z[nzxy,0,1].imag, 
                                                  z_obj.zerr[nzxy,0,1].imag,
                                                  **kw_xx)
                        eryx = mtplottools.plot_errorbar(axpxy, 
                                                  period[nzyx], 
                                                  z_obj.z[nzyx,1,0].imag, 
                                                  z_obj.zerr[nzyx,1,0].imag,
                                                  **kw_yy)
                    #plot tipper
                    if plot_tipper == True:
                        ertx = mtplottools.plot_errorbar(axtr, 
                                                 period,
                                                 t_obj.tipper[ntx, 0, 0].real,
                                                 t_obj.tippererr[ntx, 0, 0],
                                                 **kw_xx)
                        erty = mtplottools.plot_errorbar(axtr, 
                                                 period,
                                                 t_obj.tipper[nty, 0, 1].real,
                                                 t_obj.tippererr[nty, 0, 1],
                                                 **kw_yy)
                                                 
                        ertx = mtplottools.plot_errorbar(axti, 
                                                 period,
                                                 t_obj.tipper[ntx, 0, 0].imag,
                                                 t_obj.tippererr[ntx, 0, 0],
                                                 **kw_xx)
                        erty = mtplottools.plot_errorbar(axti, 
                                                 period,
                                                 t_obj.tipper[nty, 0, 1].imag,
                                                 t_obj.tippererr[nty, 0, 1],
                                                 **kw_yy)
                    
                    if plot_tipper == False:    
                        ax_list = [axrxy, axpxy]
                        line_list = [erxy[0], eryx[0]]
                        label_list = ['$Z_{xy}$', '$Z_{yx}$']
                    else:    
                        ax_list = [axrxy, axpxy, axtr, axti]
                        line_list = [[erxy[0], eryx[0]], 
                                     [ertx[0], erty[0]]]
                        label_list = [['$Z_{xy}$', '$Z_{yx}$'],
                                      ['$T_{x}$', '$T_{y}$']]
                    
                elif self.plot_component == 4:
                    if plot_tipper == False:
                        axrxy = fig.add_subplot(gs[0, 0:2])
                        axpxy = fig.add_subplot(gs[1, 0:2], sharex=axrxy)
                        
                        axrxx = fig.add_subplot(gs[0, 2:], sharex=axrxy)
                        axpxx = fig.add_subplot(gs[1, 2:], sharex=axrxy)
                    else:
                        axrxy = fig.add_subplot(gs[0, 0:2])
                        axpxy = fig.add_subplot(gs[1, 0:2], sharex=axrxy)
                        
                        axrxx = fig.add_subplot(gs[0, 2:4], sharex=axrxy)
                        axpxx = fig.add_subplot(gs[1, 2:4], sharex=axrxy)
                        
                        axtr = fig.add_subplot(gs[0, 4:], sharex=axrxy)
                        axti = fig.add_subplot(gs[1, 4:], sharex=axrxy)
                        
                    if self.plot_z == False:
                        #plot resistivity
                        erxx= mtplottools.plot_errorbar(axrxx, 
                                                  period[nzxx], 
                                                  rp.resxx[nzxx], 
                                                  rp.resxx_err[nzxx],
                                                  **kw_xx)
                        erxy = mtplottools.plot_errorbar(axrxy, 
                                                  period[nzxy], 
                                                  rp.resxy[nzxy], 
                                                  rp.resxy_err[nzxy],
                                                  **kw_xx)
                        eryx = mtplottools.plot_errorbar(axrxy, 
                                                  period[nzyx], 
                                                  rp.resyx[nzyx], 
                                                  rp.resyx_err[nzyx],
                                                  **kw_yy)
                        eryy = mtplottools.plot_errorbar(axrxx, 
                                                  period[nzyy], 
                                                  rp.resyy[nzyy], 
                                                  rp.resyy_err[nzyy],
                                                  **kw_yy)
                        #plot phase                         
                        erxx= mtplottools.plot_errorbar(axpxx, 
                                                  period[nzxx], 
                                                  rp.phasexx[nzxx], 
                                                  rp.phasexx_err[nzxx],
                                                  **kw_xx)
                        erxy = mtplottools.plot_errorbar(axpxy, 
                                                  period[nzxy], 
                                                  rp.phasexy[nzxy], 
                                                  rp.phasexy_err[nzxy],
                                                  **kw_xx)
                        eryx = mtplottools.plot_errorbar(axpxy, 
                                                  period[nzyx], 
                                                  rp.phaseyx[nzyx], 
                                                  rp.phaseyx_err[nzyx],
                                                  **kw_yy)
                        eryy = mtplottools.plot_errorbar(axpxx, 
                                                  period[nzyy], 
                                                  rp.phaseyy[nzyy], 
                                                  rp.phaseyy_err[nzyy],
                                                  **kw_yy)
                    elif self.plot_z == True:
                         #plot real
                        erxx = mtplottools.plot_errorbar(axrxx, 
                                                  period[nzxx], 
                                                  z_obj.z[nzxx,0,0].real, 
                                                  z_obj.zerr[nzxx,0,0].real,
                                                  **kw_xx)
                        erxy = mtplottools.plot_errorbar(axrxy, 
                                                  period[nzxy], 
                                                  z_obj.z[nzxy,0,1].real, 
                                                  z_obj.zerr[nzxy,0,1].real,
                                                  **kw_xx)
                        eryx = mtplottools.plot_errorbar(axrxy, 
                                                  period[nzyx], 
                                                  z_obj.z[nzyx,1,0].real, 
                                                  z_obj.zerr[nzyx,1,0].real,
                                                  **kw_yy)
                        eryy = mtplottools.plot_errorbar(axrxx, 
                                                  period[nzyy], 
                                                  z_obj.z[nzyy,1,1].real, 
                                                  z_obj.zerr[nzyy,1,1].real,
                                                  **kw_yy)
                        #plot phase                         
                        erxx = mtplottools.plot_errorbar(axpxx, 
                                                  period[nzxx], 
                                                  z_obj.z[nzxx,0,0].imag, 
                                                  z_obj.zerr[nzxx,0,0].imag,
                                                  **kw_xx)
                        erxy = mtplottools.plot_errorbar(axpxy, 
                                                  period[nzxy], 
                                                  z_obj.z[nzxy,0,1].imag, 
                                                  z_obj.zerr[nzxy,0,1].imag,
                                                  **kw_xx)
                        eryx = mtplottools.plot_errorbar(axpxy, 
                                                  period[nzyx], 
                                                  z_obj.z[nzyx,1,0].imag, 
                                                  z_obj.zerr[nzyx,1,0].imag,
                                                  **kw_yy)
                        eryy = mtplottools.plot_errorbar(axpxx, 
                                                  period[nzyy], 
                                                  z_obj.z[nzyy,1,1].imag, 
                                                  z_obj.zerr[nzyy,1,1].imag,
                                                  **kw_yy)
                    #plot tipper
                    if plot_tipper == True:
                        ertx = mtplottools.plot_errorbar(axtr, 
                                                 period,
                                                 t_obj.tipper[ntx, 0, 0].real,
                                                 t_obj.tippererr[ntx, 0, 0],
                                                 **kw_xx)
                        erty = mtplottools.plot_errorbar(axtr, 
                                                 period,
                                                 t_obj.tipper[nty, 0, 1].real,
                                                 t_obj.tippererr[nty, 0, 1],
                                                 **kw_yy)
                                                 
                        ertx = mtplottools.plot_errorbar(axti, 
                                                 period,
                                                 t_obj.tipper[ntx, 0, 0].imag,
                                                 t_obj.tippererr[ntx, 0, 0],
                                                 **kw_xx)
                        erty = mtplottools.plot_errorbar(axti, 
                                                 period,
                                                 t_obj.tipper[nty, 0, 1].imag,
                                                 t_obj.tippererr[nty, 0, 1],
                                                 **kw_yy)
                    
                    if plot_tipper == False:
                        ax_list = [axrxy, axrxx, axpxy, axpxx]
                        line_list = [[erxy[0], eryx[0]], [erxx[0], eryy[0]]]
                        label_list = [['$Z_{xy}$', '$Z_{yx}$'], 
                                      ['$Z_{xx}$', '$Z_{yy}$']]
                    else:
                        ax_list = [axrxy, axrxx, axpxy, axpxx, axtr, axti]
                        line_list = [[erxy[0], eryx[0]], [erxx[0], eryy[0]],
                                     [ertx[0]], erty[0]]
                        label_list = [['$Z_{xy}$', '$Z_{yx}$'], 
                                      ['$Z_{xx}$', '$Z_{yy}$'],
                                      ['$T_x$', '$T_y$']]
                        
                #set axis properties
                for aa, ax in enumerate(ax_list):
                    ax.tick_params(axis='y', pad=self.ylabel_pad)
                    ylabels = ax.get_yticks().tolist()
                    ylabels[-1] = ''
                    ylabels[0] = ''
                    ax.set_yticklabels(ylabels)
                    if len(ax_list) == 2:
                        ax.set_xlabel('Period (s)', fontdict=fontdict)
                        if aa == 0:
                            plt.setp(ax.get_xticklabels(), visible=False)
                            if self.plot_z == False:
                                ax.set_yscale('log')
                                ax.set_ylabel('App. Res. ($\mathbf{\Omega \cdot m}$)',
                                              fontdict=fontdict)
                            elif self.plot_z == True:
                                ax.set_ylabel('Re[Impedance (m/s)]',
                                               fontdict=fontdict)
                            if self.res_limits is not None:
                                ax.set_ylim(self.res_limits)
                        else:
                            ax.set_ylim(self.phase_limits)
                            if self.plot_z == False:
                                ax.set_ylabel('Phase (deg)',
                                              fontdict=fontdict)
                            elif self.plot_z == True:
                                ax.set_ylabel('Im[Impedance (m/s)]',
                                              fontdict=fontdict)
                    elif len(ax_list) == 4 and plot_tipper == False:
                        if aa < 2:
                            plt.setp(ax.get_xticklabels(), visible=False)
                            if self.plot_z == False:
                                ax.set_yscale('log')
                            if self.res_limits is not None:
                                ax.set_ylim(self.res_limits)
                        else:
                            if self.plot_z == False:
                                ax.set_ylim(self.phase_limits)
                            ax.set_xlabel('Period (s)', fontdict=fontdict)
                        if aa == 0:
                            if self.plot_z == False:
                                ax.set_ylabel('App. Res. ($\mathbf{\Omega \cdot m}$)',
                                              fontdict=fontdict)
                            elif self.plot_z == True:
                                ax.set_ylabel('Re[Z (mV/km nT)]',
                                               fontdict=fontdict)
                        elif aa == 2:
                            if self.plot_z == False:
                                ax.set_ylabel('Phase (deg)',
                                              fontdict=fontdict)
                            elif self.plot_z == True:
                                ax.set_ylabel('Im[Z (mV/km nT)]',
                                              fontdict=fontdict)
                    elif len(ax_list) == 4 and plot_tipper == True:
                        if aa == 0 or aa == 2:
                            plt.setp(ax.get_xticklabels(), visible=False)
                            if self.plot_z == False:
                                ax.set_yscale('log')
                            if self.res_limits is not None:
                                ax.set_ylim(self.res_limits)
                        else:
                            ax.set_ylim(self.phase_limits)
                            ax.set_xlabel('Period (s)', fontdict=fontdict)
                        if aa == 0:
                            if self.plot_z == False:
                                ax.set_ylabel('App. Res. ($\mathbf{\Omega \cdot m}$)',
                                              fontdict=fontdict)
                            elif self.plot_z == True:
                                ax.set_ylabel('Re[Z (mV/km nT)]',
                                               fontdict=fontdict)
                        elif aa == 1:
                            if self.plot_z == False:
                                ax.set_ylabel('Phase (deg)',
                                              fontdict=fontdict)
                            elif self.plot_z == True:
                                ax.set_ylabel('Im[Z (mV/km nT)]',
                                              fontdict=fontdict)
#                        else:
#                            plt.setp(ax.yaxis.get_ticklabels(), visible=False)

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
                    
                    resp_z_obj = self.resp_object[jj].mt_dict[station].Z
                    resp_z_err = (z_obj.z-resp_z_obj.z)/z_obj.zerr
    
                    resp_t_obj = self.resp_object[jj].mt_dict[station].Tipper
                    
                    rrp = mtplottools.ResPhase(resp_z_obj)
    
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
                    
                    #--> make key word dictionaries for plotting
                    kw_xx = {'color':cxy,
                             'marker':self.mted,
                             'ms':self.ms,
                             'ls':':',
                             'lw':self.lw,
                             'e_capsize':self.e_capsize,
                             'e_capthick':self.e_capthick}        
                   
                    kw_yy = {'color':cyx,
                             'marker':self.mtmd,
                             'ms':self.ms,
                             'ls':':',
                             'lw':self.lw,
                             'e_capsize':self.e_capsize,
                             'e_capthick':self.e_capthick}
                             
                    if self.plot_style == 1:
                        if self.plot_component == 2:
                            if self.plot_z == False:
                                #plot resistivity
                                rerxy = mtplottools.plot_errorbar(axrxy, 
                                                          period[nzxy], 
                                                          rrp.resxy[nzxy], 
                                                          **kw_xx)
                                reryx = mtplottools.plot_errorbar(axryx, 
                                                          period[nzyx], 
                                                          rrp.resyx[nzyx], 
                                                          **kw_yy)
                                #plot phase                         
                                rerxy = mtplottools.plot_errorbar(axpxy, 
                                                          period[nzxy], 
                                                          rrp.phasexy[nzxy], 
                                                          **kw_xx)
                                reryx = mtplottools.plot_errorbar(axpyx, 
                                                          period[nzyx], 
                                                          rrp.phaseyx[nzyx], 
                                                          **kw_yy)
                            elif self.plot_z == True:
                                #plot real
                                rerxy = mtplottools.plot_errorbar(axrxy, 
                                                          period[nzxy], 
                                                          resp_z_obj.z[nzxy,0,1].real, 
                                                          **kw_xx)
                                reryx = mtplottools.plot_errorbar(axryx, 
                                                          period[nzyx], 
                                                          resp_z_obj.z[nzyx,1,0].real, 
                                                          **kw_yy)
                                #plot phase                         
                                rerxy = mtplottools.plot_errorbar(axpxy, 
                                                          period[nzxy], 
                                                          resp_z_obj.z[nzxy,0,1].imag, 
                                                          **kw_xx)
                                reryx = mtplottools.plot_errorbar(axpyx, 
                                                          period[nzyx], 
                                                          resp_z_obj.z[nzyx,1,0].imag, 
                                                          **kw_yy)
                            if plot_tipper == True:
                                rertx = mtplottools.plot_errorbar(axtr, 
                                             period,
                                             resp_t_obj.tipper[ntx, 0, 0].real,
                                             **kw_xx)
                                rerty = mtplottools.plot_errorbar(axtr, 
                                             period,
                                             resp_t_obj.tipper[nty, 0, 1].real,
                                             **kw_yy)
                                                         
                                rertx = mtplottools.plot_errorbar(axti, 
                                             period,
                                             resp_t_obj.tipper[ntx, 0, 0].imag,
                                             **kw_xx)
                                rerty = mtplottools.plot_errorbar(axti, 
                                             period,
                                             t_obj.tipper[nty, 0, 1].imag,
                                             **kw_yy)
                            if plot_tipper == False:
                                line_list[0] += [rerxy[0]]
                                line_list[1] += [reryx[0]]
                                label_list[0] += ['$Z^m_{xy}$ '+
                                                   'rms={0:.2f}'.format(rms_xy)]
                                label_list[1] += ['$Z^m_{yx}$ '+
                                               'rms={0:.2f}'.format(rms_yx)]
                            else:
                                line_list[0] += [rerxy[0]]
                                line_list[1] += [reryx[0]]
                                line_list[2] += [rertx[0], rerty[0]]
                                label_list[0] += ['$Z^m_{xy}$ '+
                                                   'rms={0:.2f}'.format(rms_xy)]
                                label_list[1] += ['$Z^m_{yx}$ '+
                                               'rms={0:.2f}'.format(rms_yx)]
                                label_list[2] += ['$T^m_{x}$', '$T^m_{y}$']
                        elif self.plot_component == 4:
                            if self.plot_z == False:
                                #plot resistivity
                                rerxx= mtplottools.plot_errorbar(axrxx, 
                                                          period[nzxx], 
                                                          rrp.resxx[nzxx], 
                                                          **kw_xx)
                                rerxy = mtplottools.plot_errorbar(axrxy, 
                                                          period[nzxy], 
                                                          rrp.resxy[nzxy], 
                                                          **kw_xx)
                                reryx = mtplottools.plot_errorbar(axryx, 
                                                          period[nzyx], 
                                                          rrp.resyx[nzyx], 
                                                          **kw_yy)
                                reryy = mtplottools.plot_errorbar(axryy, 
                                                          period[nzyy], 
                                                          rrp.resyy[nzyy], 
                                                          **kw_yy)
                                #plot phase                         
                                rerxx= mtplottools.plot_errorbar(axpxx, 
                                                          period[nzxx], 
                                                          rrp.phasexx[nzxx], 
                                                          **kw_xx)
                                rerxy = mtplottools.plot_errorbar(axpxy, 
                                                          period[nzxy], 
                                                          rrp.phasexy[nzxy], 
                                                          **kw_xx)
                                reryx = mtplottools.plot_errorbar(axpyx, 
                                                          period[nzyx], 
                                                          rrp.phaseyx[nzyx], 
                                                          **kw_yy)
                                reryy = mtplottools.plot_errorbar(axpyy, 
                                                          period[nzyy], 
                                                          rrp.phaseyy[nzyy], 
                                                          **kw_yy)
                            elif self.plot_z == True:
                                #plot real
                                rerxx = mtplottools.plot_errorbar(axrxx, 
                                                          period[nzxx], 
                                                          resp_z_obj.z[nzxx,0,0].real, 
                                                          **kw_xx)
                                rerxy = mtplottools.plot_errorbar(axrxy, 
                                                          period[nzxy], 
                                                          resp_z_obj.z[nzxy,0,1].real, 
                                                          **kw_xx)
                                reryx = mtplottools.plot_errorbar(axryx, 
                                                          period[nzyx], 
                                                          resp_z_obj.z[nzyx,1,0].real, 
                                                          **kw_yy)
                                reryy = mtplottools.plot_errorbar(axryy, 
                                                          period[nzyy], 
                                                          resp_z_obj.z[nzyy,1,1].real, 
                                                          **kw_yy)
                                #plot phase                         
                                rerxx = mtplottools.plot_errorbar(axpxx, 
                                                          period[nzxx], 
                                                          resp_z_obj.z[nzxx,0,0].imag, 
                                                          **kw_xx)
                                rerxy = mtplottools.plot_errorbar(axpxy, 
                                                          period[nzxy], 
                                                          resp_z_obj.z[nzxy,0,1].imag, 
                                                          **kw_xx)
                                reryx = mtplottools.plot_errorbar(axpyx, 
                                                          period[nzyx], 
                                                          resp_z_obj.z[nzyx,1,0].imag, 
                                                          **kw_yy)
                                reryy = mtplottools.plot_errorbar(axpyy, 
                                                          period[nzyy], 
                                                          resp_z_obj.z[nzyy,1,1].imag, 
                                                          **kw_yy)
                            if plot_tipper == True:
                                rertx = mtplottools.plot_errorbar(axtxr, 
                                             period,
                                             resp_t_obj.tipper[ntx, 0, 0].real,
                                             **kw_xx)
                                rerty = mtplottools.plot_errorbar(axtyr, 
                                             period,
                                             resp_t_obj.tipper[nty, 0, 1].real,
                                             **kw_yy)
                                                         
                                rertx = mtplottools.plot_errorbar(axtxi, 
                                             period,
                                             resp_t_obj.tipper[ntx, 0, 0].imag,
                                             **kw_xx)
                                rerty = mtplottools.plot_errorbar(axtyi, 
                                             period,
                                             t_obj.tipper[nty, 0, 1].imag,
                                             **kw_yy)
                                             
                            if plot_tipper == False:
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
                            else:
                                line_list[0] += [rerxx[0]]
                                line_list[1] += [rerxy[0]]
                                line_list[2] += [reryx[0]]
                                line_list[3] += [reryy[0]]
                                line_list[4] += [rertx[0]]
                                line_list[5] += [rerty[0]]
                                label_list[0] += ['$Z^m_{xx}$ '+
                                                   'rms={0:.2f}'.format(rms_xx)]
                                label_list[1] += ['$Z^m_{xy}$ '+
                                               'rms={0:.2f}'.format(rms_xy)]
                                label_list[2] += ['$Z^m_{yx}$ '+
                                               'rms={0:.2f}'.format(rms_yx)]
                                label_list[3] += ['$Z^m_{yy}$ '+
                                               'rms={0:.2f}'.format(rms_yy)]
                                label_list[4] += ['$T^m_{x}$ ']
                                label_list[5] += ['$T^m_{y}$']
                                           
                    elif self.plot_style == 2:
                        if self.plot_component == 2:
                            if self.plot_z == False:                            
                                #plot resistivity
                                rerxy = mtplottools.plot_errorbar(axrxy, 
                                                          period[nzxy], 
                                                          rrp.resxy[nzxy], 
                                                          **kw_xx)
                                reryx = mtplottools.plot_errorbar(axrxy, 
                                                          period[nzyx], 
                                                          rrp.resyx[nzyx], 
                                                          **kw_yy)
                                #plot phase                         
                                rerxy = mtplottools.plot_errorbar(axpxy, 
                                                          period[nzxy], 
                                                          rrp.phasexy[nzxy], 
                                                         **kw_xx)
                                reryx = mtplottools.plot_errorbar(axpxy, 
                                                          period[nzyx], 
                                                          rrp.phaseyx[nzyx], 
                                                          **kw_yy)
                            elif self.plot_z == True:
                                #plot real
                                rerxy = mtplottools.plot_errorbar(axrxy, 
                                                          period[nzxy], 
                                                          resp_z_obj.z[nzxy,0,1].real, 
                                                          **kw_xx)
                                reryx = mtplottools.plot_errorbar(axrxy, 
                                                          period[nzyx], 
                                                          resp_z_obj.z[nzyx,1,0].real, 
                                                          **kw_yy)
                                #plot phase                         
                                rerxy = mtplottools.plot_errorbar(axpxy, 
                                                          period[nzxy], 
                                                          resp_z_obj.z[nzxy,0,1].imag, 
                                                          **kw_xx)
                                reryx = mtplottools.plot_errorbar(axpxy, 
                                                          period[nzyx], 
                                                          resp_z_obj.z[nzyx,1,0].imag, 
                                                          **kw_xx)
                            if plot_tipper == True:
                                rertx = mtplottools.plot_errorbar(axtr, 
                                             period,
                                             resp_t_obj.tipper[ntx, 0, 0].real,
                                             **kw_xx)
                                rerty = mtplottools.plot_errorbar(axtr, 
                                             period,
                                             resp_t_obj.tipper[nty, 0, 1].real,
                                             **kw_yy)
                                                         
                                rertx = mtplottools.plot_errorbar(axti, 
                                             period,
                                             resp_t_obj.tipper[ntx, 0, 0].imag,
                                             **kw_xx)
                                rerty = mtplottools.plot_errorbar(axti, 
                                             period,
                                             t_obj.tipper[nty, 0, 1].imag,
                                             **kw_yy)
                                
                            if plot_tipper == False:
                                line_list += [rerxy[0], reryx[0]]
                                label_list += ['$Z^m_{xy}$ '+
                                               'rms={0:.2f}'.format(rms_xy),
                                               '$Z^m_{yx}$ '+
                                               'rms={0:.2f}'.format(rms_yx)]
                            else:
                                line_list[0] += [rerxy[0], reryx[0]]
                                line_list[1] += [rertx[0], rerty[0]]
                                label_list[0] += ['$Z^m_{xy}$ '+
                                               'rms={0:.2f}'.format(rms_xy),
                                               '$Z^m_{yx}$ '+
                                               'rms={0:.2f}'.format(rms_yx)]
                                label_list[1] += ['$T^m_{x}$ ', '$T^m_{y}$']
                                
                        elif self.plot_component == 4:
                            if self.plot_z == False:
                                #plot resistivity
                                rerxx= mtplottools.plot_errorbar(axrxx, 
                                                          period[nzxx], 
                                                          rrp.resxx[nzxx], 
                                                          **kw_xx)
                                rerxy = mtplottools.plot_errorbar(axrxy, 
                                                          period[nzxy], 
                                                          rrp.resxy[nzxy], 
                                                          **kw_xx)
                                reryx = mtplottools.plot_errorbar(axrxy, 
                                                          period[nzyx], 
                                                          rrp.resyx[nzyx], 
                                                          **kw_yy)
                                reryy = mtplottools.plot_errorbar(axrxx, 
                                                          period[nzyy], 
                                                          rrp.resyy[nzyy], 
                                                          **kw_yy)
                                #plot phase                         
                                rerxx= mtplottools.plot_errorbar(axpxx, 
                                                          period[nzxx], 
                                                          rrp.phasexx[nzxx], 
                                                          **kw_xx)
                                rerxy = mtplottools.plot_errorbar(axpxy, 
                                                          period[nzxy], 
                                                          rrp.phasexy[nzxy], 
                                                          **kw_xx)
                                reryx = mtplottools.plot_errorbar(axpxy, 
                                                          period[nzyx], 
                                                          rrp.phaseyx[nzyx], 
                                                          **kw_yy)
                                reryy = mtplottools.plot_errorbar(axpxx, 
                                                          period[nzyy], 
                                                          rrp.phaseyy[nzyy], 
                                                          **kw_yy)
                            elif self.plot_z == True:
                                #plot real
                                rerxx = mtplottools.plot_errorbar(axrxx, 
                                                          period[nzxx], 
                                                          resp_z_obj.z[nzxx,0,0].real, 
                                                          **kw_xx)
                                rerxy = mtplottools.plot_errorbar(axrxy, 
                                                          period[nzxy], 
                                                          resp_z_obj.z[nzxy,0,1].real, 
                                                          **kw_xx)
                                reryx = mtplottools.plot_errorbar(axrxy, 
                                                          period[nzyx], 
                                                          resp_z_obj.z[nzyx,1,0].real, 
                                                          **kw_yy)
                                reryy = mtplottools.plot_errorbar(axrxx, 
                                                          period[nzyy], 
                                                          resp_z_obj.z[nzyy,1,1].real, 
                                                          **kw_yy)
                                #plot phase                         
                                rerxx = mtplottools.plot_errorbar(axpxx, 
                                                          period[nzxx], 
                                                          resp_z_obj.z[nzxx,0,0].imag, 
                                                          **kw_xx)
                                rerxy = mtplottools.plot_errorbar(axpxy, 
                                                          period[nzxy], 
                                                          resp_z_obj.z[nzxy,0,1].imag, 
                                                          **kw_xx)
                                reryx = mtplottools.plot_errorbar(axpxy, 
                                                          period[nzyx], 
                                                          resp_z_obj.z[nzyx,1,0].imag, 
                                                          **kw_yy)
                                reryy = mtplottools.plot_errorbar(axpxx, 
                                                          period[nzyy], 
                                                          resp_z_obj.z[nzyy,1,1].imag, 
                                                          **kw_yy)
                                                          
                            if plot_tipper == True:
                                rertx = mtplottools.plot_errorbar(axtr, 
                                             period,
                                             resp_t_obj.tipper[ntx, 0, 0].real,
                                             **kw_xx)
                                rerty = mtplottools.plot_errorbar(axtr, 
                                             period,
                                             resp_t_obj.tipper[nty, 0, 1].real,
                                             **kw_yy)
                                                         
                                rertx = mtplottools.plot_errorbar(axti, 
                                             period,
                                             resp_t_obj.tipper[ntx, 0, 0].imag,
                                             **kw_xx)
                                rerty = mtplottools.plot_errorbar(axti, 
                                             period,
                                             t_obj.tipper[nty, 0, 1].imag,
                                             **kw_yy)
                                             
                            if plot_tipper == False:
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
                            else:
                                line_list[0] += [rerxy[0], reryx[0]]
                                line_list[1] += [rerxx[0], reryy[0]]
                                line_list[2] += [rertx[0], rerty[0]]
                                label_list[0] += ['$Z^m_{xy}$ '+
                                                   'rms={0:.2f}'.format(rms_xy),
                                                  '$Z^m_{yx}$ '+
                                                  'rms={0:.2f}'.format(rms_yx)]
                                label_list[1] += ['$Z^m_{xx}$ '+
                                                   'rms={0:.2f}'.format(rms_xx),
                                                  '$Z^m_{yy}$ '+
                                                  'rms={0:.2f}'.format(rms_yy)]
                                label_list[2] += ['$T^m_{x}$ ', '$T^m_{y}$']
                    
            #make legends
            if self.plot_style == 1:
                legend_ax_list = ax_list[0:self.plot_component]
                if plot_tipper == True:
                    if self.plot_component == 2:
                        legend_ax_list.append(ax_list[4])
                    elif self.plot_component == 4:
                        legend_ax_list.append(ax_list[8])
                        legend_ax_list.append(ax_list[10])
                for aa, ax in enumerate(legend_ax_list):
                    ax.legend(line_list[aa],
                              label_list[aa],
                              loc=self.legend_loc,
                              markerscale=self.legend_marker_scale,
                              borderaxespad=self.legend_border_axes_pad,
                              labelspacing=self.legend_label_spacing,
                              handletextpad=self.legend_handle_text_pad,
                              borderpad=self.legend_border_pad,
                              prop={'size':max([self.font_size/(nr+1), 5])})
            if self.plot_style == 2:
                if self.plot_component == 2:
                    legend_ax_list = [ax_list[0]]
                    if plot_tipper == True:
                        legend_ax_list.append(ax_list[2])
                    for aa, ax in enumerate(legend_ax_list):
                        ax.legend(line_list[aa],
                                  label_list[aa],
                                  loc=self.legend_loc,
                                  markerscale=self.legend_marker_scale,
                                  borderaxespad=self.legend_border_axes_pad,
                                  labelspacing=self.legend_label_spacing,
                                  handletextpad=self.legend_handle_text_pad,
                                  borderpad=self.legend_border_pad,
                                  prop={'size':max([self.font_size/(nr+1), 5])})
                else:
                    legend_ax_list = ax_list[0:self.plot_component/2]
                    if plot_tipper == True:
                        if self.plot_component == 2:
                            legend_ax_list.append(ax_list[2])
                        elif self.plot_component == 4:
                            legend_ax_list.append(ax_list[4])
                    for aa, ax in enumerate(legend_ax_list):
                        ax.legend(line_list[aa],
                                  label_list[aa],
                                  loc=self.legend_loc,
                                  markerscale=self.legend_marker_scale,
                                  borderaxespad=self.legend_border_axes_pad,
                                  labelspacing=self.legend_label_spacing,
                                  handletextpad=self.legend_handle_text_pad,
                                  borderpad=self.legend_border_pad,
                                  prop={'size':max([self.font_size/(nr+1), 5])})
        
        ##--> BE SURE TO SHOW THE PLOT
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
# Exceptions
#==============================================================================
class ModEMError(Exception):
    pass