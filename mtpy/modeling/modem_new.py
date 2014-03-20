#!/usr/bin/env python
"""
==================
ModEM
==================


# Generate data file for ModEM
# by Paul Soeffky 2013
# revised by LK 2014

"""

import os,sys
import mtpy.core.edi as mtedi
import numpy as np
import mtpy.utils.latlongutmconversion as utm2ll
import mtpy.utils.merge_periods as merge_periods
import glob


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
                              'Full_Vertical_Components':['txy', 'tyx']}
        
                              
        self.header_strings = \
        ['#Created using MTpy error floor {0:.0f}%\n'.format(self.error_floor), 
        '# Period(s) Code GG_Lat GG_Lon X(m) Y(m) Z(m) Component Real Imag Error']

        
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
                                               ('rel_north', np.float)])
                                        
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
            
        ns = len(self.edi_obj_list)
        nf = len(self.period_list)
        
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
                        self.data_array[ii][ff]['txy'] = \
                                                edi.Tipper.tipper[jj, 0, 0]
                        self.data_array[ii][ff]['tyx'] = \
                                                edi.Tipper.tipper[jj, 0, 1]
                        
                        self.data_array[ii][ff]['txy_err'] = \
                                                edi.Tipper.tippererr[jj, 0, 0]
                        self.data_array[ii][ff]['tyx_err'] = \
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

        dlines = self.header_strings            
        for inv_mode in self.inv_mode_dict[self.inv_mode]:
            dlines.append('> {0}\n'.format(inv_mode))
            dlines.append('> exp({0}i\omega t)\n'.format(self.wave_sign))
            if inv_mode.find('Impedance') > 0:
                dlines.append('> {0}\n'.format(self.units))
            elif inv_mode.find('Vertical') >=0:
                dlines.append('> []\n')
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
                            eas = '{0:> 12.3f}'.format(self.coord_array[ss]['rel_east'])
                            nor = '{0:> 12.3f}'.format(self.coord_array[ss]['rel_north'])
                            ele = '{0:> 12.3f}'.format(0)
                            com = '{0:>4}'.format(comp.capitalize())
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
                    
                                               
            
        
            
        
        

        
        
                        
            

##start Impedance tensor part ---------------------------------------------
#
#header_string += '> {0} {1}\n'.format(lat0,lon0)
#
#impstring = ''
#periodlist = []
#
#components = ['XX','XY','YX','YY']
#
##loop for reading in periods
## in case merging is requested, updating period
#period_dict = {}
#
#for idx_edi, edi in enumerate(lo_ediobjs):
#    freq2 = edi.freq
#    periods=1/freq2
#    periods = [np.round(i,5) for i in periods]
#    periodlist.extend(periods)
#
#periodlist = sorted(list(set(periodlist)),reverse=False)
#    
#if merge_periods == True:
#    #mp.plot_merging(periodlist,merge_threshold)
#    new_periods = mp.merge_periods(periodlist,merge_threshold)
#else:
#    new_periods = periodlist[:]
##setting up a dictionary for old and new period
#for idx,per in enumerate(periodlist):
#    period_dict[str(per)] = new_periods[idx]
#
#
#periodlist = []
#
#
#for idx_edi, edi in enumerate(lo_ediobjs):
#
#    freq2 = edi.freq
#    periods=1/freq2
#
#    zerr=edi.Z.zerr
#    zval=edi.Z.z
#
#    northing = rel_coords[idx_edi,1]
#    easting = rel_coords[idx_edi,0]
#    
#    
#    #Generate Impedance Array
#    for i in range(len(periods)):
#
#        raw_period = periods[i]
#        raw_period = np.round(raw_period,5)
#        period = float(period_dict[str(raw_period)])
#        periodlist.append(period)
#
#        Z = zval[i]
#        Zerr = zerr[i]
#
#        period_impstring = ''
#
#        for i in range(2):
#            for j in range(2):
#                try:
#                    rel_err = Zerr[i,j]/np.abs(Z[i,j])
#                    if rel_err < errorfloor/100.:
#                        raise
#                except:
#                    Zerr[i,j] = errorfloor/100. * np.abs(Z[i,j])
#
#                comp = components[2*i+j]
#                period_impstring += '{0:.5f} {1} '.format(period,edi.station)
#                period_impstring += '{0:.3f} {1:.3f} '.format(edi.lat,edi.lon)
#                period_impstring += '{0:.3f} {1:.3f} {2} '.format(northing, easting,0.)
#                period_impstring += 'Z{0} {1:.5E} {2:.5E} {3:.5E} '.format(comp,float(np.real(Z[i,j])),
#                                                float(np.imag(Z[i,j])), Zerr[i,j] )
#                period_impstring += '\n'
#
#        impstring += period_impstring
#
#
#n_periods = len(set(periodlist))
#
#
#print 'Z periods: ',n_periods , 'files:', len(lo_ediobjs)
#
#header_string += '> {0} {1}\n'.format(n_periods,len(lo_ediobjs))
#
##print outstring
#data=open(r'ModEMdata.dat', 'w')
#data.write(header_string)
#data.write(impstring)
#data.close()
#
#
##start Tipper part ---------------------------------------------
#
#errorfloor *= 2.
#
##Tipper part
#header_string = ''
#header_string += """# Period(s) Code GG_Lat GG_Lon X(m) Y(m) Z(m) Component Real Imag Error
#> Full_Vertical_Components \n> exp(-i\omega t)\n> []
#> 0.00
#"""
#
#header_string += '> {0} {1}\n'.format(lat0,lon0)
#
#tipperstring = ''
#periodlist = []
#n_periods = 0
#components = ['X','Y']
#
#stationlist = []
#
#
#for idx_edi, edi in enumerate(lo_ediobjs):
#
#    freq2 = edi.freq
#    periods=1/freq2
#
#    tippererr=edi.Tipper.tippererr
#    tipperval=edi.Tipper.tipper
#
#    northing = rel_coords[idx_edi,1]
#    easting = rel_coords[idx_edi,0]
#    
#    #Generate Tipper Array
#    for i in range(len(periods)):
#
#        period = periods[i]
#        period = np.round(period,5)
#        try:
#            T = tipperval[i][0]
#        except:
#            continue
#        try:
#            Terr = tippererr[i][0]
#        except:
#            Terr = np.zeros_like(T,'float')
#
#        if np.sum(np.abs(T)) == 0:
#            continue
#
#        stationlist.append(e.station)
#
#        periodlist.append(period)
#
#
#        period_tipperstring = ''
#
#
#        for i in range(2):
#        
#            try:
#                rel_err = Terr[i]/np.abs(T[i])
#                if rel_err < errorfloor/100.:
#                    raise
#            except:
#                Terr[i] = errorfloor/100. * np.abs(T[i])
#
#            comp = components[i]
#            period_tipperstring += '{0:.5f} {1} '.format(period,edi.station)
#            period_tipperstring += '{0:.3f} {1:.3f} '.format(edi.lat,edi.lon)
#            period_tipperstring += '{0:.3f} {1:.3f} {2} '.format(northing, easting,0.)
#            period_tipperstring += 'T{0} {1:.5E} {2:.5E} {3:.5E} '.format(comp,float(np.real(T[i])),
#                                            float(np.imag(T[i])), Terr[i] )
#            period_tipperstring += '\n'
#
#        tipperstring += period_tipperstring
#
#
#n_periods = len(set(periodlist))
#n_stations = len(set(stationlist))
#if use_tipper is True:
#    print 'Tipper periods: ',n_periods, 'stations:', n_stations
#else:
#    print 'no Tipper information in data file'
#
#header_string += '> {0} {1}\n'.format(n_periods,len(lo_ediobjs))
#
#
#
#
#if (len(tipperstring)>0 ) and (use_tipper is True):
#    data = open(r'ModEMdata.dat', 'a')
#    data.write(header_string)
#    data.write(tipperstring.expandtabs(4))
#    data.close()
#
#
#print "end"
#
class ModEMError(Exception):
    pass