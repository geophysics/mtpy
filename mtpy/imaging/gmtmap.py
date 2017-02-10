#!/usr/bin/env python

"""
mtpy/utils/gmtmap.py

Functions for the generation of geographical maps, using GMT
Creates input files and template scripts

@Geoscience Australia, 2017
(Alison Kirkby)

"""

#=================================================================


import numpy as np

import os.path as op

import mtpy.modeling.modem_new as mtmn
import mtpy.analysis.pt as mtpt
import mtpy.utils.latlongutmconversion as utm2ll

from mtpy.utils.exceptions import *

#=================================================================


def get_closest_point(pointarray, index=None, target_point=None):
    """
    get closest index and value to target_point from pointlist
    provide either index or target point
    if no target point or index provided, use first point in pointarray as default
    
    """
    if index is None:
        if target_point is not None:
            diff = np.abs(pointarray - target_point)
            index = list(diff).index(min(diff))
        else:
            index = 0

    return index, pointarray[index]


class GMTEllipse():
    """
    class to write phase tensor data to plot in GMT
    """
    def __init__(self, **kwargs):
        self.workdir = kwargs.pop('workdir',None)
        self.filename = kwargs.pop('filename',None)
        self.data_type = None # 'data', 'response', or 'residual'
        self.data_array = kwargs.pop('data_array',None)
        self.colorby = kwargs.pop('colorby','phimin')
        self.clim = kwargs.pop('clim', None)
        self.cmap = 'polar'
        self.mt_dict = None
        
        self.ellipse_scale = kwargs.pop('ellipse_scale', 1.)

        self.period_array = None
        self.plot_period = kwargs.pop('plot_period',None)
        self.plot_period_index = kwargs.pop('plot_period_index',None)
        
        

    def _check_data_type(self,datafile,respfile):
        if ((datafile is None) and (self.data_type in ['data','residual'])):
            self.data_type = None
            print "Warning - provided input files and specified data type are incompatible: updating data_type"
        if ((respfile is None) and (self.data_type in ['response','residual'])):
            self.data_type = None
            print "Warning - provided input files and specified data type are incompatible: updating data_type"        

    
    def _get_climits(self):
        if self.data_array is not None:
            self.clim = (min(self.data_array[:,2]),max(self.data_array[:,2]))
        else:
            self.clim = (0,90)
            print "Cannot get color limits from data, please provide data array"


    def _construct_filename(self):
        
        cbystr,dtypestr = '',''
        
        if self.data_type is not None:
            dtypestr = self.data_type
            
        if self.colorby is not None:
            cbystr = self.colorby
            
        if self.plot_period is None:
            self.get_plot_period_index()
        
        # make a file extension based on period
        if self.plot_period >= 1.:
            ppstr = '%1i'%round(self.plot_period)
        else:
            nzeros = np.abs(np.int(np.floor(np.log10(self.plot_period))))
            fmt = '%0'+str(nzeros+1)+'i'
            ppstr = fmt%(period*10**nzeros)        
            
        
        self.filename = 'ellipse_{}_{}.{}'.format(dtypestr,cbystr,ppstr)
        
        

    def get_plot_period_index(self):
        
        self.plot_period_index, self.plot_period = \
        get_closest_point(np.log10(self.period_array), 
                          index=self.plot_period_index, 
                          target_point=np.log10(self.plot_period))
        self.plot_period = 10**self.plot_period
        

    def read_ModEM_data(self,datafile=None,respfile=None):
        
        if self.workdir is None:
            if datafile is not None:
                self.workdir = op.dirname(datafile)
            elif respfile is not None:
                self.workdir = op.dirname(respfile)
            else:
                print "Please provide data and/or response file"
                return
                
        # check data type compatible with provided input files
        self._check_data_type(datafile,respfile)

        # determine whether to automatically choose data type
        if self.data_type is None:
            find_datatype = True
        else:
            find_datatype = False
        
        # read files
        if datafile is not None:
            mdObj = mtmn.Data()
            mdObj.read_data_file(datafile)
            if self.workdir is None:
                self.workdir = op.dirname(datafile)
            if find_datatype:
                self.data_type = 'data'
            self.period_array = mdObj.period_list
        if respfile is not None:
            mrObj = mtmn.Data()
            mrObj.read_data_file(respfile)
            if self.workdir is None:
                self.workdir = op.dirname(respfile)
            if find_datatype:
                if self.data_type == 'data':
                    self.data_type = 'residual'
                else:
                    self.data_type = 'response'
            self.period_array = mrObj.period_list
        
        # get period index and period for plotting
        self.get_plot_period_index()
        
        # get mt_dict containing data, responses, or residual depending on data_type
        if self.data_type == 'data':
            self.mt_dict = mdObj.mt_dict
        elif self.data_type == 'response':
            self.mt_dict = mrObj.mt_dict
        elif self.data_type == 'residual':
            self.mt_dict = {}
            for key in mdObj.mt_dict.keys():
                self.mt_dict[key] = mtpt.ResidualPhaseTensor(pt_object1=mdObj.mt_dict[key], 
                                                             pt_object2=mrObj.mt_dict[key])


    def build_data_array(self):
        
        if self.mt_dict is None:
            print "Cannot save GMT, please read a ModEM data and/or response file first"
            
        self.data_array = np.zeros((len(self.mt_dict),6))
        
        for i,key in enumerate(self.mt_dict.keys()):
            for ii, att in enumerate(['lon', 'lat', self.colorby,'azimuth','phimin','phimax']):
                if ii < 2:
                    self.data_array[i,ii] = getattr(self.mt_dict[key],att)
                else:
                    self.data_array[i,ii] = getattr(self.mt_dict[key].pt,att)[0][self.plot_period_index]


        # normalise by maximum value of phimax
        norm = self.ellipse_scale/np.amax(self.data_array[:,4])
        self.data_array[:,5] *= norm
        self.data_array[:,4] *= norm
        
        # correct azimuth so it is positive anticlockwise from positive x axis
        if self.data_type != 'residual':
            self.data_array[:,3] = 90. - self.data_array[:,3]
            

        # if clim not provided, get it from the data
        if self.clim is None:
            self._get_climits()
            
             
    def write_gmtdata(self,savepath=None):
        
        if self.data_array is None:
            self.build_data_array()
        
        if self.filename is None:
            self._construct_filename()
        
        if savepath is None:
            savepath = self.workdir
        
        # write to text file in correct format
        fmt = ['%+11.6f','%+10.6f'] + ['%+9.4f']*2 +['%8.4f']*2
        np.savetxt(op.join(savepath,self.filename),self.data_array,fmt=fmt)
    

class GMTResistivity():
    """
    class to write resistivity data to plot in GMT
    """
    def __init__(self, **kwargs):
        self.workdir = kwargs.pop('workdir',None)
        self.filename = kwargs.pop('filename',None)
        self.data_array = kwargs.pop('data_array',None)
        self.clim = kwargs.pop('clim', None)
        self.cmap = 'polar'
        self.colorby = 'Resistivity'

        self.depth = kwargs.pop('depth',None) # list of depths in the model
        self.plot_depth = kwargs.pop('plot_depth',None)
        self.plot_depth_index = kwargs.pop('plot_depth_index',None)
        
        self.longitude = None # longitude of resistivity points
        self.latitude = None # latitude of resistivity points


    def _get_climits(self):
        if self.data_array is not None:
            self.clim = (min(self.data_array[:,2]),max(self.data_array[:,2]))
        else:
            self.clim = (0,2)
            print "Cannot get color limits from data, please provide data array"


    def _construct_filename(self):

        if self.plot_depth is None:
            self.get_plot_depth_index()
        
        self.filename = 'resistivity_%1im.xyz'%self.plot_depth


    def get_plot_depth_index(self):
        
        self.plot_depth_index, self.plot_depth = \
        get_closest_point(self.depth, index=self.plot_depth_index, 
                          target_point=self.plot_depth)


    def read_ModEM_model(self, model_fn, center_position, epsg, out_epsg=4326):
        """
        read in a ModEM model. Need to provide model filename, projection epsg 
        number, and real-world center position [easting, northing]
        """    
        
        # get workdir
        if self.workdir is None:
            self.workdir = op.dirname(model_fn)
        
        # read model file
        mObj = mtmn.Model()
        mObj.read_model_file(model_fn=model_fn)
        
        # get model x, y and depths from model (convert to cell centres)
        centrex, centrey, self.depth = [np.mean([arr[1:], arr[:-1]],axis=0) \
                                        for arr in [mObj.grid_east, mObj.grid_north, mObj.grid_z]]
        
        # get depth index and/or depth to plot
        self.get_plot_depth_index()    
        
        # get appropriate resistivity slice
        self.resistivity_slice = mObj.res_model[:,:,self.plot_depth_index]
        
        # get x, y in local coordinates and convert to easting and northing
        easting, northing = np.meshgrid(centrex + center_position[0],
                                        centrey + center_position[1])
        
        # project to lat, long
        self.latitude, self.longitude = utm2ll.project(easting,northing,epsg,out_epsg)
        
        
    def build_data_array(self):
        """
        build an array to contain resistivity data
        
        """
        # check if the input data exist
        if self.resistivity_slice is None:
            print "Cannot save GMT, please read a ModEM model file first"
            return
            
        # make x, y, z array, converting resistivity to log 10
        self.data_array = np.vstack([self.longitude.flatten(),
                                     self.latitude.flatten(),
                                     np.log10(self.resistivity_slice.flatten())]).T

        # if clim not provided, get it from the data array
        if self.clim is None:
            self._get_climits()        


    def write_gmtdata(self,savepath=None):
        
        if self.data_array is None:
            self.build_data_array()
        
        if self.filename is None:
            self._construct_filename()
        
        # write to text file in correct format
        fmt = ['%+11.6f','%+10.6f','%9.3e']
        np.savetxt(op.join(savepath,self.filename),self.data_array,fmt=fmt)



class GMTScript():
    """
    class to write a template gmt script for plotting data in gmt
    
    """

    
    def __init__(self, **kwargs):
        
        self.workdir = kwargs.pop('workdir',None)
        self.xlim = kwargs.pop('xlim', None)
        self.ylim = kwargs.pop('ylim', None)
        self.pad = kwargs.pop('pad', None)
        self.plotdata_dict = kwargs.pop('plotdata_dict',{}) # dict containing GMT Data objects
        self.mapsize = kwargs.pop('mapsize', '18c')
        self.psfilename = kwargs.pop('psfilename','map.ps')
        self.gmtset = {'FORMAT_GEO_MAP':'ddd:mm:ss',
                       'FONT_ANNOT_PRIMARY':'9p,Helvetica,black',
                       'MAP_FRAME_TYPE':'fancy'}
        
        # populate plot data dictionary
        for key in kwargs.keys():
            if key.lower() in ['ellipse','resistivity']:
                self.plotdata_dict[str.capitalize(key.lower())] = kwargs[key]
                
        if None in [self.xlim,self.ylim]:
            self._get_spatial_limits()
            

    def _get_spatial_limits(self):
        
        if len(self.plotdata_dict)== 0:
            print "Cannot get xlim and ylim, no data provided"
            return
            
        xlim_data = np.zeros((len(self.plotdata_dict),2))
        ylim_data = np.zeros((len(self.plotdata_dict),2))
        
        for i,dObj in enumerate(self.plotdata_dict.values()):
            xlim_data[i,0], ylim_data[i,0] = np.amin(dObj.data_array[:,:2],axis=0)
            xlim_data[i,1], ylim_data[i,1] = np.amax(dObj.data_array[:,:2],axis=0)

        self.xlim = tuple(np.amin(xlim_data,axis=0))
        self.ylim = tuple(np.amin(ylim_data,axis=0))
        print xlim_data,ylim_data
        

    def _get_scalebar_latitude(self):
        self.scalebar_latitude = int(round(ymax + ymin)/2.)
        
        
    def _get_tick_spacing(self):
        self.tick_spacing = int(np.round(20.*(xmax - xmin),tr))


    def build_gmt_lines(self):
        
        gmtlines = ['w={}'.format(self.xlim[0]),
                    'e={}'.format(self.xlim[1]),
                    's={}'.format(self.ylim[0]),
                    'n={}'.format(self.ylim[1]),
                    r"wesn=$w/$s/$e/$n'r'",
                    '',
                    '# define output file and remove it if it exists',
                    'PS={}'.format(self.psfilename),
                    'rm $PS',
                    '',
                    '# set gmt parameters']
        
        # add gmtset parameters
        gmtlines += ['gmtset {} {}'.format(key,self.gmtset[key]) for key in self.gmtset.keys()] + ['']
        
        # make color palettes
        for key in self.plotdata_dict.keys():
            if key in ['Ellipse','Resistivity']:
                gmtlines.append('# make colour palette for {}'.format(key))
                dObj = self.plotdata_dict[key]
                if dObj.clim is None:
                    dObj._get_climits()
                gmtlines.append('makecpt -C{} -T{}/{} -Z > {}.cpt'.format(dObj.cmap,dObj.clim[0],dObj.clim[1],dObj.colorby))
                gmtlines.append('')

        # plot resistivity if available
        oflag, kflag = False,True
        if 'Resistivity' in self.plotdata_dict.keys():
            dObj = self.plotdata_dict['Resistivity']
            grdname = dObj.filename.split('.')[0]+'.grd'
            gmtlines += ['# make depth slice',
                         'file={}'.format(dObj.filename),
                         'surface $file -R$wesn -I10m -V -T0.75 -G{}'.format(grdname),
                         'grdimage {} -Cres.cpt -JM{} -Y6c -R$wesn -V -K -P >> $PS'.format(grdname,self.mapsize)
                         ]
            oflag,kflag = True,False
            
        # draw coastline
        gmtlines += ['# draw coastline',
                     'pscoast -R$wesn -JM18c -W0.5p -Ba1f1/a1f1WSen -Gwhite -Slightgrey -Lfx14c/1c/{}/{}+u -Df -P{}{} >> $PS'.format(self.scalebar_latitude,self.tick_spacing,' -O'*oflag, ' -K'*kflag), '']
        oflag, kflag = True,False

        # draw ellipses if available
        if 'Ellipse' in self.plotdata_dict.keys():
            dObj = self.plotdata_dict['Ellipse']
            gmtlines += ['# draw ellipses','psxy {} -R -J -P -Se -C{}.cpt -W0.01p{}{} >> $PS'.format(dObj.filename, dObj.colorby,' -O'*oflag, ' -K'*kflag),'']
            
        # make color scale
        for key in self.plotdata_dict.keys():
            label,position = None, 1
            if key == 'Ellipse':
                label = str.capitalize(self.plotdata_dict.colorby)
                position += 4
            elif key == 'Resistivity':
                label = 'Log resistivity'
                position += 4
            if label is not None:
                gmtlines += ['psscale -Cres.cpt -D4c/{}c/3c/0.5ch -B:"{}": --FONT_LABEL=9p{}{} >> $PS'.format(position,label,' -O'*oflag, ' -K'*kflag)]

        # get rid of '-K' in final line
        gmtlines[-1] = gmtlines[-1].replace(' -K','')

        # convert ps file to png
        gmtlines += ['# save to png','ps2raster -Tg -A -E400 $PS']        
        
    