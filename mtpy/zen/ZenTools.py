# -*- coding: utf-8 -*-
"""
====================
ZenTools
====================

    * Tools for reading and writing files for Zen and processing software
    * Tools for copying data from SD cards
    * Tools for copying schedules to SD cards
    
    
Created on Tue Jun 11 10:53:23 2013

@author: jpeacock-pr
"""

#==============================================================================

import numpy as np
import time
import os
import struct
import string
import win32api
import shutil
from collections import Counter
import subprocess

#==============================================================================

class Zen3D(object):
    """
    Deal with the raw data output from the Zen box as Z3D files.
    
    Arguments:
    ----------
        **fn**: string
                full path to .Z3D file to be manipulated
                
                
    Methods:
    ---------
        **read_3d** : read 3D file making sure all the time stamps are 
                     correctly spaced.  The returned time series starts at 
                     the first stamp which has the correct amount of data
                     points between it and the next time stamp.  Note that
                     there are usually a few seconds at the end and maybe 
                     beginning that aren't correct because the internal 
                     computer is busy switchin sampling rate.
                     
        **get_gps_stamp_location** : locates the gps stamp location
        
        **get_gps_time** : converts the gps counts to seconds
        
        **get_date_time** : converts gps seconds into the actual date and time
        
    =================== =======================================================
    Attributes           Description
    =================== =======================================================
        ch_adcard_sn     serial number of a/d card in channel
        ch_cmp           MT component of channel
        ch_length        distance between electrodes for channel, 
                         doesn't matter for magnetic channels
        ch_number        number of channel
        date_time        np.ndarray of date,time of gps stamps
        df               sampling rate 
        fn               full path to file name read in
        gps_diff         difference between gps time stamps
        gps_lst          list of gps stamps
        gps_time         np.ndarray of gps times from time stamps
        gps_week         gps week
        header_dict      dictionary of header parameters
        log_lines        list of information to write into a log file later
        meta_dict        dictionary of meta data parameters
        rx_stn           name of station
        start_time       starting time and date of first time stamp with 
                         correct number of samples
        temperature      np.ndarray of temperature measurements at each time 
                         stamp
        time_series      np.ndarray of time series data in counts
        tx_id            name of transmitter if used
        verbose          [ True | False ] for printing information in 
                         interpreter
                         
        _data_type       np.dtype to convert binary formatted string
        _data_types      list of data types in binary formatted string
        _gps_epoch       gps_epoch in time.gmtime format. 
        _gps_stamp       string of gps_stamp 
        _header_len      length of header string in bytes. (512)
        _meta_len        length of meta data in bytes. (512)
        _raw_data        data in binary format
        _seconds_diff    difference in seconds from start time to look for 
                         gps stamp. *default* is 5
        _stamp_len       length of gps time stamp in bits
        _stamp_lst       list of gps time stamp variables
        _week_len        length of a gps week in seconds
    =================== =======================================================
    
    """
    
    def __init__(self, fn=None, **kwargs):
        
        self.fn = fn
        self._header_len = kwargs.pop('header_len', 512)
        self._meta_len = kwargs.pop('meta_len', 512)
        self._stamp_len = kwargs.pop('stamp_len', 36)
        self._gps_stamp = kwargs.pop('gps_stamp', '\xff\xff\xff\xff')
        
        self._stamp_lst = ['gps', 'time', 'lat', 'lon', 'status', 'status2', 
                           'gps_accuracy', 'something', 'temperature']
                           
        self._data_types = [np.int32, np.int32, np.float32, np.float32, 
                            np.uint32, np.int32, np.float32, np.float32, 
                            np.float32]
                            
        self._data_type = np.dtype([(st, dt) for st, dt in 
                                     zip(self._stamp_lst, self._data_types)])
                                     
        self._week_len = 604800
        self._gps_epoch = (1980, 1, 6, 0, 0, 0, -1, -1, 0)
        self._leap_seconds = 16
        
        #seconds different between scheduling time and actual collection time
        self._seconds_diff = 5 
        
        self.log_lines = []
        self.verbose = True
        self._skip_sample_tolerance = 5
        self.sample_diff_lst = []
    
    
    def read_3d(self):
        """
        read in the time series and gps time stamps.
        
        Makes sure that the number of samples between each time stamp is
        the sampling rate.  If it is not an error is raised.  
        
        Creates a time series that starts at the time where the first gps
        time stamp has the correct number of points, and stops where the first
        incorrect number of points occurs.  A corresponding time,date array
        is created.
        
        
        """
        #read in as a binary file.
        raw_data = open(self.fn, 'rb').read()
        self._raw_data = raw_data
        
        try:
            self.log_lines[0] != '-'*72+'\n'
        except IndexError:
            self.log_lines.append('-'*72+'\n')
            self.log_lines.append('--> Reading File: {0}\n'.format(self.fn))
        
        #number of bytes in the file
        num_bytes = len(raw_data)
        
        #beginning index of data blocks
        ds = self._header_len+self._meta_len
        
        #----read in header information----------------------------------------
        header_lst = raw_data[0:self._header_len].replace('\n', ',').split(',')
        
        header_dict = {}
        for hh in header_lst:
            if hh != '' and hh.find('builddate') == -1:
                hkv = hh.split(':')
                if len(hkv) == 2:
                    if hkv[0].lower() == 'period' or \
                        hkv[0].lower() == 'duty':
                        try:
                            header_dict[hkv[0].strip().lower()] +=\
                                                                hkv[1].strip()
                        except KeyError:
                            header_dict[hkv[0].strip().lower()] =\
                                                                hkv[1].strip()
                    else:
                        header_dict[hkv[0].strip().lower()] = hkv[1].strip()
                elif len(hkv) == 3:
                    header_dict['start_time'] = hh.strip()
                else:
                    pass
            elif hh == '':
                pass
            else:
                hline = hh.split(';')
                for ll in hline:
                    if ll.find('builddate') > 0:
                        hlst = ll.split('&')
                        for kk in hlst:
                            klst = kk.split(':')
                            header_dict[klst[0].strip().lower()] = klst[1].strip()
                    else:
                        hlst = ll.split(':')
                        try:
                            header_dict[hlst[0].strip().lower()] = hlst[1].strip()
                        except IndexError:
                            pass
        #make attributes that will be useful latter
        self.header_dict = header_dict
        self.df = float(header_dict['a/d rate'])
        self.gain = float(header_dict['a/d gain'])
        self.gps_week = int(header_dict['gpsweek'])
        try:
            self.schedule_date = header_dict['schedule for this file']
        except KeyError:
            self.schedule_date = header_dict['schedule']
        self.schedule_time = header_dict['start_time']
        
        #get the start date/time in UTC time
        self.start_time = self.compute_schedule_start(self.schedule_date, 
                                                      self.schedule_time)
                                            
        self.header_dict['schedule'] = self.start_time.split(',')[0]
        self.header_dict['start_time'] = self.start_time.split(',')[1]
        self.schedule_date = self.start_time.split(',')[0]
        self.schedule_time = self.start_time.split(',')[1]
        
        #--> get serial number of a/d board
        try:
            self.ch_adcard_sn = header_dict['serial']
        except KeyError:
            self.ch_adcard_sn = header_dict['brd339 serial']
        
        #---read in meta raw_data----------------------------------------------------------
        meta_lst = raw_data[self._header_len-1:ds].replace('\n','|').split('|')
        
        meta_dict = {}
        for mm in meta_lst:
            mlst = mm.split(',')
            if len(mlst) == 2:
                meta_dict[mlst[0].strip().lower()] = mlst[1].strip().lower()
            else:
                pass
        self.meta_dict = meta_dict  
        self.ch_number = meta_dict['ch.number']
        self.ch_cmp = meta_dict['ch.cmp'].replace('b','h')
        self.ch_length = meta_dict['ch.varasp']
        self.rx_stn = meta_dict['rx.stn']
        self.tx_id = meta_dict['tx.id']
        #---read in gps raw_data-------------------------------------------------
        #sampling rate times 4 bytes for 32 bit measurement
        df = int(header_dict['a/d rate'])      
        dt = df*4
        
        #length of data block plus gps stamp
        block_len = self._stamp_len+dt
        
        #number of data blocks
        num_blocks = int(np.ceil(num_bytes/float(block_len)))
        
        #get position of gps stamps
        gps_lst = np.zeros(num_blocks, dtype=np.int)
        #gps_times = np.zeros(num_blocks)
        
        gps_dict = dict([(key, np.zeros(num_blocks, dtype=dtp)) 
                          for key, dtp in zip(self._stamp_lst, 
                                              self._data_types)])
        #make the time array floats instead of ints so can get the decimal 
        #place if it isn't 0.
        gps_dict['time'] = gps_dict['time'].astype(np.float32)
        
        #get gps information from the data
        #get first time stamp that matches the starting time
        s1 = 0
        gps_lst[0] = self.get_gps_stamp_location()
        gps_info = np.fromstring(raw_data[gps_lst[0]:gps_lst[0]+self._stamp_len], 
                                 dtype=self._data_type)
        gps_info['time'] = gps_info['time'].astype(np.float32)
        gps_info['time'] = self.get_gps_time(gps_info['time'])
        start_test = self.get_date_time(self.gps_week, gps_info['time'])
        
        #--> test to make sure the first time corresponds to the scheduled 
        #start time
        time_stop = 0
        while start_test != self.start_time and s1 <= self._seconds_diff and \
                time_stop <= self._seconds_diff:
            s1 += 1
            gps_lst[0] = self.get_gps_stamp_location(gps_lst[0]+7)
            gps_info = np.fromstring(raw_data[gps_lst[0]:gps_lst[0]+\
                                                self._stamp_len], 
                                     dtype=self._data_type)
                                     
            gps_info['time'] = gps_info['time'].astype(np.float32)
            gps_info['time'] = self.get_gps_time(gps_info['time'])
            start_test = self.get_date_time(self.gps_week, gps_info['time'])
            if s1 == self._seconds_diff:
                s1 = 0
                self.start_time = self.start_time[:-2]+\
                                 '{0:02}'.format(int(self.start_time[-2:])+1)
                gps_lst[0] = self.get_gps_stamp_location()
                time_stop += 1  
       
       #----Raise an error if the first gps stamp is more than allowed time
        #    difference.
        if time_stop >= self._seconds_diff:
            raise IOError('GPS start time is more than '+\
                           '{0} '.format(self._seconds_diff)+\
                           'seconds different than scheduled start time of '+\
                           '{0}. \n '.format(self.start_time)+\
                           'Estimated start time is {0} +/- {1} sec'.format(
                           start_test, self._seconds_diff))
                     
        #put the information into the correct arrays via dictionary                         
        for jj, key in enumerate(self._stamp_lst):
            gps_dict[key][0] = gps_info[0][jj]
  
        #find the next time stamp
        for ii in range(s1,num_blocks-1):
            sfind = self.get_gps_stamp_location(gps_lst[ii-1]+7)
            #make sure it isn't the same time stamp as before
            if sfind != gps_lst[ii-1] and sfind != -1:
                gps_lst[ii] = self.get_gps_stamp_location(gps_lst[ii-1]+7)
                
                #get numbers from binary format
                try:
                    gps_info = \
                           np.fromstring(raw_data[sfind:sfind+self._stamp_len],
                                         dtype=self._data_type)
                                         
                    gps_info['time'] = gps_info['time'].astype(np.float32)
                    
                    #get gps integer into real time
                    gps_info['time'] = self.get_gps_time(gps_info['time'])
                    
                    #put data into its correct array
                    for jj, key in enumerate(self._stamp_lst):
                        gps_dict[key][ii] = gps_info[0][jj]
                except ValueError:
                    print 'Ran into end of file, gps stamp not complete.'+\
                          ' Only {0} points.'.format(len(raw_data[sfind:]))
        
        #get only the values that are non zero
        gps_dict['time'] = gps_dict['time'][np.nonzero(gps_dict['time'])] 

        num_samples = len(gps_dict['time'])
        
        #calculate the difference between time stamps
        gps_diff = np.array([gps_dict['time'][ii+1]-gps_dict['time'][ii] 
                             for ii in range(num_samples-1)])
        
        #check for any spots where gps was not locked or mised a sampling interval
        bad_lock = np.where(gps_diff[np.nonzero(gps_diff)] != 1.0)[0]
        
        if len(bad_lock) > 0:
            if self.verbose:
                print '\n'+'*'*20+'BAD GPS LOCK'+'*'*20
            self.log_lines.append(' '*4+'*'*20+'BAD GPS LOCK'+'*'*20+'\n')
            for bb in bad_lock:
                if self.verbose:
                    print 'point {0:^15}, gps diff {1:^15}'.format(gps_lst[bb],
                                                                   gps_diff[bb])
                
                self.log_lines.append(' '*4+\
                                      'point {0:^15},'.format(gps_lst[bb])+\
                                      'gps diff {0:^15}\n'.format(gps_diff[bb]))
            
            if self.verbose:                                                
                print '*'*52+'\n'
            self.log_lines.append(' '*4+'*'*52+'\n')

        #need to be sure that the number of data points between time stamps is 
        #equal to the sampling rate, if it is not then remove that interval.  
        #Most likely it is at the beginning or end of time series.
        dsamples = np.array([(gps_lst[nn+1]-gps_lst[nn]-self._stamp_len-df*4)/4 
                              for nn in range(num_samples)])
        
        bad_interval = np.where(abs(dsamples)>self._skip_sample_tolerance)[0]
        bmin = 0
        bmax = num_samples
        if len(bad_interval) > 0:        
            #need to locate the bad interval numbers
            for bb in bad_interval:
                if bb <= 10:
                    bmin = bb+1
                if bb > num_samples-10:
                    bmax = bb
        
            gps_lst = gps_lst[bmin:bmax]
            
        num_samples = len(gps_lst)
        if self.verbose:
            print 'Found {0} gps time stamps, '.format(num_samples)+\
                  'with equal intervals of {0} samples'.format(int(self.df))
              
        self.log_lines.append(' '*4+\
                            'Found {0} gps time stamps, '.format(num_samples)+\
                  'with equal intervals of {0} samples\n'.format(int(self.df)))
        
        #read in data
        data_array = np.zeros((num_samples+1)*df, dtype=np.float32)
        for ll, kk in enumerate(gps_lst[0:-1]):
            pdiff = ((gps_lst[ll+1]-(kk+self._stamp_len))-(df*4))/4
            self.sample_diff_lst.append(pdiff)
            data_array[ll*df:(ll+1)*df+pdiff] = \
                    np.fromstring(raw_data[kk+self._stamp_len:gps_lst[ll+1]], 
                          dtype=np.int32) 
                          
        if sum(self.sample_diff_lst) != 0:
            if self.verbose:
                print 'time series is off by {0} seconds'.format(
                                           float(sum(self.sample_diff_lst))/df)
                self.log_lines.append('time series is off by {0} seconds'.format(
                                          float(sum(self.sample_diff_lst))/df))
                                           
        #get only the non-zero data bits, this is dangerous if there is 
        #actually an exact 0 in the data, but rarely happens 
        self.time_series = data_array[np.nonzero(data_array)]
        
        #need to cut all the data arrays to have the same length and corresponding 
        #data points
        for key in gps_dict.keys():
            gps_dict[key] = gps_dict[key][bmin:bmax]
        
        #make attributes of imporant information
        self.gps_diff = gps_diff[bmin:bmax]
        self.gps_time = gps_dict['time']
        self.gps_lst = gps_lst
        self.temperature = gps_dict['temperature']
        
        
        self.date_time = np.zeros_like(gps_dict['time'], dtype='|S24')

        for gg, gtime in enumerate(gps_dict['time']):
            self.date_time[gg] = self.get_date_time(self.gps_week, gtime)
        
        try:
            self.start_time = self.date_time[0]
            if self.verbose:
                print 'Starting time of time series is '+\
                        '{0} UTC'.format(self.date_time[0])
            self.log_lines.append(' '*4+'Starting time of time series is '+\
                                  '{0} UTC\n'.format(self.date_time[0]))
        except IndexError:
            print 'No quality data was collected'
            self.log_lines.append(' '*4+'No quality data was collected\n')
            self.start_time = None
        
        
    def compute_schedule_start(self, start_date, start_time, 
                               leap_seconds=None):
        """
        compute the GMT time for scheduling from start time of the gps 
        according to the leap seconds.
        
        Arguments:
        -----------
            **start_date**: YYYY-MM-DD
                            schedule start date
                            
            **start_time**: hh:mm:ss
                            time of schedule start on a 24 hour basis
            
            **leap_seconds**: int
                              number of seconds that GPS is off from UTC time.
                              as of 2013 GPS is ahead by 16 seconds.
                              
        Returns:
        --------
            **ndate_time**: YYYY-MM-DD,hh:mm:ss
                            calibrated date and time in UTC time.
        
        """                                
    
        if leap_seconds is not None:
            self._leap_seconds = leap_seconds
            
        year, month, day = start_date.split('-')
        
        hour, minutes, seconds = start_time.split(':')
        
        new_year = int(year)
        new_month = int(month)
        new_day = int(day)
        new_hour = int(hour)
        new_minutes = int(minutes)
        new_seconds = int(seconds)-self._leap_seconds
       
        if new_seconds < 0:
            new_seconds = (int(seconds)-self._leap_seconds)%60
            new_minutes = int(minutes)-1
            if new_minutes < 0:
                new_minutes = (int(minutes)-1)%60
                new_hour = int(hour)-1
                if new_hour < 0:
                    new_hour = (int(hour)-1)%24
                    new_day = int(day)-1
                    if new_day <= 0:
                        new_day = (int(day)-1)%30
                        new_month = int(month)-1
                        print 'need to check date, have not implemented change'+\
                              'in days yet'
                              
        ndate_time = time.strftime("%Y-%m-%d,%H:%M:%S",
                                   (new_year, 
                                    new_month, 
                                    new_day, 
                                    new_hour, 
                                    new_minutes, 
                                    new_seconds, 0, 0, 0))
                                    
        return ndate_time
        
    def get_gps_stamp_location(self, start_index=None):
        """
        get the location in the data file where there is a gps stamp.  Makes
        sure that the time stamp is what it should be.
        
        Arguments:
        -----------
            **start_index**: int
                             starting point to look for the time stamp within
                             the file.
                             
        Returns:
        ---------
            **gps_index**: int
                           the index in the file where the start of the 
                           time stamp is.
        
        """
        
        gps_index = self._raw_data.find(self._gps_stamp, start_index)
        if self._raw_data[gps_index+4] == '\xff':
            gps_index += 1
            if self._raw_data[gps_index+4] == '\xff':
                gps_index += 1
                if self._raw_data[gps_index+4] == '\xff':
                    gps_index += 1
                    if self._raw_data[gps_index+4] == '\xff':
                        gps_index += 1
                        
        return gps_index
            

    def get_gps_time(self, gps_int, gps_week=0):
        """
        from the gps integer get the time in seconds.
        
        Arguments:
        ----------
            **gps_int**: int
                         integer from the gps time stamp line
                         
            **gps_week**: int
                          relative gps week, if the number of seconds is 
                          larger than a week then a week is subtracted from 
                          the seconds and computed from gps_week += 1
                          
        Returns:
        ---------
            **gps_time**: int
                          number of seconds from the beginning of the relative
                          gps week.
        
        """
            
        gps_seconds = gps_int/1024.
        
        gps_ms = (gps_seconds-np.floor(gps_int/1024.))*(1.024)
        
        cc = 0
        if gps_seconds > self._week_len:
            gps_week += 1
            cc = gps_week*self._week_len
            gps_seconds -= self._week_len
        
        gps_time = np.floor(gps_seconds)+gps_ms+cc
        
        return gps_time
        
    def get_date_time(self, gps_week, gps_time):
        """
        get the actual date and time of measurement GMT.  Note that GPS time is 
        off by 16 seconds from actual GMT time.
        
        Arguments:
        ----------
            **gps_week**: int
                          integer value of gps_week that the data was collected
            
            **gps_time**: int
                          number of seconds from beginning of gps_week
            
            **leap_seconds**: int
                              number of seconds gps time is off from UTC time.
                              It is currently off by 16 seconds.
                              
        Returns:
        --------
            **date_time**: YYYY-MM-DD,HH:MM:SS
                           formated date and time from gps seconds.
        
        
        """
        
        mseconds = gps_time % 1
        
        #make epoch in seconds, mktime computes local time, need to subtract
        #time zone to get UTC
        epoch_seconds = time.mktime(self._gps_epoch)-time.timezone
        
        #gps time is 14 seconds ahead of GTC time, but I think that the zen
        #receiver accounts for that so we will leave leap seconds to be 0        
        gps_seconds = epoch_seconds+(gps_week*self._week_len)+gps_time-\
                                                                self._leap_seconds

        #compute date and time from seconds
        (year, month, day, hour, minutes, seconds, dow, jday, dls) = \
                                                    time.gmtime(gps_seconds)
        
        date_time = time.strftime("%Y-%m-%d,%H:%M:%S",(year, month, day, hour, minutes, 
                                    int(seconds+mseconds), 0, 0, 0))
        return date_time

class ZenCache(object):
    """
    deals with cache files or combined time series files.
    
    ================== ========================================================
     Attributes         Description
    ================== ======================================================== 
    cal_data            list of calibrations, as is from file
    fn_lst              list of filenames merged together
    log_lines           list of information to put into a log file
    meta_data           dictionary of meta data key words and values
    nav_data            list of navigation data, as is from file
    save_fn             file to save merged file to
    ts                  np.ndarray(len(ts), num_channels) of time series
    verbose             [ True | False ] True prints information to console
    zt_lst              list of class: Zen3D objects
    _ch_factor          scaling factor for the channels, got this from Zonge
    _ch_gain            gain on channel, not sure of the format
    _ch_lowpass_dict    dictionary of values for lowpass filter, not sure how
                        they get the values
    _data_type          np.dtype of data type for cache block
    _flag               flag for new data block
    _nav_len            length of navigation information in bytes
    _stamp_len          length of new block stamp in bytes
    _type_dict          dictionary of cache block types, from Zonge.
    ================== ========================================================
    
    Methods:
    ---------
        * *check_sampling_rate* : makes sure sampling rate is the same for all
                                  files being merged.
        
        * *check_time_series* : makes sure all time series start at the same
                                time and have the same length.
                                
        * *write_cache_file* : writes a cache file for given filenames.
        
        * *read_cache* : reads in a cache file.
        
    :Example: ::
    
        >>> import ZenTools as zen
        >>> zc = zen.ZenCache()
        >>> # read a cache file
        >>> zc.read_cache(fn=r"/home/MT/mt01_20130601_190001_4096.cac")
        >>> # write a cache file
        >>> import os
        >>> file_path = r"/home/MT/Station1"
        >>> fn_lst = [os.path.join(file_path, fn) 
        >>> ...       for fn in os.listdir(file_path)
        >>> ...       if fn.find('.Z3D')>0]
        >>> zc.write_cache_file(fn_lst, r"/home/MT/Station1", station='s1')
        >>> Saved File to: /home/MT/Station1/Merged/s1_20130601_190001_4096.cac
        
    """   
    
    def __init__(self):
        
        self.fn_lst = None
        self.zt_lst = None
        self.save_fn = None
        self._ch_factor = '9.5367431640625e-10'
        self._ch_gain = '01-0'
        self._ch_lowpass_dict = {'256':'112', 
                           '1024':'576',
                           '4096':'1792'}
        self._flag = -1
        
        self._type_dict = {'nav' : 4,
                          'meta' :  514,
                          'cal' : 768,
                          'ts' : 16}
                          
        self._data_type = np.dtype([('len',np.int32), 
                                    ('flag', np.int32), 
                                    ('input_type', np.int16)])
        self._stamp_len = 10
        self._nav_len = 43
        
        self.nav_data = None
        self.cal_data = None
        self.ts = None
        self.verbose = True
        self.log_lines = []
        
        self.meta_data = {'SURVEY.ACQMETHOD' : ',timeseries',
                           'SURVEY.TYPE' : ',',
                           'LENGTH.UNITS' : ',m',
                           'DATA.DATE0' : '',
                           'DATA.TIME0' : '',
                           'TS.ADFREQ' : '',
                           'TS.NPNT': '',              
                           'CH.NUNOM' : ',',
                           'CH.FACTOR' : '',
                           'CH.GAIN' : '',
                           'CH.NUMBER' : '',
                           'CH.CMP' : '',
                           'CH.LENGTH' : '',
                           'CH.EXTGAIN' : '',
                           'CH.NOTCH' : '',
                           'CH.HIGHPASS' : '',
                           'CH.LOWPASS' : '',
                           'CH.ADCARDSN' : '',
                           'CH.STATUS' : ',',
                           'CH.SP' : ',',
                           'CH.GDPSLOT' : ',',
                           'RX.STN' : '',
                           'RX.AZIMUTH' : ',',
                           'LINE.NAME' : ',',
                           'LINE.NUMBER' : ',',
                           'LINE.DIRECTION' : ',',
                           'LINE.SPREAD' : ',',
                           'JOB.NAME' : ',',
                           'JOB.FOR' : ',',
                           'JOB.BY' : ',',
                           'JOB.NUMBER' : ',',
                           'GDP.File' : ',',
                           'GDP.SN' : ',',
                           'GDP.TCARDSN' : ',',
                           'GDP.NUMCARD' : ',',
                           'GDP.ADCARDSN' : ',',
                           'GDP.ADCARDSND' : ',',
                           'GDP.CARDTYPE' : ',',
                           'GDP.BAT' : ',',
                           'GDP.TEMP' : ',',
                           'GDP.HUMID' : ',',
                           'TS.NCYCLE' : ',',
                           'TS.NWAVEFORM' : ',',
                           'TS.DECFAC' : ',',
                           'TX.SN,NONE' : ',',
                           'TX.STN' : ',',
                           'TX.FREQ' : ',',
                           'TX.DUTY' : ',',
                           'TX.AMP' : ',',
                           'TX.SHUNT' : ','}
    
    def check_sampling_rate(self, zt_lst):
        """
        check to make sure the sampling rate is the same for all channels
        
        """
        
        nz = len(zt_lst)
        
        df_lst = np.zeros(nz)
        for ii, zt in enumerate(zt_lst):
            df_lst[ii] = zt.df
            
        tf_array = np.zeros((nz, nz))
        
        for jj in range(nz):
            tf_array[jj] = np.in1d(df_lst, [df_lst[jj]])
        
        false_test = np.where(tf_array==False)
        
        if len(false_test[0]) != 0:
            raise IOError('Sampling rates are not the same for all channels '+\
                          'Check file(s)'+zt_lst[false_test[0]])
        
    
    def check_time_series(self, zt_lst):
        """
        check to make sure timeseries line up with eachother.
        
        
        """
        
        n_fn = len(zt_lst)
        
        #test start time
        st_lst = np.array([int(zt.date_time[0][-2:]) for zt in zt_lst])
        time_max = np.where(st_lst==st_lst.max())[0]
        
        #get the number of seconds each time series is off by
        skip_dict = dict([(ii,0) for ii in range(n_fn)])
        if len(time_max) != n_fn:
            for ii in range(n_fn):
                skip_dict[ii] = st_lst.max()-st_lst[ii]
        
        #change data by amount needed        
        for ii, zt in enumerate(zt_lst):
            if skip_dict[ii] != 0:
                skip_points = skip_dict[ii]*zt.df
                print skip_points
                zt.time_series = zt.time_series[skip_points:]
                zt.gps_diff = zt.gps_diff[skip_dict[ii]:]
                zt.gps_lst = zt.gps_lst[skip_dict[ii]:]
                zt.date_time = zt.date_time[skip_dict[ii]:]
                zt.gps_time = zt.gps_time[skip_dict[ii:]]
            
        #test length of time series
        ts_len_lst = np.array([len(zt.time_series) for zt in zt_lst])
        
        #get the smallest number of points in the time series
        ts_min = ts_len_lst.min()
        
        #make a time series array for easy access
        ts_array = np.zeros((ts_min, n_fn))
        
        #trim the time series if needed
        for ii, zt in enumerate(zt_lst):
            if len(zt.time_series) != ts_min:
                ts_trim = zt.time_series[:ts_min]
            else:
                ts_trim = zt.time_series
            zt.time_series = ts_trim
            
            ts_array[:, ii] = ts_trim
            
            if self.verbose:
                print 'TS length for channel {0} '.format(zt.ch_number)+\
                      '({0}) '.format(zt.ch_cmp)+\
                      '= {0}'.format(len(ts_trim))
            self.log_lines.append(' '*4+\
                                  'TS length for channel {0} '.format(zt.ch_number)+\
                                  '({0}) '.format(zt.ch_cmp)+\
                                  '= {0}'.format(len(ts_trim)))
            self.log_lines.append(', T0 = {0}\n'.format(zt.date_time[0]))
        
        return ts_array, ts_min
    
    def write_cache_file(self, fn_lst, save_fn, station='ZEN'):
        """
        write a cache file from given filenames
        
        """
        
        n_fn = len(fn_lst)
        self.zt_lst = []
        for fn in fn_lst:
            zt1 = Zen3D(fn=fn)
            zt1.verbose = self.verbose
            try:
                zt1.read_3d()
            except IOError:
                zt1._seconds_diff = 59
                zt1.read_3d()
            self.zt_lst.append(zt1)
        
            #fill in meta data from the time series file
            self.meta_data['DATA.DATE0'] = ','+zt1.date_time[0].split(',')[0]
            self.meta_data['DATA.TIME0'] = ','+zt1.date_time[0].split(',')[1]
            self.meta_data['TS.ADFREQ'] = ',{0}'.format(int(zt1.df))
            self.meta_data['CH.FACTOR'] += ','+self._ch_factor 
            self.meta_data['CH.GAIN'] += ','+self._ch_gain
            self.meta_data['CH.CMP'] += ','+zt1.ch_cmp.upper()
            self.meta_data['CH.LENGTH'] += ','+zt1.ch_length
            self.meta_data['CH.EXTGAIN'] += ',1'
            self.meta_data['CH.NOTCH'] += ',NONE'
            self.meta_data['CH.HIGHPASS'] += ',NONE'
            self.meta_data['CH.LOWPASS'] += ','+\
                                       self._ch_lowpass_dict[str(int(zt1.df))]
            self.meta_data['CH.ADCARDSN'] += ','+zt1.ch_adcard_sn
            self.meta_data['CH.NUMBER'] += ',{0}'.format(zt1.ch_number)
            self.meta_data['RX.STN'] += ','+zt1.rx_stn
            
        #make sure all files have the same sampling rate
        self.check_sampling_rate(self.zt_lst)
        
        #make sure the length of time series is the same for all channels
        self.ts, ts_len = self.check_time_series(self.zt_lst)
        self.meta_data['TS.NPNT'] = ',{0}'.format(ts_len)
        
        #get the file name to save to 
        if save_fn[-4:] == '.cac':
            self.save_fn = save_fn
        elif save_fn[-4] == '.':
            raise NameError('File extension needs to be .cac, not'+save_fn[-4:])
        else:
            general_fn = station+'_'+\
                         self.meta_data['DATA.DATE0'][1:].replace('-','')+\
                         '_'+self.meta_data['DATA.TIME0'][1:].replace(':','')+\
                         '_'+self.meta_data['TS.ADFREQ'][1:]+'.cac'
            
            if os.path.basename(save_fn) != 'Merged':             
                save_fn = os.path.join(save_fn, 'Merged')
                if not os.path.exists(save_fn):
                    os.mkdir(save_fn)
            self.save_fn = os.path.join(save_fn, general_fn)
                
                
            
        cfid = file(self.save_fn, 'wb+')
        #--> write navigation records first        
        cfid.write(struct.pack('<i', self._nav_len))
        cfid.write(struct.pack('<i', self._flag))
        cfid.write(struct.pack('<h', self._type_dict['nav']))
        for nd in range(self._nav_len-2):
            cfid.write(struct.pack('<b', 0))
        cfid.write(struct.pack('<i', self._nav_len))
        
        #--> write meta data
        meta_str = ''.join([key+self.meta_data[key]+'\n' 
                             for key in np.sort(self.meta_data.keys())])
        
        meta_len = len(meta_str)
        
        cfid.write(struct.pack('<i', meta_len+2))
        cfid.write(struct.pack('<i', self._flag))
        cfid.write(struct.pack('<h', self._type_dict['meta']))
        cfid.write(meta_str)
        cfid.write(struct.pack('<i', meta_len+2))
        
        #--> write calibrations
        cal_data1 = 'HEADER.TYPE,Calibrate\nCAL.VER,019\nCAL.SYS,0000,'+\
                   ''.join([' 0.000000: '+'0.000000      0.000000,'*3]*27)
        cal_data2 = '\nCAL.SYS,0000,'+\
                    ''.join([' 0.000000: '+'0.000000      0.000000,'*3]*27)
                    
        cal_data = cal_data1+(cal_data2*(n_fn-1))
        cal_len = len(cal_data)
        
        cfid.write(struct.pack('<i', cal_len+2))
        cfid.write(struct.pack('<i', self._flag))
        cfid.write(struct.pack('<h', self._type_dict['cal']))
        cfid.write(cal_data[:-1]+'\n')
        cfid.write(struct.pack('<i', cal_len+2))
        
        #--> write data
        
        ts_block_len = int(ts_len)*n_fn*4+2
        
        cfid.write(struct.pack('<i', ts_block_len))
        cfid.write(struct.pack('<i', self._flag))
        cfid.write(struct.pack('<h', self._type_dict['ts']))
        for zz in range(ts_len):
            cfid.write(struct.pack('<'+'i'*n_fn, *self.ts[zz]))
        cfid.write(struct.pack('<i', ts_block_len))
         
        
        cfid.close()
        
        if self.verbose:
            print 'Saved File to: ', self.save_fn
        self.log_lines.append('='*72+'\n')
        self.log_lines.append('Saved File to: \n')
        self.log_lines.append(' '*4+'{0}\n'.format(self.save_fn))
        self.log_lines.append('='*72+'\n')
        
    
    def read_cache(self, cache_fn):
        """
        read a cache file
        
        """
        
        #open cache file to read in as a binary file
        cfid = file(cache_fn, 'rb')
        
        #read into a long string
        cdata = cfid.read()
        
        #--> read navigation data
        nav_block = np.fromstring(cdata[0:self._stamp_len], 
                                  dtype=self._data_type)
        
        #get starting and ending indices for navigation block
        ii = int(self._stamp_len)
        jj = self._stamp_len+nav_block['len']-2
        self.nav_data = np.fromstring(cdata[ii:jj], dtype=np.int8)
        
        #get indicies for length of block
        ii = int(jj)
        jj = ii+4
        nav_len_check = np.fromstring(cdata[ii:jj], np.int32)
        if nav_len_check != nav_block['len']:
            if self.verbose:
                print 'Index for second navigation length is {0}'.format(ii)
            raise IOError('Navigation length in data block are not equal: '+\
                          '{0} != {1}'.format(nav_block['len'], nav_len_check))
        
        #--> read meta data
        ii = int(jj)
        jj = ii+self._stamp_len
        
        meta_block = np.fromstring(cdata[ii:jj], dtype=self._data_type)
        ii = int(jj)
        jj = ii+meta_block['len']-2
        self.meta_data = {}
        meta_lst = cdata[ii:jj].split('\n')
        for mm in meta_lst:
            mfind = mm.find(',')
            self.meta_data[mm[0:mfind]] = mm[mfind+1:].split(',')
        
        #get index for second length test
        ii = int(jj)
        jj = ii+4
        meta_len_check = np.fromstring(cdata[ii:jj], dtype=np.int32)
        if meta_len_check != meta_block['len']:
            if self.verbose:
                print 'Index for second meta length is {0}'.format(ii)
            raise IOError('Meta length in data blocks are not equal: '+\
                          '{0} != {1}'.format(meta_block['len'], 
                                              meta_len_check))
        
        #--> read calibrations
        ii = int(jj)
        jj = ii+self._stamp_len
        cal_block = np.fromstring(cdata[ii:jj], dtype=self._data_type)
        
        ii = int(jj)
        jj = ii+cal_block['len']-2
        self.cal_data = cdata[ii:jj]
                
        
        ii = int(jj)
        jj = ii+4
        cal_len_check = np.fromstring(cdata[ii:jj], dtype=np.int32)
        if cal_len_check != cal_block['len']:
            if self.verbose:
                print 'Index for second cal length is {0}'.format(ii)
            raise IOError('Cal length in data blocks are not equal: '+\
                          '{0} != {1}'.format(cal_block['len'], 
                                              cal_len_check))
        
        #--> read data
        ii = int(jj)
        jj = ii+self._stamp_len
        
        ts_block = np.fromstring(cdata[ii:jj], dtype=self._data_type)
        
        #get time series data
        ii = int(jj)
        jj = ii+ts_block['len']-2
        self.ts = np.fromstring(cdata[ii:jj], dtype = np.int32)
        #resize time series to be length of each channel
        num_chn = len(self.meta_data['ch.cmp'.upper()])
        self.ts = self.ts.reshape(self.ts.shape[0]/num_chn, num_chn)
        
        ii = int(jj)
        jj = ii+4
        ts_len_check = np.fromstring(cdata[ii:jj], dtype=np.int32)
        if ts_len_check != ts_block['len']:
            if self.verbose:
                print 'Index for second ts length is {0}'.format(ii)
            raise IOError('ts length in data blocks are not equal: '+\
                          '{0} != {1}'.format(ts_block['len'], 
                                              ts_len_check))
    
#==============================================================================
# get the external drives for SD cards
#==============================================================================
def get_drives():
    """
    get a list of logical drives detected on the machine
    
    """
    drives = []
    bitmask = win32api.GetLogicalDrives()
    for letter in string.uppercase:
        if bitmask & 1:
            drives.append(letter)
        bitmask >>= 1

    return drives
   
#==============================================================================
# get the names of the drives which should correspond to channels
#==============================================================================
def get_drive_names():
    """
    get a list of drive names detected assuming the cards are names by box 
    and channel.
    
    """
    
    drives = get_drives()
    
    drive_dict = {}
    for drive in drives:
        try:
            drive_name = win32api.GetVolumeInformation(drive+':\\')[0]
            if drive_name.find('CH') > 0:
                drive_dict[drive] = drive_name
        except:
            pass
    
    if drives == {}:
        print 'No external drives detected, check the connections.'
        return None
    else:
        return drive_dict

#==============================================================================
# copy files from SD cards   
#==============================================================================
def copy_from_sd(station, savepath=r"d:\Peacock\MTData", 
                 channel_dict={'1':'HX', '2':'HY', '3':'HZ',
                                   '4':'EX', '5':'EY'},
                 copy_date=None):
    """
    copy files from sd cards into a common folder
    
    do not put an underscore in station, causes problems at the moment
    
    """
    
    drive_names = get_drive_names()
    if drive_names is None:
        raise IOError('No drives to copy from.')
    save_path = os.path.join(savepath,station)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    log_fid = file(os.path.join(save_path,'Log_file.log'),'w')
    
    if copy_date is not None:
        if type(copy_date) is not list or type(copy_date) is not np.ndarray:
            copy_date = [copy_date]
    
    fn_lst = []
    for key in drive_names.keys():
        dr = r"{0}:\\".format(key)
        print '='*25+drive_names[key]+'='*25
        log_fid.write('='*25+drive_names[key]+'='*25+'\n')
        for fn in os.listdir(dr):
            full_path_fn = os.path.normpath(os.path.join(dr, fn))
            if fn[-4:] == '.cfg':
                shutil.copy(full_path_fn, os.path.join(save_path, fn))
                    
            try:
                file_size = os.stat(full_path_fn)[6]
                if file_size >= 1600L and fn.find('zenini') != 0:
                    zt = Zen3D(fn=full_path_fn)
                    try:
                        zt.read_3d()
                        log_fid.writelines(zt.log_lines)
                    except IOError:
                        try:
                            zt._seconds_diff = 59
                            zt.read_3d()
                            log_fid.writelines(zt.log_lines)
                        except IOError:
                            print full_path_fn+' is more than 1 minute off'+\
                                  'start time, Did not copy, check the file'
                            log_fid.writelines(' '*4+full_path_fn+
                                               ' is more than 1 minute off'+\
                                               'start time, Did not copy \n')
                            break
                    
                    if zt.start_time is not None:
                        fn_find = True
                        if copy_date is not None:
                            fn_find = False
                            for cdate in copy_date:
                                if zt.start_time.find(cdate) == 0:
                                    fn_find = True
                                    break
                                
                        if fn_find:
                            
                            channel = channel_dict[drive_names[key][-1]]
                            st = zt.start_time.split(',')[1].replace(':','')
                            sd = zt.start_time.split(',')[0].replace('-','')
                            sv_fn = '{0}_{1}_{2}_{3}_{4}.Z3D'.format(station, 
                                                                     sd, 
                                                                     st,
                                                                     int(zt.df),
                                                                     channel)
                                                                 
                            full_path_sv = os.path.join(save_path, sv_fn)
                            fn_lst.append(full_path_sv)
                            
                            shutil.copy(full_path_fn, full_path_sv)
                            print 'copied {0} to {1}\n'.format(full_path_fn, 
                                                             full_path_sv)
                                                             
                            log_fid.write('copied {0} to \n'.format(full_path_fn)+\
                                          '       {0}\n'.format(full_path_sv))
                        else:
                            print '+++ SKIPPED {0}+++\n'.format(zt.fn)
                            log_fid.write(' '*4+\
                                          '+++ SKIPPED {0}+++\n'.format(zt.fn))
                        
                    else:
                        print '{0} '.format(full_path_fn)+\
                               'not copied due to bad data.'
                               
                        log_fid.write(' '*4+'***{0} '.format(full_path_fn)+\
                                      'not copied due to bad data.\n\n')
            except WindowsError:
                print 'Faulty file at {0}'.format(full_path_fn)
                log_fid.write('---Faulty file at {0}\n\n'.format(full_path_fn))
    log_fid.close()
    
    return fn_lst
 
#==============================================================================
# merge files into cache files for each sample block   
#==============================================================================
def merge_3d_files(fn_lst, savepath=None):
    """
    merge the component .Z3D files into cache files.
    
    """
    
    start_time = time.ctime()
    merge_lst = np.array([[fn]+\
                          os.path.basename(fn)[:-4].split('_')
                          for fn in fn_lst if fn[-4:]=='.Z3D'])
                              
    merge_lst = np.array([merge_lst[:,0], 
                          merge_lst[:,1],  
                          np.core.defchararray.add(merge_lst[:,2],
                                                   merge_lst[:,3]),
                          merge_lst[:,4],
                          merge_lst[:,5]])
    merge_lst = merge_lst.T
                              
    time_counts = Counter(merge_lst[:,2])
    time_lst = time_counts.keys()
    
    log_lines = []
  
    merged_fn_lst = []
    for tt in time_lst:
        log_lines.append('+'*72+'\n')
        log_lines.append('Files Being Merged: \n')
        cache_fn_lst = merge_lst[np.where(merge_lst==tt)[0],0].tolist()
        
        for cfn in cache_fn_lst:
            log_lines.append(' '*4+cfn+'\n')
        if savepath is None:
            save_path = os.path.dirname(cache_fn_lst[0])
            station_name = merge_lst[np.where(merge_lst==tt)[0][0],1]
        else:
            station_name = 'ZEN'
            
        zc = ZenCache()
        zc.verbose = False
        zc.write_cache_file(cache_fn_lst, save_path, station=station_name)
        for zt in zc.zt_lst:
            log_lines.append(zt.log_lines)
        merged_fn_lst.append(zc.save_fn)
        log_lines.append('\n---> Merged Time Series Lengths and Start Time \n')
        log_lines.append(zc.log_lines)
        log_lines.append('\n')
    
    end_time = time.ctime()
    
    print 'Start time: {0}'.format(start_time)
    print 'End time:   {0}'.format(end_time)
    
    if os.path.basename(save_path) != 'Merged':
        log_fid = file(os.path.join(save_path, 'Merged', 
                                    station_name+'_Merged.log'), 'w')
    else:
        log_fid = file(os.path.join(save_path, station_name+'_Merged.log'),
                       'w')
    for line in log_lines:
        log_fid.writelines(line)
    log_fid.close()
        
    return merged_fn_lst
    
#==============================================================================
# delete files from sd cards    
#==============================================================================
def delete_files_from_sd(delete_date=None, delete_type=None, 
                         delete_folder=r"d:\Peacock\MTData\Deleted",
                         verbose=True):
    """
    delete files from sd card, if delete_date is not None, anything on this 
    date and before will be deleted.  Deletes just .Z3D files, leaves 
    zenini.cfg
    
    Agruments:
    -----------
        **delete_date** : YYYY-MM-DD
                         date to delete files from 
                         
        **delete_type** : [ 'all' | 'before' | 'after' | 'on' ]
                          * 'all' --> delete all files on sd card
                          * 'before' --> delete files on and before delete_date
                          * 'after' --> delete files on and after delete_date
                          * 'on' --> delete files on delete_date
                          
        **delete_folder** : string
                            full path to a folder where files will be moved to
                            just in case.  If None, files will be deleted 
                            for ever.
                            
    Returns:
    ---------
        **delete_fn_lst** : list
                            list of deleted files.
    
    """
    
    drive_names = get_drive_names()
    if drive_names is None:
        raise IOError('No drives to copy from.')

    log_lines = []
    if delete_folder is not None:
        if not os.path.exists(delete_folder):
            os.mkdir(delete_folder)
        log_fid = file(os.path.join(delete_folder,'Log_file.log'),'w')
    
    if delete_date is not None:
        delete_date = int(delete_date.replace('-',''))
    
    delete_fn_lst = []
    for key in drive_names.keys():
        dr = r"{0}:\\".format(key)
        log_lines.append('='*25+drive_names[key]+'='*25+'\n')
        for fn in os.listdir(dr):
            if fn[-4:].lower() == '.Z3D'.lower():
                full_path_fn = os.path.normpath(os.path.join(dr, fn))
                if delete_type == 'all' or delete_date is None:
                    if delete_folder is None:
                        os.remove(full_path_fn)
                        delete_fn_lst.append(full_path_fn)
                        log_lines.append('Deleted {0}'.format(full_path_fn))
                    else:
                        shutil.move(full_path_fn, 
                                    os.path.join(delete_folder,
                                    os.path.basename(full_path_fn)))
                        delete_fn_lst.append(full_path_fn)
                        log_lines.append('Moved {0} '.format(full_path_fn)+
                                         'to {0}'.format(delete_folder))
                else:
                    zt_date = int(time.strftime('%Y%m%d',
                                            time.localtime(
                                            os.stat(full_path_fn)[-1])))
                   
                    if delete_type == 'before':
                        if zt_date <= delete_date:
                            if delete_folder is None:
                                os.remove(full_path_fn)
                                delete_fn_lst.append(full_path_fn)
                                log_lines.append('Deleted {0}\n'.format(full_path_fn))
                            else:
                                shutil.move(full_path_fn, 
                                            os.path.join(delete_folder,
                                            os.path.basename(full_path_fn)))
                                delete_fn_lst.append(full_path_fn)
                                log_lines.append('Moved {0} '.format(full_path_fn)+
                                                 'to {0}\n'.format(delete_folder))
                    elif delete_type == 'after':
                        if zt_date >= delete_date:
                            if delete_folder is None:
                                os.remove(full_path_fn)
                                delete_fn_lst.append(full_path_fn)
                                log_lines.append('Deleted {0}\n'.format(full_path_fn))
                            else:
                                shutil.move(full_path_fn, 
                                            os.path.join(delete_folder,
                                            os.path.basename(full_path_fn)))
                                delete_fn_lst.append(full_path_fn)
                                log_lines.append('Moved {0} '.format(full_path_fn)+
                                                 'to {0}\n'.format(delete_folder))
                    elif delete_type == 'on':
                        if zt_date == delete_date:
                            if delete_folder is None:
                                os.remove(full_path_fn)
                                delete_fn_lst.append(full_path_fn)
                                log_lines.append('Deleted {0}\n'.format(full_path_fn))
                            else:
                                shutil.move(full_path_fn, 
                                            os.path.join(delete_folder,
                                            os.path.basename(full_path_fn)))
                                delete_fn_lst.append(full_path_fn)
                                log_lines.append('Moved {0} '.format(full_path_fn)+
                                                 'to {0}\n'.format(delete_folder))
    if delete_folder is not None:
        log_fid = file(os.path.join(delete_folder, 'Delete_log.log'), 'w')
        log_fid.writelines(log_lines)
        log_fid.close()
    if verbose:
        for lline in log_lines:
            print lline
    
    return delete_fn_lst
    
#==============================================================================
# run mtft24.exe from a command window
#==============================================================================
def run_mtft24(dirpath):
    """
    opens mtft24.exe from the command line
    """
    
    curdir = os.getcwd()
    os.chdir(dirpath)
    p = subprocess.call('mtft24.exe')
    os.chdir(curdir)
    
    return p
    


#==============================================================================
# read and write a zen schedule 
#==============================================================================
class ZenSchedule(object):
    """
    deals with reading, writing and copying schedule
    
    """
    
    def __init__(self):
        
        self.verbose = True
        self.sr_dict = {'256':'0', '512':'1', '1024':'2', '2048':'3', 
                        '4096':'4'}
        self.gain_dict = dict([(mm, 2**mm) for mm in range(7)])
        self.sa_keys = ['date', 'time', 'resync_yn', 'log_yn', 'tx_duty', 
                        'tx_period', 'sr', 'gain', 'nf_yn']
        self.sa_lst = []
        self.ch_cmp_dict = {'1':'hx', '2':'hy', '3':'hz', '4':'ex', '5':'ey',
                            '6':'hz'}
        self.ch_num_dict = dict([(self.ch_cmp_dict[key], key) 
                                 for key in self.ch_cmp_dict])
        
        self.meta_keys = ['TX.ID', 'RX.STN', 'Ch.Cmp', 'Ch.Number', 
                          'Ch.varAsp']
        self.meta_dict = {'TX.ID':'none', 'RX.STN':'01', 'Ch.Cmp':'HX',
                          'Ch.Number':'1', 'Ch.varAsp':50}
        self.light_dict = {'YellowLight':0,
                          'BlueLight':1,
                          'RedLight':0,
                          'GreenLight':1}
        self.offset = time.strftime("%Y-%m-%d,%H:%M:%S",time.gmtime())

    def read_schedule(self, fn):
        """
        read zen schedule file
        
        """
        
        sfid = file(fn, 'r')
        lines = sfid.readlines()
        
        for line in lines:
            if line.find('scheduleaction') == 0:
                line_lst = line.strip().split(' ')[1].split(',')
                sa_dict = {}
                for ii, key in enumerate(self.sa_keys):
                    sa_dict[key] = line_lst[ii]
                self.sa_lst.append(sa_dict)
                
            elif line.find('metadata'.upper()) == 0:
                line_lst = line.strip().split(' ')[1].split('|')
                for md in line_lst[:-1]:
                    md_lst = md.strip().split(',')
                    self.meta_dict[md_lst[0]] = md_lst[1]
                    
            elif line.find('offset') == 0:
                line_str = line.strip().split(' ')
                self.offset = line_str[1]
                
            elif line.find('Light') > 0:
                line_lst = line.strip().split(' ')
                try:
                    self.light_dict[line_lst[0]]
                    self.light_dict[line_lst[0]] = line_lst[1]
                except KeyError:
                    pass
                
    def write_schedule(self, station, clear_schedule=True, 
                       clear_metadata=True, varaspace=100, 
                       savename=0):
        """
        write a zen schedule file
        
        **Note**: for the older boxes use 'Zeus3Ini.cfg' for the savename
        
        """
        
        drive_names = get_drive_names()
        self.meta_dict['RX.STN'] = station
        self.meta_dict['Ch.varAsp'] = '{0}'.format(varaspace)
        
        if savename == 0:
            save_name = 'zenini.cfg'
        elif savename == 1:
            save_name = 'Zeus3Ini.cfg'
        else:
            save_name = savename
         
        for dd in drive_names.keys():
            dname = drive_names[dd]
            sfid = file(os.path.normpath(os.path.join(dd+':\\', save_name)),
                        'w')
            if clear_schedule:
                sfid.write('clearschedule\n')
            if clear_metadata:
                sfid.write('metadata clear\n')
            for sa_dict in self.sa_lst:
                sa_line = ''.join([sa_dict[key]+',' for key in self.sa_keys])
                sfid.write('scheduleaction '+sa_line[:-1]+'\n')
            sfid.write('offsetschedule {0}\n'.format(self.offset))
            
            self.meta_dict['Ch.Cmp'] = self.ch_cmp_dict[dname[-1]]
            self.meta_dict['Ch.Number'] = dname[-1]
            meta_line = ''.join(['{0},{1}|'.format(key,self.meta_dict[key]) 
                                 for key in self.meta_keys])
            sfid.write('METADATA '+meta_line+'\n')
            for lkey in self.light_dict.keys():
                sfid.write('{0} {1}\n'.format(lkey, self.light_dict[lkey]))
            sfid.close()
            print 'Wrote {0}:\{1} to {2} as {3}'.format(dd, save_name, dname,
                                                   self.ch_cmp_dict[dname[-1]])
            
            
def copy_and_merge(station, z3d_savepath=None, merge_savepath=None, 
                   delete_dict=None, run_mtft24_yn='y', 
                   channel_dict={'1':'HX', '2':'HY', '3':'HZ','4':'EX', 
                                 '5':'EY'},
                   copy_date=None):
    """
    copy files from sd card then merge them together and run mtft24.exe
    
    """
    
    #--> copy files from sd cards
    cpkwargs = {}
    cpkwargs['channel_dict'] = channel_dict
    cpkwargs['copy_date'] = copy_date
    if z3d_savepath != None:
        cpkwargs['savepath'] = z3d_savepath
    
    fn_lst = copy_from_sd(station, **cpkwargs)
    
    #--> merge files into cache files
    mfn_lst = merge_3d_files(fn_lst, savepath=merge_savepath)
    
    #--> open an mtft24.exe instance as a command option
    if run_mtft24_yn == 'y':
        if len(mfn_lst) > 0:
            run_mtft24(os.path.dirname(mfn_lst[0]))
        else:
            print 'No files to merge, check log file'
    
    
    
               
                
        
        
    
                     
    

    
    
    

    
    
    