# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 15:19:30 2011

@author: a1185872
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator

def read_output_file(output_fn):
    """
    Reads in an output file from winglink and returns the data
    in the form of a dictionary of structured arrays.
    
    Arguments:
    -----------
        **output_fn** : string
                        the full path to winglink outputfile
        
    Returns:
    ----------
        **wl_data** : dictionary with keys of station names
                      each station contains a structured array with keys
                      * 'station' --> station name
                      * 'period' --> periods to plot
                      * 'te_res' --> TE resistivity in linear scale
                      * 'tm_res' --> TM resistivity in linear scale
                      * 'te_phase' --> TE phase in deg
                      * 'tm_phase' --> TM phase in deg
                      * 're_tip' --> real tipper amplitude.
                      * 'im_tip' --> imaginary tipper amplitude
                      * 'rms' --> RMS for the station
                    
        .. note:: each data is an np.ndarray(2, num_periods) where the first
                  index is the data and the second index is the model response
    """
    
    ofid = open(output_fn, 'r')
    lines = ofid.readlines()
    
    idict = {}
    stationlst = []
    
    #get title line
    titleline=lines[1].replace('"','')
    titleline=titleline.rstrip().split(',')
    title=titleline[1].split(':')[1]
    profile = titleline[0].split(':')[1]
    inversiontype = lines[2].rstrip()
    
    dkeys = ['obs_tm_res', 'obs_tm_phase', 'mod_tm_res', 'mod_tm_phase', 
             'obs_te_res', 'obs_te_phase', 'mod_te_res', 'mod_te_phase', 
             'obs_re_tip', 'obs_im_tip', 'mod_re_tip', 'mod_im_tip', 'period']
    
    for line in lines[3:]:
        # get the beginning of the station block
        if line.find('Data for station') == 0:
            station = line.rstrip().split(':')[1][1:]
            idict[station] = {}
            stationlst.append(station)
            print 'Reading in station: ', station
            for key in dkeys:
                idict[station][key] = []
        # get rms
        elif line.find('RMS') == 0:
            idict[station]['rms'] = float(line.strip().split(' = ')[1])
        # skip the divding line
        elif line.find('==') == 0:
            pass
        #get the data
        else:
            linelst = line.split()
            if len(linelst) == len(dkeys):
                for kk, key in enumerate(dkeys):
                    try:
                        if key.find('phase') >= 0:
                            idict[station][key].append(-1*float(linelst[kk]))
                        else:
                            idict[station][key].append(float(linelst[kk]))
                    except ValueError:
                        idict[station][key].append(0)
            else:
                pass
    
    #get data into a more useful format that takes into account any masking of 
    #data points.
    
    data = {}
    for st in idict.keys():
        data[st] = {}
        data[st]['station'] = st
        data[st]['period'] = np.array(idict[st]['period'])
        data[st]['te_res'] = np.array([np.array(idict[st]['obs_te_res']),
                                       np.array(idict[st]['mod_te_res'])])
        data[st]['tm_res'] = np.array([np.array(idict[st]['obs_tm_res']),
                                       np.array(idict[st]['mod_tm_res'])])
        data[st]['te_phase'] = np.array([np.array(idict[st]['obs_te_phase']),
                                         np.array(idict[st]['mod_te_phase'])])
        data[st]['tm_phase'] = np.array([np.array(idict[st]['obs_tm_phase']),
                                         np.array(idict[st]['mod_tm_phase'])])
        data[st]['re_tip'] = np.array([np.array(idict[st]['obs_re_tip']),
                                       np.array(idict[st]['mod_re_tip'])])
        data[st]['im_tip'] = np.array([np.array(idict[st]['obs_im_tip']),
                                       np.array(idict[st]['mod_im_tip'])])
        data[st]['rms'] = float(idict[st]['rms'])
     
    return data
    
def plotResponses(outputfile,maxcol=8,plottype='all',**kwargs):
    """
    plotResponse will plot the responses modeled from winglink against the 
    observed data.
    
    Inputs:
        outputfile = full path and filename to output file
        maxcol = maximum number of columns for the plot
        plottype = 'all' to plot all on the same plot
                   '1' to plot each respones in a different figure
                   station to plot a single station or enter as a list of 
                   stations to plot a few stations [station1,station2].  Does
                   not have to be verbatim but should have similar unique 
                   characters input pb01 for pb01cs in outputfile
    Outputs:
        None
    """
    
    idict,rplst,plst,stationlst,titlelst=readOutputFile(outputfile)
    nstations=len(idict)
    
    #plot all responses onto one plot
    if plottype=='all':
        maxcol=8         
        nrows=int(np.ceil(nstations/float(maxcol)))
        
        fig=plt.figure(1,[14,10])
        gs=gridspec.GridSpec(nrows,1,wspace=.15,left=.03)
        count=0
        for rr in range(nrows):
            g1=gridspec.GridSpecFromSubplotSpec(6,maxcol,subplot_spec=gs[rr],
                                                    hspace=.15,wspace=.05)
            count=rr*(maxcol)
            for cc in range(maxcol):
                try:
                    station=stationlst[count+cc]
                except IndexError:
                    break
                #plot resistivity
                axr=plt.Subplot(fig,g1[:4,cc])
                fig.add_subplot(axr)
                axr.loglog(idict[station]['period'],idict[station]['obsresxy'],
                           's',ms=2,color='b',mfc='b')
                axr.loglog(idict[station]['period'],idict[station]['modresxy'],
                           '*', ms=5,color='r',mfc='r')
                axr.loglog(idict[station]['period'],idict[station]['obsresyx'],
                           'o',ms=2,color='c',mfc='c')
                axr.loglog(idict[station]['period'],idict[station]['modresyx'],
                           'x',ms=5,color='m',mfc='m')
                axr.set_title(station+'; rms= %.2f' % idict[station]['rms'],
                              fontdict={'size':12,'weight':'bold'})
                axr.grid(True)
                axr.set_xticklabels(['' for ii in range(10)])
                if cc>0:
                    axr.set_yticklabels(['' for ii in range(6)])
                    
                
                #plot phase
                axp=plt.Subplot(fig,g1[-2:,cc])
                fig.add_subplot(axp)
                axp.semilogx(idict[station]['period'],
                             np.array(idict[station]['obsphasexy']),
                             's',ms=2,color='b',mfc='b')
                axp.semilogx(idict[station]['period'],
                             np.array(idict[station]['modphasexy']),
                             '*',ms=5,color='r',mfc='r')
                axp.semilogx(idict[station]['period'],
                             np.array(idict[station]['obsphaseyx']),
                             'o',ms=2,color='c',mfc='c')
                axp.semilogx(idict[station]['period'],
                             np.array(idict[station]['modphaseyx']),
                             'x',ms=5,color='m',mfc='m')
                axp.set_ylim(0,90)
                axp.grid(True)
                axp.yaxis.set_major_locator(MultipleLocator(30))
                axp.yaxis.set_minor_locator(MultipleLocator(5))
        
                if cc==0 and rr==0:
                    axr.legend(['$Obs_{xy}$','$Mod_{xy}$','$Obs_{yx}$',
                                '$Mod_{yx}$'],
                                loc=2,markerscale=1,borderaxespad=.05,
                                labelspacing=.08,
                                handletextpad=.15,borderpad=.05)
                if cc==0:
                    axr.set_ylabel('App. Res. ($\Omega \cdot m$)',
                                   fontdict={'size':12,'weight':'bold'})
                    axp.set_ylabel('Phase (deg)',
                                   fontdict={'size':12,'weight':'bold'})
                    axr.yaxis.set_label_coords(-.08,.5)
                    axp.yaxis.set_label_coords(-.08,.5)
        
                if cc>0:
                    axr.set_yticklabels(['' for ii in range(6)])
                    axp.set_yticklabels(['' for ii in range(6)])
                if rr==nrows-1:
                    axp.set_xlabel('Period (s)',
                                   fontdict={'size':12,'weight':'bold'})
                                   
    #plot each respones in a different figure
    elif plottype=='1':
        gs=gridspec.GridSpec(6,2,wspace=.05)
        for ii,station in enumerate(stationlst):
            fig=plt.figure(ii+1,[7,8])
            
            #plot resistivity
            axr=fig.add_subplot(gs[:4,:])
            
            axr.loglog(idict[station]['period'],idict[station]['obsresxy'],
                       's',ms=2,color='b',mfc='b')
            axr.loglog(idict[station]['period'],idict[station]['modresxy'],
                       '*', ms=5,color='r',mfc='r')
            axr.loglog(idict[station]['period'],idict[station]['obsresyx'],
                       'o',ms=2,color='c',mfc='c')
            axr.loglog(idict[station]['period'],idict[station]['modresyx'],
                       'x',ms=5,color='m',mfc='m')
            axr.set_title(station+'; rms= %.2f' % idict[station]['rms'],
                          fontdict={'size':12,'weight':'bold'})
            axr.grid(True)
            axr.set_xticklabels(['' for ii in range(10)])
                            
            #plot phase
            axp=fig.add_subplot(gs[-2:,:])
            axp.semilogx(idict[station]['period'],
                         np.array(idict[station]['obsphasexy']),
                         's',ms=2,color='b',mfc='b')
            axp.semilogx(idict[station]['period'],
                         np.array(idict[station]['modphasexy']),
                         '*',ms=5,color='r',mfc='r')
            axp.semilogx(idict[station]['period'],
                         np.array(idict[station]['obsphaseyx']),
                         'o',ms=2,color='c',mfc='c')
            axp.semilogx(idict[station]['period'],
                         np.array(idict[station]['modphaseyx']),
                         'x',ms=5,color='m',mfc='m')
            axp.set_ylim(0,90)
            axp.grid(True)
            axp.yaxis.set_major_locator(MultipleLocator(10))
            axp.yaxis.set_minor_locator(MultipleLocator(1))
            
            axr.set_ylabel('App. Res. ($\Omega \cdot m$)',
                           fontdict={'size':12,'weight':'bold'})
            axp.set_ylabel('Phase (deg)',
                           fontdict={'size':12,'weight':'bold'})
            axp.set_xlabel('Period (s)',fontdict={'size':12,'weight':'bold'})
            axr.legend(['$Obs_{xy}$','$Mod_{xy}$','$Obs_{yx}$',
                                '$Mod_{yx}$'],
                                loc=2,markerscale=1,borderaxespad=.05,
                                labelspacing=.08,
                                handletextpad=.15,borderpad=.05)
            axr.yaxis.set_label_coords(-.05,.5)
            axp.yaxis.set_label_coords(-.05,.5)
    
    else:
        pstationlst=[]
        if type(plottype) is not list:
            plottype=[plottype]
        for station in stationlst:
            for pstation in plottype:
                if station.find(pstation)>=0:
                    print 'plotting ',station
                    pstationlst.append(station)

        gs=gridspec.GridSpec(6,2,wspace=.05,left=.1)
        for ii,station in enumerate(pstationlst):
            fig=plt.figure(ii+1,[7,7])
            
            #plot resistivity
            axr=fig.add_subplot(gs[:4,:])
            
            axr.loglog(idict[station]['period'],idict[station]['obsresxy'],
                       's',ms=2,color='b',mfc='b')
            axr.loglog(idict[station]['period'],idict[station]['modresxy'],
                       '*', ms=5,color='r',mfc='r')
            axr.loglog(idict[station]['period'],idict[station]['obsresyx'],
                       'o',ms=2,color='c',mfc='c')
            axr.loglog(idict[station]['period'],idict[station]['modresyx'],
                       'x',ms=5,color='m',mfc='m')
            axr.set_title(station+'; rms= %.2f' % idict[station]['rms'],
                          fontdict={'size':12,'weight':'bold'})
            axr.grid(True)
            axr.set_xticklabels(['' for ii in range(10)])
                            
            #plot phase
            axp=fig.add_subplot(gs[-2:,:])
            axp.semilogx(idict[station]['period'],
                         np.array(idict[station]['obsphasexy']),
                         's',ms=2,color='b',mfc='b')
            axp.semilogx(idict[station]['period'],
                         np.array(idict[station]['modphasexy']),
                         '*',ms=5,color='r',mfc='r')
            axp.semilogx(idict[station]['period'],
                         np.array(idict[station]['obsphaseyx']),
                         'o',ms=2,color='c',mfc='c')
            axp.semilogx(idict[station]['period'],
                         np.array(idict[station]['modphaseyx']),
                         'x',ms=5,color='m',mfc='m')
            axp.set_ylim(0,90)
            axp.grid(True)
            axp.yaxis.set_major_locator(MultipleLocator(10))
            axp.yaxis.set_minor_locator(MultipleLocator(1))
            
            axr.set_ylabel('App. Res. ($\Omega \cdot m$)',
                           fontdict={'size':12,'weight':'bold'})
            axp.set_ylabel('Phase (deg)',
                           fontdict={'size':12,'weight':'bold'})
            axp.set_xlabel('Period (s)',fontdict={'size':12,'weight':'bold'})
            axr.legend(['$Obs_{xy}$','$Mod_{xy}$','$Obs_{yx}$',
                                '$Mod_{yx}$'],
                                loc=2,markerscale=1,borderaxespad=.05,
                                labelspacing=.08,
                                handletextpad=.15,borderpad=.05)
            axr.yaxis.set_label_coords(-.05,.5)
            axp.yaxis.set_label_coords(-.05,.5)
            
def readModelFile(modelfile,profiledirection='ew'):
    """
    readModelFile reads in the XYZ txt file output by Winglink.    
    
    Inputs:
        modelfile = fullpath and filename to modelfile
        profiledirection = 'ew' for east-west predominantly, 'ns' for 
                            predominantly north-south.  This gives column to 
                            fix
    """
    
    mfid=open(modelfile,'r')
    lines=mfid.readlines()    
    nlines=len(lines)    
    
    X=np.zeros(nlines)
    Y=np.zeros(nlines)
    Z=np.zeros(nlines)
    rho=np.zeros(nlines)
    clst=[]
    #file starts from the bottom of the model grid in X Y Z Rho coordinates
    if profiledirection=='ew':        
        for ii,line in enumerate(lines):
            linestr=line.split()
            X[ii]=float(linestr[0])
            Y[ii]=float(linestr[1])
            Z[ii]=float(linestr[2])
            rho[ii]=float(linestr[3])
            if ii>0:
                if X[ii]<X[ii-1]:
                    clst.append(ii)

    clst=np.array(clst)
    cspot=np.where(np.remainder(clst,clst[0])!=0)[0]


    return X,Y,Z,rho,clst
            
    
        
    