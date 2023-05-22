#================================PlotFuncs.py==================================#
# Created by Ciaran O'Hare 2020

# Description:
# This file has many functions which are used throughout the project, but are
# all focused around the bullshit that goes into making the plots

#==============================================================================#

import sys
from numpy import *
from numpy.random import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib import colors
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
from scipy.stats import norm
import pandas as pd
import webplotdigitizer as wpd
import d_factor

pltdir = 'plots/'
pltdir_png = pltdir+'plots_png/'

#==============================================================================#
class SterileNeutrinoMixing():
    
    def FigSetup(xlab=r'$m_s$ [keV]',ylab=r'$\sin^2(2\theta)$',\
                     sin_min = 1.0e-13,sin_max = 1.0e-9,\
                     m_min = 1,m_max = 50,\
                     lw=2.5,lfs=45,tfs=32,tickdir='in',alpha=0.8,\
                     Grid=False,Shape='Rectangular',mathpazo=False,TopAndRightTicks=False):

#             plt.rcParams['axes.linewidth'] = lw
#             plt.rc('text', usetex=True)
#             plt.rc('font', family='serif',size=tfs)

#             if mathpazo:
#                 mpl.rcParams['text.latex.preamble'] = [r'\usepackage{mathpazo}']

            if Shape=='Wide':
                fig = plt.figure(figsize=(16.5,5))
            elif Shape=='Rectangular':
                fig = plt.figure(figsize=(16.5,11))

            ax = fig.add_subplot(111)

            ax.set_xlabel(xlab,fontsize=lfs)
            ax.set_ylabel(ylab,fontsize=lfs)
            
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_xlim([m_min,m_max])
            ax.set_ylim([sin_min,sin_max])

            ax.tick_params(which='major',direction=tickdir,width=2.5,length=13,right=TopAndRightTicks,top=TopAndRightTicks,pad=8,labelsize=tfs)
            ax.tick_params(which='minor',direction=tickdir,width=1,length=10,right=TopAndRightTicks,top=TopAndRightTicks,pad=8,labelsize=tfs)

            locmaj = mpl.ticker.LogLocator(base=10.0, subs=(1.0, 3.0), numticks=50)
            locmin = mpl.ticker.LogLocator(base=10.0, subs=arange(2, 10)*.1,numticks=100)
            ax.xaxis.set_major_locator(locmaj)
            ax.xaxis.set_minor_locator(locmin)
            ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.0f'))
            ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

            locmaj = mpl.ticker.LogLocator(base=10.0, subs=(1.0, ), numticks=100)
            locmin = mpl.ticker.LogLocator(base=10.0, subs=arange(2, 10)*.1,numticks=100)
            ax.yaxis.set_major_locator(locmaj)
            ax.yaxis.set_minor_locator(locmin)
            ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

            if Grid:
                ax.grid(zorder=0)
                            
            return fig,ax


    def DessertScience2020(ax,ec='gray',fc='lightgray',z=0,label=False,fs=15,alpha=0.5,data_dir='../data/'):
        # arXiv[1812.06976]
        y2 = ax.get_ylim()[1]
        DessertLimit = pd.read_csv(data_dir+'DessertScienceLimit.txt',delimiter=' ',header=None)
        m_arr = DessertLimit.iloc[0]
        sin_arr = DessertLimit.iloc[1]
        plt.fill_between(m_arr,sin_arr,y2=y2,edgecolor=ec,facecolor=fc,zorder=z,alpha=alpha)
        if label==True:
            plt.text(7,1e-12,r'XMM MW Halo',fontsize=fs,color='k',rotation=90,ha='left',va='top')

        return
    
    def FosterGP2021(ax,ec='gray',fc='lightgray',z=1,label=False,fs=15,alpha=0.5,data_dir='../data/'):
        # arXiv[2102.02207]
        y2 = ax.get_ylim()[1]
        FosterLimit = np.load(data_dir+'LimitsFoster2021_MWHalo.npy')
        m_arr = FosterLimit[0]
        sin_arr = FosterLimit[1]
        plt.fill_between(m_arr,sin_arr,y2=y2,edgecolor=ec,facecolor=fc,zorder=z,alpha=alpha)
        if label==True:
            plt.text(7,1e-12,r'XMM MW Halo',fontsize=fs,color='k',rotation=90,ha='left',va='top')

        return


    def PerezBulge2019(ax,ec='gray',fc='lightgray',z=5,label=False,fs=17,alpha=0.5,Rescale=True,data_dir='../data/'):
        # 1908.09037

        y2 = ax.get_ylim()[1]
        if Rescale:
            Dfac_at_Bulge_Perez = d_factor.D_factor_alt(rs=20,rho_local=0.4,r_GC = 8).return_D_factor(0,10)
            Dfac_at_Bulge_fiducial = d_factor.D_factor_alt(rs=20,rho_local=0.4,r_GC = 8.127).return_D_factor(0,10)
            RescalingFactor = Dfac_at_Bulge_fiducial/Dfac_at_Bulge_Perez
        else:
            RescalingFactor = 1
        
        m_arr,sin_arr = wpd.read_csv(data_dir+'Perez2019Bulge.csv')
        plt.fill_between(m_arr,sin_arr*RescalingFactor,y2=y2,edgecolor=ec,facecolor=fc,zorder=z,alpha=alpha)
        if label==True:
            plt.text(16,1e-12,r'NuSTAR',fontsize=fs,color='w',ha='left')
        
        return

    def BulbulHalo2020(ax,ec='gray',fc='lightgray',z=4,label=False,fs=15,alpha=0.5,Rescale=True,data_dir='../data/'):
        # 2008.02283
        
        if Rescale:
            Dfac_in_Halo_Bulbul = d_factor.D_factor_alt(rs=16,rho_local=0.67,r_GC = 8).return_D_factor(0,95)
            Dfac_in_Halo_fiducial = d_factor.D_factor_alt(rs=20,rho_local=0.4,r_GC = 8.127).return_D_factor(0,95)
            RescalingFactor = Dfac_in_Halo_fiducial/Dfac_in_Halo_Bulbul
        else:
            RescalingFactor = 1
        y2 = ax.get_ylim()[1]
        m_arr,sin_arr = wpd.read_csv(data_dir+'Bulbul2020BS.csv')
        plt.fill_between(m_arr,sin_arr*RescalingFactor,y2=y2,edgecolor=ec,facecolor=fc,zorder=z,alpha=alpha)
        plt.plot([m_arr[0],m_arr[0]],[sin_arr[0]*RescalingFactor,y2],color=ec,alpha=alpha,zorder=z)
        plt.plot([m_arr[-1],m_arr[-1]],[sin_arr[-1]*RescalingFactor,y2],color=ec,alpha=alpha,zorder=z)
        if label==True:
            plt.text(9.5,1.3e-11,r'Chandra MW Halo',fontsize=fs,color='w',rotation=0,ha='center',va='top',zorder=10)

        return
    
    def SicilianHalo2022(ax,ec='gray',fc='lightgray',z=4.1,label=False,fs=15,alpha=0.5,Rescale=True,data_dir='../data/'):
        # 2208.12271
        
        if Rescale:
            Dfac_in_Halo_Sicilian = d_factor.D_factor_alt(rs=26,rho_local=0.28,r_GC = 8).return_D_factor(0,95)
            Dfac_in_Halo_fiducial = d_factor.D_factor_alt(rs=20,rho_local=0.4,r_GC = 8.127).return_D_factor(0,95)
            RescalingFactor = Dfac_in_Halo_fiducial/Dfac_in_Halo_Sicilian
        else:
            RescalingFactor = 1
        y2 = ax.get_ylim()[1]
        m_arr,sin_arr = wpd.read_csv(data_dir+'LimitsSicilian2022_MWHalo.csv')
        plt.fill_between(m_arr,sin_arr*RescalingFactor,y2=y2,edgecolor=ec,facecolor=fc,zorder=z,alpha=alpha)
        plt.plot([m_arr[0],m_arr[0]],[sin_arr[0]*RescalingFactor,y2],color=ec,alpha=alpha,zorder=z)
        plt.plot([m_arr[-1],m_arr[-1]],[sin_arr[-1]*RescalingFactor,y2],color=ec,alpha=alpha,zorder=z)

        return
    
    def RoachHalo2022(ax,ec='gray',fc='lightgray',z=4.1,label=False,fs=15,alpha=0.5,Rescale=True,data_dir='../data/'):
        # 2207.04572
        
        if Rescale:
            Dfac_in_Halo_Roach = d_factor.D_factor_alt(rs=20,rho_local=0.4,r_GC = 8.1).return_D_factor(0,100)
            Dfac_in_Halo_fiducial = d_factor.D_factor_alt(rs=20,rho_local=0.4,r_GC = 8.127).return_D_factor(0,100)
            RescalingFactor = Dfac_in_Halo_fiducial/Dfac_in_Halo_Roach
        else:
            RescalingFactor = 1
        y2 = ax.get_ylim()[1]
        m_arr,sin_arr = wpd.read_csv(data_dir+'LimitsRoach2022_MWHalo.csv')
        plt.fill_between(m_arr,sin_arr*RescalingFactor,y2=y2,edgecolor=ec,facecolor=fc,zorder=z,alpha=alpha)
        plt.plot([m_arr[0],m_arr[0]],[sin_arr[0]*RescalingFactor,y2],color=ec,alpha=alpha,zorder=z)
        plt.plot([m_arr[-1],m_arr[-1]],[sin_arr[-1]*RescalingFactor,y2],color=ec,alpha=alpha,zorder=z)

        return
    
    def MiscXMMChandra(ax,ec='gray',fc='lightgray',z=3,fs=15,label=False,alpha=0.5,data_dir='../data/'):
        y2 = ax.get_ylim()[1]
        
        m_arr,sin_arr = wpd.read_csv(data_dir+'XrayConglomerated.csv')
        plt.plot([m_arr[0],m_arr[0]],[sin_arr[0],y2],color=fc,alpha=alpha,zorder=z)
        plt.plot([m_arr[-1],m_arr[-1]],[sin_arr[-1],y2],color=fc,alpha=alpha,zorder=z)
        plt.fill_between(m_arr,sin_arr,y2=y2,edgecolor=ec,facecolor=fc,zorder=z,alpha=alpha)
        if label==True:
            plt.text(6.25,6e-10,r'XMM and Chandra',fontsize=fs,color='w',rotation=90,ha='center',va='top',zorder=10)

    def AbajAndromeda2013(ax,ec='gray',fc='lightgray',z=2,fs=15,label=False,alpha=0.5,data_dir='../data/'):
        # 1311.0282
        y2 = ax.get_ylim()[1]
        
        m_arr,sin_arr = wpd.read_csv(data_dir+'M31_Xray.csv')
        plt.plot([m_arr[0],m_arr[0]],[sin_arr[0],y2],color=ec,alpha=alpha,zorder=z)
        plt.plot([m_arr[-1],m_arr[-1]],[sin_arr[-1],y2],color=ec,alpha=alpha,zorder=z)
        plt.fill_between(m_arr,sin_arr,y2=y2,edgecolor=ec,facecolor=fc,zorder=z,alpha=alpha)
        if label==True:
            plt.text(5.2,8e-10,r'M31',fontsize=fs,color='w',rotation=0,ha='center',va='top',zorder=5)


    def DMUnderproduction(ax,ec='gray',fc='lightgray',z=0.1,fs=20,alpha=0.2,label=False,text_col='k',data_dir='../data/'):
        y2 = ax.get_ylim()[-1]

        # arxiv: 2009.07206
        m_arr,sin_arr = wpd.read_csv(data_dir+'BBN.csv')
        plt.fill_between(m_arr,0,sin_arr,edgecolor=ec,facecolor=fc,zorder=z,alpha=alpha)
        if label==True:
            plt.text(5,2e-13,r'DM Underproduction',fontsize=fs,color=text_col,ha='center')

        return
    
    def DES_Subhalo_Counts(ax,ec='gray',fc='lightgray',z=0.1,fs=20,label=False,alpha=0.2,text_col='k',data_dir='../data/'):
        y2 = ax.get_ylim()[-1]

        # arxiv: 2008.00022
        m_arr,sin_arr = wpd.read_csv(data_dir+'DES_MWsubhalo.csv')
        sin_arr_continued = np.concatenate([sin_arr,[y2]])
        m_arr_continued = np.concatenate([m_arr,[m_arr[-1]]])
        plt.fill_betweenx(sin_arr_continued,m_arr_continued,edgecolor=ec,facecolor=fc,zorder=z,alpha=alpha)
        if label==True:
            plt.text(18,1e-10,r'DES',fontsize=fs,color=text_col,ha='center')

        return


#==============================================================================#


#==============================================================================#
def MySaveFig(fig,pltname,pngsave=True):
    fig.savefig(pltdir+pltname+'.pdf',bbox_inches='tight')
    if pngsave:
        fig.savefig(pltdir_png+pltname+'.png',bbox_inches='tight')

def cbar(mappable,extend='neither',minorticklength=8,majorticklength=10,\
            minortickwidth=2,majortickwidth=2.5,pad=0.2,side="right",orientation="vertical"):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(side, size="5%", pad=pad)
    cbar = fig.colorbar(mappable, cax=cax,extend=extend,orientation=orientation)
    cbar.ax.tick_params(which='minor',length=minorticklength,width=minortickwidth)
    cbar.ax.tick_params(which='major',length=majorticklength,width=majortickwidth)
    cbar.solids.set_edgecolor("face")

    return cbar

def MySquarePlot(xlab='',ylab='',\
                 lw=2.5,lfs=45,tfs=25,size_x=13,size_y=12,Grid=False):
    plt.rcParams['axes.linewidth'] = lw
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=tfs)
    mpl.rcParams['text.latex.preamble'] = [r'\usepackage{mathpazo}']

    fig = plt.figure(figsize=(size_x,size_y))
    ax = fig.add_subplot(111)

    ax.set_xlabel(xlab,fontsize=lfs)
    ax.set_ylabel(ylab,fontsize=lfs)

    ax.tick_params(which='major',direction='in',width=2,length=13,right=True,top=True,pad=7)
    ax.tick_params(which='minor',direction='in',width=1,length=10,right=True,top=True)
    if Grid:
        ax.grid()
    return fig,ax

def MyDoublePlot(xlab1='',ylab1='',xlab2='',ylab2='',\
                 wspace=0.25,lw=2.5,lfs=45,tfs=25,size_x=20,size_y=11,Grid=False):
    plt.rcParams['axes.linewidth'] = lw
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=tfs)
    mpl.rcParams['text.latex.preamble'] = [r'\usepackage{mathpazo}']
    fig, axarr = plt.subplots(1, 2,figsize=(size_x,size_y))
    gs = gridspec.GridSpec(1, 2)
    gs.update(wspace=wspace)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax1.tick_params(which='major',direction='in',width=2,length=13,right=True,top=True,pad=7)
    ax1.tick_params(which='minor',direction='in',width=1,length=10,right=True,top=True)
    ax2.tick_params(which='major',direction='in',width=2,length=13,right=True,top=True,pad=7)
    ax2.tick_params(which='minor',direction='in',width=1,length=10,right=True,top=True)

    ax1.set_xlabel(xlab1,fontsize=lfs)
    ax1.set_ylabel(ylab1,fontsize=lfs)

    ax2.set_xlabel(xlab2,fontsize=lfs)
    ax2.set_ylabel(ylab2,fontsize=lfs)

    if Grid:
        ax1.grid()
        ax2.grid()
    return fig,ax1,ax2


def MyTriplePlot(xlab1='',ylab1='',xlab2='',ylab2='',xlab3='',ylab3='',\
                 wspace=0.25,lw=2.5,lfs=45,tfs=25,size_x=20,size_y=7,Grid=False):
    plt.rcParams['axes.linewidth'] = lw
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=tfs)
    mpl.rcParams['text.latex.preamble'] = [r'\usepackage{mathpazo}']
    fig, axarr = plt.subplots(1, 3,figsize=(size_x,size_y))
    gs = gridspec.GridSpec(1, 3)
    gs.update(wspace=wspace)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])

    ax1.tick_params(which='major',direction='in',width=2,length=13,right=True,top=True,pad=7)
    ax1.tick_params(which='minor',direction='in',width=1,length=10,right=True,top=True)

    ax2.tick_params(which='major',direction='in',width=2,length=13,right=True,top=True,pad=7)
    ax2.tick_params(which='minor',direction='in',width=1,length=10,right=True,top=True)

    ax3.tick_params(which='major',direction='in',width=2,length=13,right=True,top=True,pad=7)
    ax3.tick_params(which='minor',direction='in',width=1,length=10,right=True,top=True)

    ax1.set_xlabel(xlab1,fontsize=lfs)
    ax1.set_ylabel(ylab1,fontsize=lfs)

    ax2.set_xlabel(xlab2,fontsize=lfs)
    ax2.set_ylabel(ylab2,fontsize=lfs)

    ax3.set_xlabel(xlab3,fontsize=lfs)
    ax3.set_ylabel(ylab3,fontsize=lfs)

    if Grid:
        ax1.grid()
        ax2.grid()
        ax3.grid()
    return fig,ax1,ax2,ax3
#==============================================================================#


#==============================================================================#
def reverse_colourmap(cmap, name = 'my_cmap_r'):
    reverse = []
    k = []

    for key in cmap._segmentdata:
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []

        for t in channel:
            data.append((1-t[0],t[2],t[1]))
        reverse.append(sorted(data))

    LinearL = dict(zip(k,reverse))
    my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL)
    return my_cmap_r
#==============================================================================#


#==============================================================================#
def col_alpha(col,alpha=0.1):
    rgb = colors.colorConverter.to_rgb(col)
    bg_rgb = [1,1,1]
    return [alpha * c1 + (1 - alpha) * c2
            for (c1, c2) in zip(rgb, bg_rgb)]
#==============================================================================#



from matplotlib import patches
from matplotlib import text as mtext
import numpy as np
import math

class CurvedText(mtext.Text):
    """
    A text object that follows an arbitrary curve.
    """
    def __init__(self, x, y, text, axes, **kwargs):
        super(CurvedText, self).__init__(x[0],y[0],' ', **kwargs)

        axes.add_artist(self)

        ##saving the curve:
        self.__x = x
        self.__y = y
        self.__zorder = self.get_zorder()

        ##creating the text objects
        self.__Characters = []
        for c in text:
            if c == ' ':
                ##make this an invisible 'a':
                t = mtext.Text(0,0,'a')
                t.set_alpha(0.0)
            else:
                t = mtext.Text(0,0,c, **kwargs)

            #resetting unnecessary arguments
            t.set_ha('center')
            t.set_rotation(0)
            t.set_zorder(self.__zorder +1)

            self.__Characters.append((c,t))
            axes.add_artist(t)


    ##overloading some member functions, to assure correct functionality
    ##on update
    def set_zorder(self, zorder):
        super(CurvedText, self).set_zorder(zorder)
        self.__zorder = self.get_zorder()
        for c,t in self.__Characters:
            t.set_zorder(self.__zorder+1)

    def draw(self, renderer, *args, **kwargs):
        """
        Overload of the Text.draw() function. Do not do
        do any drawing, but update the positions and rotation
        angles of self.__Characters.
        """
        self.update_positions(renderer)

    def update_positions(self,renderer):
        """
        Update positions and rotations of the individual text elements.
        """

        #preparations

        ##determining the aspect ratio:
        ##from https://stackoverflow.com/a/42014041/2454357

        ##data limits
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        ## Axis size on figure
        figW, figH = self.axes.get_figure().get_size_inches()
        ## Ratio of display units
        _, _, w, h = self.axes.get_position().bounds
        ##final aspect ratio
        aspect = ((figW * w)/(figH * h))*(ylim[1]-ylim[0])/(xlim[1]-xlim[0])

        #points of the curve in figure coordinates:
        x_fig,y_fig = (
            np.array(l) for l in zip(*self.axes.transData.transform([
            (i,j) for i,j in zip(self.__x,self.__y)
            ]))
        )

        #point distances in figure coordinates
        x_fig_dist = (x_fig[1:]-x_fig[:-1])
        y_fig_dist = (y_fig[1:]-y_fig[:-1])
        r_fig_dist = np.sqrt(x_fig_dist**2+y_fig_dist**2)

        #arc length in figure coordinates
        l_fig = np.insert(np.cumsum(r_fig_dist),0,0)

        #angles in figure coordinates
        rads = np.arctan2((y_fig[1:] - y_fig[:-1]),(x_fig[1:] - x_fig[:-1]))
        degs = np.rad2deg(rads)


        rel_pos = 10
        for c,t in self.__Characters:
            #finding the width of c:
            t.set_rotation(0)
            t.set_va('center')
            bbox1  = t.get_window_extent(renderer=renderer)
            w = bbox1.width
            h = bbox1.height

            #ignore all letters that don't fit:
            if rel_pos+w/2 > l_fig[-1]:
                t.set_alpha(0.0)
                rel_pos += w
                continue

            elif c != ' ':
                t.set_alpha(1.0)

            #finding the two data points between which the horizontal
            #center point of the character will be situated
            #left and right indices:
            il = np.where(rel_pos+w/2 >= l_fig)[0][-1]
            ir = np.where(rel_pos+w/2 <= l_fig)[0][0]

            #if we exactly hit a data point:
            if ir == il:
                ir += 1

            #how much of the letter width was needed to find il:
            used = l_fig[il]-rel_pos
            rel_pos = l_fig[il]

            #relative distance between il and ir where the center
            #of the character will be
            fraction = (w/2-used)/r_fig_dist[il]

            ##setting the character position in data coordinates:
            ##interpolate between the two points:
            x = self.__x[il]+fraction*(self.__x[ir]-self.__x[il])
            y = self.__y[il]+fraction*(self.__y[ir]-self.__y[il])

            #getting the offset when setting correct vertical alignment
            #in data coordinates
            t.set_va(self.get_va())
            bbox2  = t.get_window_extent(renderer=renderer)

            bbox1d = self.axes.transData.inverted().transform(bbox1)
            bbox2d = self.axes.transData.inverted().transform(bbox2)
            dr = np.array(bbox2d[0]-bbox1d[0])

            #the rotation/stretch matrix
            rad = rads[il]
            rot_mat = np.array([
                [math.cos(rad), math.sin(rad)*aspect],
                [-math.sin(rad)/aspect, math.cos(rad)]
            ])

            ##computing the offset vector of the rotated character
            drp = np.dot(dr,rot_mat)

            #setting final position and rotation:
            t.set_position(np.array([x,y])+drp)
            t.set_rotation(degs[il])

            t.set_va('center')
            t.set_ha('center')

            #updating rel_pos to right edge of character
            rel_pos += w-used
