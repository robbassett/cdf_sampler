import numpy as np
import cdf_sampler as cds
from matplotlib import pyplot as plt

def gauss(x,A,x0,sig):
    return A*np.exp(((-1.)*(x-x0)*(x-x0))/(2.*np.pi*sig*sig))

def match_function():
    x = np.arange(1.,50.,.001)
    y1= gauss(x,55.,12.,2.5)
    y2= gauss(x,33.,33.,1.1)
    y3= 1.+2.*x
    y=y1+y2+y3

    z=cds.cdf_sampler(x,y)
    z.sample_n(10000)

    f = plt.figure()
    a2= f.add_subplot(111)
    a2.hist(z.sample,bins=35)
    ax=a2.twinx()
    ax.plot(x,y,'r-',lw=3)
    plt.show()

def histo_sampler(nbins,osf):
    d=np.load('./z3p5_100tau.npy')
    t911 = np.exp((-1.)*d[:,140])
    red  = 1.
    if osf >= 50: red = 3.
    nb2  = int(nbins*osf/red)

    y,bin_edges = np.histogram(t911,bins=nbins)
    slr = cds.histogram_oversampler(bin_edges,y,osf,spline=True)
    slr.sample_n(10000)

    F  = plt.figure()
    ax = F.add_subplot(221)
    ax.hist(t911,bins=nbins,histtype='step',color='k',lw=3)
    ax.plot(slr.spl[0],slr.spl[1],'g--')
    a2 = ax.twinx()
    a2.hist(slr.sample,bins=nbins,histtype='step',color='r')
    a3 = F.add_subplot(222)
    a3.hist(slr.sample,bins=nb2,histtype='step',color='r')
    a4 = a3.twinx()
    a4.plot(slr.spl[0],slr.spl[1],'g--')
    ax.set_yticks([])
    a2.set_yticks([])
    a3.set_yticks([])
    a4.set_yticks([])

    slr = cds.histogram_oversampler(bin_edges,y,osf,spline=False)
    slr.sample_n(10000)
    ax = F.add_subplot(223)
    ax.hist(t911,bins=nbins,histtype='step',color='k',lw=3)
    a2 = ax.twinx()
    a2.hist(slr.sample,bins=nbins,histtype='step',color='r')
    a3 = F.add_subplot(224)
    a3.hist(slr.sample,bins=nb2,histtype='step',color='r')
    ax.set_yticks([])
    a2.set_yticks([])
    a3.set_yticks([])
    
    plt.subplots_adjust(hspace=.1,wspace=.1)
    plt.show()

# Simple match to function
match_function()

# Example of the histogram oversampler
histo_sampler(40,100)
