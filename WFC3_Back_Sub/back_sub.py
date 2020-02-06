#! /usr/bin env python
"""
Code to estimate and subtract a three-components (Zodiacal, HeI from Earth, and Scattered light) background model from WFC3 IR data
"""

import numpy as np
import os,urllib.request,shutil,glob
from scipy import optimize
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
from astropy.io import fits
from astropy.modeling.functional_models import Gaussian1D
from photutils.background import Background2D
from photutils import detect_threshold, detect_sources
from wfc3tools import calwf3

module_path = os.path.split(os.path.abspath(__file__))[0]

# A few paths in case they are not set up.

# We force any CRDS to look for the HST reference file server
os.environ["CRDS_SERVER_URL"]="https://hst-crds.stsci.edu"

# If no CRDS_PATH exists, we create one
if not "CRDS_PATH" in os.environ.keys():
    os.environ["CRDS_PATH"] = os.path.join(os.environ["HOME"],"crds_cache")

# Point to the iref directory in the CRDS_PATH of iref is not already defined 
if not "iref" in os.environ.keys():
    os.environ["iref"] = "$HOME/crds_cache/references/hst/iref/"
os.environ["tref"] = os.path.join(module_path,"data/")

def test(d):
    return d
    from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
    print("de-noising")
    tt = denoise_tv_chambolle(d, weight=.5, multichannel=False)
    return tt

def get_visits(pattern):
    """
    A utility function to group existing files into a set of visits. 

    Attributes
    ----------
        pattern: a string to match to files, e.g. '*_flt.fits', '*_ima.fits', or '*_raw.fits'


    Output
    ------
        A dictionary with filter names as labels and visit names (6 letters) as sub-labels.
    """

    files = glob.glob(pattern)

    dic = {}
    for f in files:
        p,name = os.path.split(f)
        obs_id = name[0:9]
        visit = obs_id[0:6]

        try:
            filt = fits.open(f)[0].header["FILTER"]
        except KeyError:
            print("FILTER keyword not found in {}".format(f))
            continue
            
        if filt not in list(dic.keys()):
            dic[filt] = {}

        if visit not in list(dic[filt].keys()):
            dic[filt][visit] = []

        dic[filt][visit].append(obs_id)

    return dic


def get_data(obs_id):
    """
    A helper function to download a raw dataset from MAST using a direct URL

    Attributes
    ----------
    obs_id: string containing a 9 letter dataset name, e.g. 'idn604snq'

    Output
    ------
    string containing the name of the raw file that was downloaded
    """

    url = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HST/product/{}_raw.fits".format(obs_id)

    raw_name = "./{}_raw.fits".format(obs_id)
    
    if os.path.isfile(raw_name):
        os.unlink(raw_name)
        
    with urllib.request.urlopen(url) as response, open(raw_name, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    
    return raw_name









class Sub_Back():
    def __init__(self,obs_ids,grism, thr=0.05):
        #ima_names = ["{}_ima.fits" for obs_id in obs_ids]
        self.obs_ids = obs_ids
        self.raw_names = ["{}_raw.fits" for obs_id in obs_ids]
        self.thr = thr
        self.plot = True

        self.grism = grism

        self.G102_zodi_file = os.path.join(module_path,"data/G102_Zodi_CLN9_V10_b_clean.fits")
        self.G102_HeI_file = os.path.join(os.path.join(module_path,"data/G102_HeI_V9_b_clean.fits"))
        self.G102_Scatter_file = os.path.join(os.path.join(module_path,"data/G102_Scatter_V9_b_superclean.fits"))
        self.G102_FF_file = "tref$uc72113oi_pfl_patched2.fits"

        self.G141_zodi_file = os.path.join(module_path,"data/G141_Zodi_CLN9_V10_b_clean.fits")
        self.G141_HeI_file = os.path.join(os.path.join(module_path,"data/G141_HeI_V9_b_clean.fits"))
        self.G141_Scatter_file = os.path.join(os.path.join(module_path,"data/G141_Scatter_V9_b_superclean.fits"))
        self.G141_FF_file = "tref$uc721143i_pfl_patched2.fits"

        self.bit_mask = (1+2+4+8+16+32+64+128+256+1024+2048+4096)

    def load_backgrounds(self):
        if self.grism=="G102":
            print("Loading ",self.G102_zodi_file)
            self.zodi = fits.open(self.G102_zodi_file)[1].data
            print("Loading ",self.G102_HeI_file)
            self.HeI = fits.open(self.G102_HeI_file)[1].data
            print("Loading ",self.G102_Scatter_file)
            self.Scatter = fits.open(self.G102_Scatter_file)[1].data
            self.FF_file = self.G102_FF_file
        if self.grism=="G141":
            self.zodi = fits.open(self.G141_zodi_file)[1].data
            self.HeI = fits.open(self.G141_HeI_file)[1].data
            self.Scatter = fits.open(self.G141_Scatter_file)[1].data
            self.FF_file = self.G141_FF_file

    def process_obs_ids(self):
        """
        Function to perform all the required steps to remove the time varying HeI and Scattered light component as well as the Zodi
        component from a group of WFC3 IR G102 or G141 RAW files.
        This runs CALWF3 on each input RAW files, create a mask MSK file, estimates the HeI and Scattered light levels during the course
        of each observations and removes those contributions, runs CALWF3 on the result to produce an FLT file and, finally, estimates and
        remove the Zodi light from the final FLT file to generate a set of background subtracted FLT files.
        """

        self.load_backgrounds()

        raw_names = ["{}_raw.fits".format(x) for x in self.obs_ids]

        flt_names = [self.raw_to_flt(x) for x in raw_names]

        [self.create_msk("{}_flt.fits".format(x),thr=self.thr) for x in self.obs_ids]
        self.Get_HeI_Zodi_Scatter_Levels()
        self.sub_HeI_Scat()
        self.flt_names = [self.ima_to_flt("{}_ima.fits".format(x)) for x in self.obs_ids]
        [self.sub_Zodi(x) for x in self.flt_names]
        self.log_levels()


        # Second pass
        # [self.create_msk("{}_flt.fits".format(x),thr=self.thr) for x in self.obs_ids]
        # flt_names = [self.raw_to_flt(x) for x in raw_names]
        # self.Get_HeI_Zodi_Scatter_Levels()
        # self.sub_HeI_Scat()
        # self.flt_names = [self.ima_to_flt("{}_ima.fits".format(x)) for x in self.obs_ids]

        # [self.sub_Zodi(x) for x in self.flt_names]

        if self.plot:
            self.diagnostic_plots()

        return flt_names

    def log_levels(self):
        """Copy all the information from IMA into our final FLT file.
        Keywords added are: BSAMP (number of reads), HeI_#, Scat_#, HeI_#, TIME (ROUTIME), and DTIME (IMSET exposure)"""
        for flt_name in self.flt_names:
            ima_name = flt_name.split("_flt.fits")[0]+"_ima.fits"
            print(ima_name)

            with fits.open(flt_name,mode="update") as fflt:
                with fits.open(ima_name) as fima:
                    extnum = fima[0].header["NSAMP"]
                    fflt[1].header["BSAMP"] = extnum
                    for NSAMP in range(1,extnum):
                        HeI = fima["SCI",NSAMP].header["HeI_{}".format(NSAMP)]
                        Scat = fima["SCI",NSAMP].header["Scat_{}".format(NSAMP)]
                        ROUTTIME = fima["SCI",NSAMP].header["ROUTTIME"]
                        DELTATIM = fima["SCI",NSAMP].header["DELTATIM"]
                        print(NSAMP,ROUTTIME,HeI,Scat)
                        fflt[1].header["HeI_{}".format(NSAMP)] = HeI
                        fflt[1].header["Scat_{}".format(NSAMP)] = Scat
                        fflt[1].header["TIME_{}".format(NSAMP)] = ROUTTIME
                        fflt[1].header["DTIME_{}".format(NSAMP)] = DELTATIM

                        
                 
    def raw_to_flt(self,raw_name):
        """
        Function to run CALWF3 on a raw dataset. CRCORR is set to OMIT and FLATCORR is set to perform.
        If available, crds is ran to download and updat the RAW file header to point to the latest calibration files.
        The flat-field calibration file names are replaced with the ones included in this package and pointed to by
        G102_FF and G141_FF.

        Attributes
        ----------
        None

        Output
        ------
        string containing the name of the FLT file that was created

        """

        CRCORR="OMIT"
        FLATCORR="PERFORM"

        obs_id = raw_name.split("_raw")[0]
        
        files = ["{}_flt.fits".format(obs_id),"{}_ima.fits".format(obs_id)]
        for ff in files:
            if os.path.isfile(ff):
                os.unlink(ff)

        print("Processing ",raw_name)
        res = os.system("crds bestrefs --files {}  --sync-references=1  --update-bestrefs ".format(raw_name))
        if res!=0:
            print("CRDS did not run.")    

        fin = fits.open(raw_name,mode="update")
        fin[0].header["CRCORR"] = CRCORR
        fin[0].header["FLATCORR"] = FLATCORR 
        filt = fin[0].header["FILTER"]

        self.org_FF_file = fin[0].header["PFLTFILE"]

        fin[0].header["PFLTFILE"] = self.FF_file
    

        fin.close()
        calwf3(raw_name)
        flt_name = raw_name.split("_raw.fits")[0]+"_flt.fits"

        if not os.path.isfile(flt_name):
            print("raw_to_flt() failed to generate ",flt_name)
            sys.exit(1)

        return flt_name

    def ima_to_flt(self,ima_name):
        """
        Function to run CALWF3 on an exisiting IMA file. CRCORR is set to PERFORM.

        Attributes
        ----------
        ima_name string containing the name of the IMA file to process

        Output
        ------
        string containing the name of the FLT file that has been created
        """
        import wfc3tools
        from wfc3tools import wf3ir

        CRCORR="PERFORM"
        
        fin = fits.open(ima_name,mode="update")
        fin[0].header["CRCORR"] = CRCORR
        fin.close()

        obs_id = ima_name.split("_ima.fits")[0]
        flt_name = "%s_flt.fits" % (obs_id)
        if os.path.isfile(flt_name):
            os.unlink(flt_name)
        tmp_name = "%s_ima_ima.fits" % (obs_id)
        if os.path.isfile(tmp_name):
            os.unlink(tmp_name)
        tmp_name = "%s_ima_flt.fits" % (obs_id)
        if os.path.isfile(tmp_name):
            os.unlink(tmp_name)

        wf3ir(ima_name)
        
        shutil.move(tmp_name,flt_name)
        tmp_name = "%s_ima_ima.fits" % (obs_id)
        if os.path.isfile(tmp_name):
            os.unlink(tmp_name)
            
        tmp = fits.open(flt_name)[1].data

        if not os.path.isfile(flt_name):
            print("raw_to_flt() failed to generate ",flt_name)
            sys.exit(1)

        return flt_name


    def create_msk(self,flt_name,kernel_fwhm=1.25,background_box=(1014//6,2),thr=0.05,npixels=80):  
        """      
        This function will create a FITS files ipppssoot_msk.fits 

        Attributes
        ----------
        flt_name string containing the name of the FLT name to create a mask for
        kernel_fwhm Float The size of the detection kernel (default = 1.25 pixel)
        background_box Int The saie fo the background box when estimating the background (default = (1014//6,1) pixels) 
        thr Float Threshold above noise to detect signal (default = 0.05)
        npixels Int number of pixels for a spectrum to be detected (default = 100)    

        Output
        ______
            String containing the name of the MSK file
        """

        segm = self.get_mask(flt_name,kernel_fwhm=kernel_fwhm,background_box=background_box,thr=thr,npixels=npixels) 
        
        dq3 = fits.open(flt_name)["DQ"].data.astype(int)

        dq3 = dq3.astype(int)

        DQ = np.bitwise_and(dq3,np.zeros(np.shape(dq3),np.int)+ self.bit_mask)

        kernel = Gaussian2DKernel(x_stddev=1)
        segm = segm*1.
        #segm = convolve(segm, kernel)
        segm[segm>1e-5] = 1.
        segm[segm<=1e-5] = 0.
        segm[DQ>0] = 1.
        
        msk_name = flt_name.split("_flt.fits")[0]+"_msk.fits"
        fits.writeto(msk_name,segm,overwrite=True)
        return segm


    def get_mask(self,flt_name,kernel_fwhm=1.25,background_box=20,thr=0.05,npixels=100):
        """
        Function to create a mask (set to 0 for no detection and 1 for detection) appropriate to mask WFC3 slitless data. 
        Attributes
        ----------
        flt_name string containing the name of the FLT name to create a mask for
        kernel_fwhm Float The size of the detection kernel (default = 1.25 pixel)
        background_box Int The saie fo the background box when estimating the background (default = 20 pixels) 
        thr Float Threshold above noise to detect signal (default = 0.25)
        npixels Int number of pixels for a spectrum to be detected (default = 15)    

        Output
        ------
        A numpy array containing the mask
        """
        h = fits.open(flt_name)[0].header
        filt = h["FILTER"]

        fin = fits.open(flt_name)
        image = fin["SCI"].data
        err = fin["ERR"].data

        dq = fin["DQ"].data
        dq = np.bitwise_and(dq,np.zeros(np.shape(dq),np.int16)+ self.bit_mask)
        
        g = Gaussian1D(mean=0.,stddev=kernel_fwhm/2.35)
        x = np.arange(16.)-8
        a = g(x)
        kernel = np.tile(a,(16*int(kernel_fwhm+1),1)).T
        kernel = kernel/np.sum(kernel)

        b = Background2D(image,background_box)

        image = image-b.background
        threshold = thr * err
        
        image[dq>0] = 0. #np.nan
        
        mask = detect_sources(image, threshold, npixels=npixels,filter_kernel=kernel).data
        
        ok = (mask == 0.) & (dq==0)
        mask[~ok] = 1.
        
        return mask

    def Get_HeI_Zodi_Scatter_Levels(self,border=0):
        """
        Function to estimate the Zodi, HeI, and Scatter levels in each IMSET of an IMA file.
        A set of IMA files can be processed at once and the Zodi level is assumed to be identical in all of them. The HeI and Scatter
        levels are allowed to vary freely. See code by R. Ryan in Appendix of ISR WFC3 2015-17 for details.

        Atributes
        ---------
        ima_names List A list containing the names of IMA files to process together. 
        border int The number of collumns to avoid on the left and right hand side of the detector (default = 0)

        Output
        ------
        
        Zodi Float The value of Zodi scale
        HeIs Dic A dictionary containing all the HeI scale value for each  IMSET in each IMA file
        Scats Dic A dictionary containing all the Scatter scale value for each  IMSET in each IMA file

        """

        ima_names = ["{}_ima.fits".format(x) for x in self.obs_ids]

        nimas = len(ima_names)
        nexts = [fits.open(ima_name)[-1].header["EXTVER"] for ima_name in ima_names] # We drop the last ext/1st read   
        filt = fits.open(ima_names[0])[0].header["FILTER"]

        # Temp
        zodi = self.zodi*1
        HeI = self.HeI*1
        Scatter = self.Scatter*1
        
        data0s = []
        err0s = []
        samp0s = []
        dq0s = []
        dqm0s = []
        masks = []
        
        for j in range(nimas):
            obs_id = ima_names[j][0:9]
            mask = fits.open("{}_msk.fits".format(obs_id))[0].data
            masks.append([mask for ext in range(1,nexts[j])])
            data0s.append([fits.open(ima_names[j])["SCI",ext].data[5:1014+5,5:1014+5]*1 for ext in range(1,nexts[j])])
            err0s.append([fits.open(ima_names[j])["ERR",ext].data[5:1014+5,5:1014+5]*1 for ext in range(1,nexts[j])])
            dq0s.append([fits.open(ima_names[j])["DQ",ext].data[5:1014+5,5:1014+5]*1 for ext in range(1,nexts[j])])

        dqm0s = [[np.bitwise_and(dq0,np.zeros(np.shape(dq0),np.int16)+ self.bit_mask) for dq0 in dq0s[j]] for j in range(nimas)]

        ok = (np.isfinite(zodi)) & (np.isfinite(HeI)) & (np.isfinite(Scatter))
        zodi[~ok] = 0.
        HeI[~ok] = 0.
        Scatter[~ok] = 0.


        # Setting up image weights
        whts = []
        for j in range(len(ima_names)):
            whts_j = []
            for i in range(len(err0s[j])):
                err = err0s[j][i]
                err[err<=1e-6] = 1e-6
                w = 1./err**2
                w[~ok] = 0.
                whts_j.append(w)
            whts.append(whts_j)
            

        nflt = sum(nexts)
        npar = 2*nflt+1
        print("We are solving for ",npar," HeI values")
        
        v = np.zeros(npar,np.float)
        m = np.zeros([npar,npar],np.float)
        
        ii = -1
        for j in range(len(ima_names)):
            whts[j] = np.array(whts[j])
            data0s[j] = np.array(data0s[j])
            masks[j] = np.array(masks[j])
            dqm0s[j] = np.array(dqm0s[j])


            whts[j][~np.isfinite(data0s[j])] = 0.
            data0s[j][~np.isfinite(data0s[j])] = 0.
            whts[j][masks[j]>0] = 0.
            whts[j][dqm0s[j]!=0] = 0.
            
            for i in range(len(data0s[j])):
                ii = ii + 1
                print("name:",ima_names[j],"imset:",i+1,ii)
                
                img = data0s[j][i]
                wht = whts[j][i]
                
                if border>0:
                    wht[0:border] = 0.
                    wht[-border:0] = 0.
                    

                # Populate up matrix and vector
                v[ii] = np.sum(wht*data0s[j][i]*HeI)
                v[-1] += np.sum(wht*data0s[j][i]*zodi)

                m[ii,ii] = np.sum(wht*HeI*HeI)
                m[ii,-1] = np.sum(wht*HeI*zodi)
                m[-1,ii] = m[ii,-1]
                m[-1,-1] += np.sum(wht*zodi*zodi)

                v[ii+nflt] = np.sum(wht*data0s[j][i]*Scatter)
                
                m[ii+nflt,ii+nflt] = np.sum(wht*Scatter*Scatter)
            
                m[ii,ii+nflt] = np.sum(wht*HeI*Scatter)
                
                m[ii+nflt,-1] = np.sum(wht*zodi*Scatter)
                
                m[ii+nflt,ii] = m[ii,ii+nflt]
                
                m[-1,ii+nflt] = m[ii+nflt,-1]
       
        
        res = optimize.lsq_linear(m,v)
        x = res.x

        # res = optimize.nnls(m,v)
        # x = res[0]

        
        Zodi = x[-1]
        HeIs = {}
        Scats = {}
        ii = -1
        for j in range(len(data0s)):
            HeIs[ima_names[j]] = {}
            Scats[ima_names[j]] = {}
            for i in range(len(data0s[j])):
                ii = ii + 1
                print("%s %d Zodi: %3.3f  He: %3.3f S: %3.3f" % (ima_names[j],i,x[-1],x[ii],x[ii+nflt]))

                HeIs[ima_names[j]][i+1] = x[ii]
                Scats[ima_names[j]][i+1] = x[ii+nflt]

        self.Zodi = Zodi
        self.HeIs = HeIs
        self.Scats = Scats

    def sub_HeI_Scat(self):
        """
        Function to subtract the appropriately scaled HeI and Scatter light models from each of the IMSET of the IMA files 
        included in the HeIs and Scats dictionaries. Header keywords are populated to reflect the amount of HeI and Scattered light
        subtracted. Function will fail to run a second time on a dataset.

        """

        for f in self.HeIs.keys():
            print("Updating ",f)
            fin = fits.open(f,mode="update")
            
            filt = fin[0].header["FILTER"]

            zodi = self.zodi*1
            HeI = self.HeI *1
            Scatter = self.Scatter *1

            for extver in self.HeIs[f].keys():
                print("EXTVER:",extver)
                try:
                    val = fin["SCI",extver].header["HeI"] # testing
                    print("HeI found in ",f,"Aborting..")
                    continue
                except:
                    pass
        
                print("IMSET:",extver,"subtracting",self.HeIs[f][extver],self.Scats[f][extver])
                print("Before:",np.nanmedian(fin["SCI",extver].data[5:1014+5,5:1014+5] ))
                fin["SCI",extver].data[5:1014+5,5:1014+5] = fin["SCI",extver].data[5:1014+5,5:1014+5] - self.HeIs[f][extver]*HeI - self.Scats[f][extver]*Scatter 
                print("After:",np.nanmedian(fin["SCI",extver].data[5:1014+5,5:1014+5] ))
                fin["SCI",extver].header["HeI_{}".format(extver)] = (self.HeIs[f][extver],"HeI level subtracted (e-)")
                fin["SCI",extver].header["Scat_{}".format(extver)] = (self.Scats[f][extver],"Scat level estimated (e-)")

            fin.close()

    def sub_Zodi(self,flt_name):
        """
        Function to re-compute and subtract the Zodi component from an FLT file. A match MSK file is assumed to exist.

        Attributes
        ----------
        flt_name String containing the name of the FLT file to process

        Output
        ------
        None

        """

        obs_id = os.path.split(flt_name)[-1][0:9]
        fin = fits.open(flt_name,mode="update")

        try:
            val = fin["SCI"].header["Zodi"] # testing
            print("Subtracted Zodi level found in ",flt_name,"Aborting..")
            return
        except:
            pass

        if not os.path.isfile("{}_msk.fits".format(obs_id)):
            print("sub_Zodi could not find the MSK file ","{}_msk.fits".format(obs_id))
            sys.exit(1)

        filt = fin[0].header["FILTER"]
        
        zodi = self.zodi*1
            
        # d = fin["SCI"].data
        # dq0 = fin["DQ"].data
        # dq = np.bitwise_and(dq0,np.zeros(np.shape(dq0),np.int16)+ self.bit_mask) 

        # msk = fits.open("{}_msk.fits".format(obs_id))[0].data
        # ok = (msk==0) & (dq==0)

        # tmp = test(d)*1.
        # tmp[~ok] = np.nan
        # scale = np.nanmedian(tmp/test(zodi))
        print("Zodi, Scale",self.Zodi)

        fin["SCI"].data = fin["SCI"].data - zodi*self.Zodi
        fin["SCI"].header["Zodi"] = (self.Zodi,"Zodi level estimated (e-)")
        fin.close(output_verify="ignore")

    def plot_levels(self):
        """Generate a plot showing the Zodi, HeI, and Scattered light levels for each IMSET read"""
        from astropy.io import fits
        import matplotlib.pyplot as plt

        for n,f in enumerate(self.flt_names):
            print(f)
            with fits.open(f) as fin:
                h = fin["SCI",1].header
                zodi = h["ZODI"]
                xs = []
                ys1 = []
                ys2 = []
                for i in range(1,h["BSAMP"]):
                    TIME = h["TIME_{}".format(i)]
                    DTIME = h["DTIME_{}".format(i)]
                    STIME = TIME-DTIME/24/3600
                    HeI = h["HeI_{}".format(i)]
                    Scat = h["Scat_{}".format(i)]
                    xs.append(STIME)
                    ys1.append(HeI)
                    ys2.append(Scat)
                    plt.axvspan(STIME,TIME,alpha=0.2)


                print(xs,ys1)
                plt.text(TIME,-.2,f[0:9])
                label = None
                if n==0:
                    label = "Zodiacal"
                plt.axhline(zodi,label=label)        
                label = None
                if n==0:
                    label = "HeI"
                plt.scatter(xs,ys1,color='g',label=label)
                label = None
                if n==0:
                    label = "Scattered"
                plt.scatter(xs,ys2,color='r',label=label)
                bottom, top = plt.ylim() 
                plt.ylim(bottom=-0.25,top=top)
                plt.legend()
        plt.grid()
        plt.xlabel("UT Time (MJD)")
        plt.ylabel(r'$e^-/s$')

    def diagnostic_plots(self):
        """
        Function to output diagnostic plots for each of the processed observations, plotting the median residuals in the
        final background subtracted FLT files (after applying the detection mask).
        Attributes
        ----------
        obs_ids List containing the IDs of the FLT files to process

        Output
        ------
        Name of the plot file
        """
        from astropy.io import fits
        import numpy as np
        import scipy
        import matplotlib.pyplot as plt

        plt.rcParams["figure.figsize"] = (10,2.5*(len(self.obs_ids)+1))
        fig = plt.figure()
        plt.clf()
        plt.subplot(len(self.obs_ids)+1,1,1)
        self.plot_levels()
        
        for i,obs in enumerate(self.obs_ids):
            plt.subplot(len(self.obs_ids)+1,1,i+2)
            f = "{}_flt.fits".format(obs)
            d = fits.open(f)[1].data
            dq = fits.open(f)["DQ"].data
            dq = np.bitwise_and(dq,np.zeros(np.shape(dq),np.int16)+ self.bit_mask)
            f = "{}_msk.fits".format(obs)
            m = fits.open(f)[0].data
            ok = (m==0) & (np.isfinite(d))
            d[m>0]=np.nan
            plt.plot(np.nanmedian(d,axis=0),label=obs)
            plt.grid()
            plt.ylabel("e-/s")
            plt.xlabel("col")
            plt.xlim(0,1014)
            plt.ylim(-0.02,0.02)
            plt.legend()
        plt.tight_layout()
        oname = "{}_diag.png".format(self.obs_ids[0][0:6])
        plt.savefig(oname)
        return "{}_diag.png".format(self.obs_ids[0][0:6])

    def restore_FF(self):
        """
        Function to undo the flattening of an FLT file by de-applying the flat-field that was used and instead
        re-applying the default pipeline flat-field which only corrects for the quandrant gain values.
        """
        
        f1 = os.path.join(os.environ["tref"],self.FF_file.split("$")[-1])
        f2 = os.path.join(os.environ["iref"],self.org_FF_file.split("$")[-1])
        with fits.open(f1) as fin:
            FF = fin[1].data[5:1014+5,5:1014+5]

        with fits.open(f2) as fin:
            org_FF = fin[1].data[5:1014+5,5:1014+5]
        
        for flt_name in self.flt_names:
            with fits.open(flt_name,mode="update") as fin:
                fin["SCI"].data = (fin["SCI"].data)*FF / org_FF
                fin["ERR"].data = fin["ERR"].data*FF / org_FF
                fin["SCI"].header["PFLTFILE"] = self.org_FF_file

      
    

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some WFC3 IR grism observations.')
    parser.add_argument('files', default='*_raw.fits', metavar='raw_files', 
                    help='Files to process, e.g. "*_raw.fits"')
    
    parser.add_argument('--grism',choices=['G102', 'G141', 'Both'], default='Both', help='Filters to process, G102, G141, or Both (default)')
    parser.add_argument('--ipppss', default='All')
    parser.add_argument('--grey_flat', default=True, help='Set to False if pflat flat-fielding is not wanted for final FLT file.')

    args = parser.parse_args()
    print("args:",args.files,args.grey_flat)
    sys.exit(1)
    if args.grism == "Both":
        grisms  = ["G102","G141"]
    else:
        grisms = [args.grism]

    obs_ids = get_visits(args.files)

    for grism in grisms:
        if not grism in obs_ids.keys():
            continue
        if args.ipppss == 'All':
            for ipppss in list(obs_ids[grism].keys()):
                t = Sub_Back(obs_ids[grism][ipppss],grism)
                t.process_obs_ids()
                t.diagnostic_plots()
                if not args.grey_flat:
                    t.restore_FF()
        else:
            ipppss = args.ipppss
            t = Sub_Back(obs_ids[grism][ipppss],grism)
            t.process_obs_ids()
            t.diagnostic_plots()
            if not args.grey_flat:
                    t.restore_FF()


