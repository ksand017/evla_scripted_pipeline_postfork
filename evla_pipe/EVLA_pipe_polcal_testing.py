"""
Prepare for calibrations
========================

Fill models for all primary calibrators.
NB: in CASA 3.4.0 can only set models based on field ID and spw, not
by intents or scans
"""

import numpy as np
import scipy as sp
import math
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from casatasks import gaincal, applycal, polcal, setjy, split, fluxscale, casalog, flagdata, flagmanager, tclean, widebandpbcor, imsubimage, imhead
from . import pipeline_save
from .utils import runtiming, logprint, find_standards, find_EVLA_band, RefAntHeuristics, MAINLOG

from casatools import image

pi = np.pi
ia = image()

def calc_rms(spec):
    '''
    calculates the rms of an input array
    '''
    return np.sqrt(np.nanmean(spec**2))

def task_logprint(msg):
    logprint(msg, logfileout="logs/polcal.log")

def S(f,S,alpha,beta):
    '''
    Power law model given input alpha, beta, and frequency array. 
    refFreqI is a globally defined variable (GHz)
    '''
    return S*(f/reffreq)**(alpha+beta*np.log10(f/reffreq))

def fitterI(freqI_band,a,b):
    '''
    fitterI fits flux I to the setjy equation, to get an alpha and beta
    (i.e. spix[0] and spix[1], respectively)
    '''
    I = np.array([])
    for i in range(len(freqI_band)):
            I = np.append(I, i_ref*(freqI_band[i]/refFreqI)**(a + (b*np.log10(freqI_band[i]/refFreqI))))
    return I

def fitter3(x,a,b,c,d):
     
    '''
    fitter4 is a 4th order polynomial that is used to fit polarization
    fraction (p)
    '''
    return a + b*x + c*(x**2) + d*(x**3) 


def fitter4(x,a,b,c,d,e):
    '''
    fitter4 is a 4th order polynomial that is used to fit polarization
    fraction (p)
    '''
    return a + b*x + c*(x**2) + d*(x**3) + e*(x**4)


def polyFit(polAngleSource, band, refFreq):
    '''
    polyFit fits a polynomial of order 4 (default) for the polarization index
    (polindex) input for CASA command setjy dependent on flux density calibrator
    and band.  The polarization index is a list of coefficients for frequency-dep.
    linear polarization fraction, as similarly stated in CASA setjy documentation.

    Inputs:
            polAngleSource - one of 4 flux density calibrator source 
                ('3C48', '3C138', '3C147', or '3C286')
            band - receiver band used in observation scan for polAngleSource
                   ('L', 'S', 'C', 'X', 'Ku', 'K', 'Ka', 'Q')
            refFreq - reference frequency in band from frequency Stokes I array
            order - degree of fit (it is 4); 
                    this is for fitting the polynomial

    Outputs:
            coeffs - list of freq. dep. coefficients for linear pol. frac.
            p_ref - polarization fraction based on fit
            RM - rotation measure, already defined for each source
            X_0 - intrinsic polarization angle, already defined as well

    '''



    # RM and X_0 values come from Table 4. RM Values for the Four Sources
    # (also from 2013 Perley, Butler paper)
    cal_data2013 = np.genfromtxt('/lustre/aoc/students/ksanders/EVLA_SCRIPTED_PIPELINE/EVLA_SP_postfork/evla_scripted_pipeline_postfork/data/PolCals_2013_3C48.3C138.3C147.3C286.dat')

    freqFitting = cal_data2013[:,0]
    print('3C48' in polAngleSource)
    if ('3C48' in polAngleSource):
            pol_perc = cal_data2013[:,1]
            pol_angle = cal_data2013[:,2]

            task_logprint("Polarization angle calibrator is 3C48\n")
    elif ("3C138" in polAngleSource):
            pol_perc = cal_data2013[:,3]
            pol_angle = cal_data2013[:,4]
            task_logprint("Polarization angle calibrator is 3C138\n")
            freqFitting = np.delete(freqFitting, 7, None)
            task_logprint("Not evaluating at frequency 3.75 GHz due to null value"
                  "for polarization fraction at that frequency.")
    elif ("3C147" in polAngleSource):
            if ((band == 'L') or (band == 'S')):
                    task_logprint("Unable to fit with L or S band for 3C147\n"
                          "Quitting script.\n")
                    # quit program, as unable to continue
                    exit()
            pol_perc = cal_data2013[:,5]
            pol_angle = cal_data2013[:,6]

            task_logprint("Polarization angle calibrator is 3C147\n")
    elif ("3C286" in polAngleSource):
            pol_perc = cal_data2013[:,7]
            pol_angle = cal_data2013[:,8]

            task_logprint("Polarization angle calibrator is 3C286\n")
    else:
    # should never get here, as should have already checked that it was 1
            task_logprint("This source is not in this table.\n"
                  "Options are 3C48, 3C138, 3C147, 3C286")
            exit()

    # Fit by band
    if (band == 'L'):
            freqFitting = freqFitting[:4]
            pol = pol_perc[:4]
            ang = pol_angle[:4]
    elif (band == 'S'):
            freqFitting = freqFitting[4:8]
            pol = pol_perc[4:8]
            ang = pol_angle[4:8]
    elif (band == 'C'):
            freqFitting = freqFitting[8:12]
            pol = pol_perc[8:12]
            ang = pol_angle[8:12]
    elif (band == 'X'):
            freqFitting = freqFitting[12:14]
            pol = pol_perc[12:14]
            ang = pol_angle[12:14]
    elif (band == 'Ku'):
            freqFitting = freqFitting[14:18]
            pol = pol_perc[14:18]
            ang = pol_angle[14:18]
    elif (band == 'K'):
            freqFitting = freqFitting[18:22]
            pol = pol_perc[18:22]
            ang = pol_angle[18:22]
    elif (band == 'Ka'): # TO DO: these ones need to be more robust
            freqFitting = freqFitting[22]
            pol = pol_perc[22]
            ang = pol_angle[22]
    elif (band == 'Q'):
            freqFitting = freqFitting[23]
            pol = pol_perc[23]
            ang = pol_angle[23]

    task_logprint("frequencies = " +str(freqFitting))
    task_logprint("reference frequency = "+str(refFreq))
    task_logprint("polarization percentages = "+ str(pol))
    task_logprint("polarization angles = "+ str(ang))
    

    x_data = (freqFitting - refFreq) / refFreq
    # popt_pf, pcov_pf = sp.optimize.curve_fit(fitter3, x_data, pol/100.)
    # popt_pa, pcov_pa = sp.optimize.curve_fit(fitter3, x_data, ang*(np.pi/180))	

    popt_pf = np.flip(np.polyfit(x_data, pol/100.0, 3))
    print('popt_pf_test = ', popt_pf)

    popt_pa = np.flip(np.polyfit(x_data, np.deg2rad(ang), 3))
    print('popt_pa_test = ', popt_pa)
    # refFreq - refFreq = 0
    p_ref = np.polyval(popt_pf[::-1], 0.0)

    #print([popt, p_ref, RM, X_0])
    return popt_pf, popt_pa, p_ref, pol, ang
'''
START OF POLCAL SCRIPT
'''
task_logprint("*** Starting EVLA_pipe_polcal_testing_v2.py ***")
time_list = runtiming("polcal", "start")
QA2_polcal = "Pass"
spw_flg_thresh = 0.50

print('leak_polcal_type', leak_polcal_type)

if do_pol == True:
    use_parang = True

    # define name of polarization calibration measurement set
    if (ms_active[-1] == '/'):
        visPola = ms_active[:-4]+"_pola_cal.ms/"
    elif (ms_active[-1] == 's'):
        visPola = ms_active[:-3]+"_pola_cal.ms/"
    else:
        visPola = 'pola_cal.ms/'
    task_logprint("Polarization .ms is "+visPola+"\n")
    
    # split out data column from .ms from user input
    if not os.path.exists(visPola):
        try:
            split(vis=ms_active, outputvis=visPola, datacolumn='corrected')
        except:
            split(vis=ms_active, outputvis=visPola, datacolumn='data')

    #create listobs file for pol MS
    visPola_listname = visPola.rstrip("ms/")+"listobs.txt"
    os.system(f"rm -rf {visPola_listname}")
    listobs(visPola, listfile=visPola_listname)
    
    plotms(vis=visPola,field=polAngleField,correlation='RR',
           timerange='',antenna='',
           xaxis='frequency',yaxis='amp',ydatacolumn='model', plotfile=str(polAngleField)+'_ampvsfreq_RR_model_presetjy.png', overwrite=True)
    plotms(vis=visPola,field=polAngleField,correlation='RR',
           timerange='',antenna='',
           xaxis='frequency',yaxis='amp',ydatacolumn='corrected', plotfile=str(polAngleField)+'_ampvsfreq_RR_corecteddata_preflag.png', overwrite=True)

    task_logprint("Doing inital flagging of polarization .ms")
    flagdata(vis=visPola, mode='tfcrop', field=calibrator_field_select_string, correlation='', freqfit='poly', extendflags=False, flagbackup=False)
    flagdata(vis=visPola, mode='rflag', field=calibrator_field_select_string, correlation='RL,LR', datacolumn='data', extendflags=True, flagbackup=False)

    task_logprint("Finding a reference antenna")

    refantfield = polarization_angle_field_select_string
    findrefant = RefAntHeuristics(
        vis=visPola,
        field=refantfield,
        geometry=True,
        flagging=True,
    )
    RefAntOutput = findrefant.calculate()
    refAnt = 'ea10' #casaguide chosen refant
    #refAnt = str(RefAntOutput[0])

    task_logprint(f"The pipeline will use antenna {refAnt} as the reference")



    # set the stokes I model for the polarization angle calibrator and plot results
    flux_dict = setjy(vis=visPola, field=polarization_angle_field_select_string)
    
    plotms(vis=visPola,field=polAngleField,correlation='RR',
           timerange='',antenna='',
           xaxis='frequency',yaxis='amp',ydatacolumn='model', plotfile=str(polAngleField)+'_ampvsfreq_RR_model_postsetjy.png', overwrite=True)
    plotms(vis=visPola,field=polAngleField,correlation='RR',
           timerange='',antenna='',
           xaxis='frequency',yaxis='amp',ydatacolumn='corrected', plotfile=str(polAngleField)+'_ampvsfreq_RR_corecteddata_postflag.png', overwrite=True)
    




    # in an array place each Stokes I flux density value per spw for all bands
    fluxI = np.array([])
    for i in range(len(flux_dict[polarization_angle_field_select_string])-1):
        # polAngleField = field_num, str(i) = spw, fluxd = fluxd, 0 = I
        fluxI = np.append(fluxI, flux_dict[polarization_angle_field_select_string][str(i)]['fluxd'][0])

    task_logprint("fluxI from setjy is :"+str(fluxI)+"\n")


    task_logprint("reference_frequencies = %s" % reference_frequencies)
    try:
        tb.open(visPola+'SPECTRAL_WINDOW')
        freqI = tb.getcol('REF_FREQUENCY')
        spwNum = freqI.shape[0]
        task_logprint("spwNum currently using = "+str(spwNum)+"\n")

        # this is the number of channels, so below is also some math to 
        # determine which ones want to include for input
        task_logprint("Please note that assuming this is a continuum scan, so assuming that "
                  "all spectral windows have same number of channels\n")
        chanNum = tb.getcol('CHAN_FREQ').shape[0]
        val = 0.1*chanNum
        upper = int(math.ceil(chanNum - val))
        lower = int(val)
        task_logprint("channels are "+str(chanNum))
        task_logprint("upper = "+str(upper))
        task_logprint("lower = "+str(lower))
    
        bandsList = unique_bands_string
        print("bands are: ", bandsList, "\n")

        spwL = 0
        spwS = 0
        spwC = 0
        spwX = 0
        spwKu = 0
        spwK = 0
        spwKa = 0
        spwQ = 0
        bandList = tb.getcol('NAME')
    finally:
        tb.close()
    # first determine what frequency bands are in the .ms
    bands = set()
    for i in range(len(bandList)):
        if ('_L#' in bandList[i]):
            bands.add('L')
            spwL = i
        elif ('_S#' in bandList[i]):
            bands.add('S')
            spwS = i
        elif ('_C#' in bandList[i]):
            bands.add('C')
            spwC = i
        elif ('_X#' in bandList[i]):
            bands.add('X')
            spwX = i
        elif ('_KU#' in bandList[i]):
            bands.add('Ku')
            spwKu = i
        elif ('_K#' in bandList[i]):
            bands.add('K')
            spwK = i
        elif ('_KA#' in bandList[i]):
            bands.add('Ka')
            spwKa = i
        elif ('_V#' in bandList[i]):
            bands.add('Q')
            spwQ = i
        else:
            task_logprint("Unable to do calibration for this "+str(bandList[i])+"\n")
            quit()
        
    task_logprint("frequency array is : "+str(freqI)+"\n")

    task_logprint("bands are: "+str(bands)+"\n")
    task_logprint("bands are: "+str(bandsList)+"\n")
    unique_spw_band_map =  np.unique(spw_band_map)
    # check bands in bandlist to make sure only calibrating science bands, not pointing bands
    # split off each band and check intents?
    bandList_temp = []
    bandSPW = []
    print('spw_band_map = ', spw_band_map)
    print('unique spw_band_map = ', np.unique(spw_band_map))
    for band in unique_spw_band_map:
        print('band is ', band)
        bandSPW_temp = []
        for i in range(len(spw_band_map)):
            if spw_band_map[i] == band:
                bandSPW_temp.append(i)
                print('bandSPW_temp = ', bandSPW_temp)
        bandSPW.append(bandSPW_temp)
        band_outvisname = band+'band.ms/'
        os.system(f"rm -rf {band_outvisname}")
        split(vis = visPola, outputvis = band_outvisname, spw = str(bandSPW_temp[0])+'~'+str(bandSPW_temp[-1]), datacolumn='data')
        
    #print('bandSPW = ', bandSPW)
    #print('bandList = ', bandList)
    #spw_start = 0
    
    # example: bandsList = ['L', 'S', 'K']
    for b in range(len(unique_spw_band_map)):
        band = unique_spw_band_map[b]
        spw_start = bandSPW[b][0]
        spw_end = bandSPW[b][-1]

        task_logprint("Attempting to calibrate band: "+band+"\n")
        try:
            tb.open(band+'band.ms/')
            unique_scan_nums = np.unique(tb.getcol('SCAN_NUMBER'))
        finally:
            tb.close()
        if len(unique_scan_nums) <= 3:
            task_logprint('This band does not have enough scans to calibrate!')
            continue

        
        
        ###FIXME: add step to determine the spws perbaseband within a band
        try:
            tb.open(band+'band.ms/SPECTRAL_WINDOW/')
            spw_names = tb.getcol('NAME')
            print('spw_names = ', spw_names)
            bb_names = [spw_names[i].rsplit('#')[1] for i in range(len(spw_names))]
            unique_bb_names = np.unique(bb_names)
            SPW_per_bb = []
            for i in range(len(unique_bb_names)):
                bb_name = unique_bb_names[i]
                print(bb_name)
                spws = []
                for j in range(len(spw_names)):
                    if bb_name in spw_names[j]:
                        spws.append(spw_names[j].rsplit('#')[-1])
                SPW_per_bb.append(spws)
        finally:
            tb.close()
        print('SPW_per_bb = ', SPW_per_bb)
        kcross_spw_map_temp = []
        for i in range(len(SPW_per_bb)):
                print(i, len(SPW_per_bb[i]))
                spws = SPW_per_bb[i]
                kcross_spw_map_temp = np.append(kcross_spw_map_temp, len(spws)*[int(spws[0])-int(SPW_per_bb[0][0])])
        print('kcross_spw_map = ', kcross_spw_map_temp)
        
        freqI_band = freqI[spw_start:spw_end+1]
        fluxI_band = fluxI[spw_start:spw_end+1]

        task_logprint('spw_start, spw_end = '+str(spw_start)+" - " +str(spw_end))

        task_logprint("freqI for band is "+str(freqI_band)+"\n")
        task_logprint("fluxI for band is "+str(fluxI_band)+"\n")
        task_logprint("len of freqI_band is "+str(len(freqI_band))+ "\ncompared to len of freqI, which is "+str(len(freqI))+"\n")

        refFreq = (min(freqI_band) + max(freqI_band)) / 2
        task_logprint("ref Freq would be :"+str(refFreq)+" Hz \n")
        diff = abs(refFreq - freqI_band[0])
        # begin process of finding which frequency in freqI_band is closest to
        # calculated reference frequency in the given frequencies (refFreq)

        for v in range(len(freqI_band)):
            # determine reference frequency from given frequencies
            if (diff >= abs(refFreq - freqI_band[v])):
                # found a freq. in freqI_band that is closer to calculated
                # reference frequency (refFreq)
                diff = abs(refFreq - freqI_band[v])
                refFreqI = freqI_band[v]
                # determine Stokes I value at the reference frequency
                i_ref = fluxI_band[v]

        task_logprint("refFreqI is :"+str(refFreqI)+" Hz \n")

        # determine polindex coefficients, polarization fraction, RM, and X_0
        # based on calibrator and band
        # NOTE: this assumes that refFreqI is given in terms of Hz,
        # so that is why dividing by 1e+09, to put in terms of GHz
        #polOut =
        coeffs_pf, coeffs_pa, p_ref, model_pol, model_ang = polyFit(polAngleField, band, refFreqI/1e+09)
        task_logprint("polyFit output:")
        print(coeffs_pf, coeffs_pa, p_ref)

        # check if the input model has PA values < -90 or greater than +90.
        # if PA < -90, the model written to setjy is flipped and the measured 
        # values from the final check of the pipeline will need to have 180 
        # subtracted to be compared to the input model values
        # if PA > +90, the measured values from the final check of the pipeline
        # will need to have 180 added to be compared to the input model values...
        model_flip_plus = False
        model_flip_minus = False
        if np.rad2deg(coeffs_pa[0]) < -90:
            model_flip_minus = True
        elif np.rad2deg(coeffs_pa[0]) > 90:
            model_flip_plus = True

        

        task_logprint("polindex input will be: "+str(coeffs_pf))
        task_logprint("polangle input will be: "+str(coeffs_pa))

        #p_ref = polOut[1]
        task_logprint("p_ref is "+str(p_ref))
        # print(stop)
        #RM = polOut[2]
        #X_0 = polOut[3] # this is in terms of radians

        RM = -68 # rad/m^2
        # # X_0 = 122*pi/180 # in radians
        # X_0 = (-100+180)*pi/180 # in radians

        # calculate Stokes Q and U
        # q_ref = p_ref*i_ref*np.cos(2*X_0)
        # u_ref = p_ref*i_ref*np.sin(2*X_0)

        # task_logprint("Flux Dict passed to setjy is: "+str(i_ref)+" "+str(q_ref)+" "+str(u_ref))

        task_logprint("Determining setjy spix input\n")
        popt_I, pcov_I = sp.optimize.curve_fit(fitterI,freqI_band,fluxI_band)
        print('popt_I = ', popt_I)
        print(str(spw_start)+'~'+str(spw_end))

        # Set the polarized model for the pol angle calibrator
        setjy_full_dict = setjy(vis=visPola, standard='manual', field=polAngleField, 
                            spw=str(spw_start)+'~'+str(spw_end),
                            fluxdensity=[i_ref, 0, 0, 0],
                            spix = popt_I,
                            reffreq=str(refFreqI)+'Hz',
                            polindex = coeffs_pf,
                            polangle = coeffs_pa,
                            usescratch=True,
                            rotmeas = 0.0)
        # task_logprint('setjy_full_dict = ')
        print('setjy_full_dict = ', setjy_full_dict)
        # set the model
        # setjy_full_dict = setjy(vis=visPola, standard='manual', field=polAngleField, 
        #                         spw=str(spw_start)+'~'+str(spw_end),
        #                         fluxdensity=[i_ref, q_ref, u_ref, 0],
        #                         spix = popt_I,
        #                         rotmeas = RM,
        #                         reffreq=str(refFreqI)+'Hz',
        #                         polindex = coeffs_pf,
        #                         usescratch=True)

        #print(stop)
        plotms(vis=visPola,field=polAngleField,correlation='RL',
               timerange='',antenna='',
               xaxis='frequency',yaxis='amp',ydatacolumn='model', plotfile=str(polAngleField)+'_ampvsfreq_RL_polarizedmodel.png', overwrite=True)

        plotms(vis=visPola,field=polAngleField,correlation='RL',
               timerange='',antenna='',
               xaxis='frequency',yaxis='phase',ydatacolumn='model', plotfile=str(polAngleField)+'_phasevsfreq_RL_polarizedmodel.png', overwrite=True)

        plotms(vis=visPola,field=polAngleField,correlation='RL',
               timerange='',antenna='',
               xaxis='frequency',yaxis='Real',ydatacolumn='model', plotfile=str(polAngleField)+'_RLRealvsfreq_Q_polarizedmodel.png', overwrite=True)
        
        plotms(vis=visPola,field=polAngleField,correlation='RL',
               timerange='',antenna='',
               xaxis='frequency',yaxis='Imag', ydatacolumn='model', plotfile=str(polAngleField)+'_RLImagvsfreq_U_polarizedmodel.png', overwrite=True)
        # print(stop)
        # Solving for the Cross Hand Single Band Delays
        kcross_sbd = polAngleField+'_'+band+'_band_data.sbd.Kcross'
        gaincal(vis=visPola, caltable=kcross_sbd, field=polAngleField,
                spw=str(spw_start)+'~'+str(spw_end)+':'+str(lower)+'~'+str(upper),
                gaintype='KCROSS',
                solint='inf',
                combine='scan', 
                refant=refAnt,
                gaintable=[''],
                gainfield=[''],
                parang=True)

        plotms(vis=kcross_sbd, xaxis='frequency', yaxis='delay', antenna = refAnt, coloraxis='corr', plotfile=str(polAngleField)+'_delayvsfreq_kcross_sbd_sol.png', overwrite=True)
        task_logprint('Checking delay values and rms...')
        use_mbd = False
        try:
            tb.open(kcross_sbd)
            delays_all = tb.getcol('FPARAM')[0][0]
            delays = np.array([x for i, x in enumerate(delays_all) if x not in delays_all[:i]])
            #delays = np.array([0.0, 2.0, 3.5, 5.5, 7.0, 10.3, 12.0])
            task_logprint('delays = %s' % delays)
            print(np.unique(delays_all))
            print(delays_all)
        finally:
            tb.close()
        

        for i in range(len(SPW_per_bb)):
            spws = SPW_per_bb[i]
            #print('spws = ', spws)
            if SPW_per_bb[0][0] == '0':
                task_logprint('SPW indexing starts at 0')
                spws = [int(spws[j]) for j in range(len(spws))]
            else:
                spws = [int(spws[j])-int(SPW_per_bb[0][0]) for j in range(len(spws))]
            task_logprint('delays per bb = %s' % delays[spws])
            bb_rms = calc_rms(delays[spws])
            task_logprint('rms across baseband = %s' % bb_rms)
            if bb_rms > 3.0:
                task_logprint('RMS of sbd solutions for this baseband > 3.0 ns...')
                task_logprint('Doing mbd kcross solutions instead...')
                use_mbd = True
        
        d_bool = delays > 10.0
        d_bool_check = True in d_bool
        flagmanager(vis = visPola, mode='save', versionname='pre_kcross_mbd_flagging')
        
        if d_bool_check == True:
            task_logprint('Delays > 10ns found! You may want to check data for RFI!') 
            task_logprint('Flagging spws with delay >10ns and re-determining Kcross solutions using MBD. ')
            bad_spw = np.where(d_bool)[0]
            spw_flag_str = str(bad_spw[0])
            for i in range(1,len(bad_spw)):
                spw_flag_str += ','+str(bad_spw[i])
            use_mbd = True
            flagdata(vis=visPola, mode='manual', spw=spw_flag_str)
        
        else:
            task_logprint('No delays > 10ns found!')
        task_logprint('Checking Flagging statistics of SPWs...')
        bad_spw = []
        s = flagdata(vis=visPola, mode='summary')
        for l in range(len(s['spw'])):
            flg_frac = s['spw'][str(l)]['flagged']/s['spw'][str(l)]['total']
            #print(str(l), flg_frac)
            if flg_frac > spw_flg_thresh:
                task_logprint('flagging fraction for spw '+ str(l)+ ' is > %s' % str(spw_flg_thresh*100.0))
                task_logprint('Removing this spw from kcross calculation.')
                bad_spw.append(str(l))
        if bad_spw !=[]:
            spw_flag_str = str(bad_spw[0])
            for l in range(1,len(bad_spw)):
                spw_flag_str += ','+str(bad_spw[l])
            #print('spw_flag_str = ', spw_flag_str)
            flagdata(vis=visPola, mode='manual', spw=spw_flag_str)
            use_mbd = True
        #print('use_mbd = ', use_mbd)
        if use_mbd == True:
            # Solving for the Cross Hand Multiband Delays
            # FIXME: There is probably a better way to handle this...
            # first attempt assumes setup spws are still included in MS and science
            # spw indexing does not start at 0
            # second attemp assumes the science spw indexing starts at 0
            for k in range(len(SPW_per_bb)):
                spw_in_bb = SPW_per_bb[k]
                kcross_mbd = polAngleField+'_'+band+'_band_data.mbd.Kcross'
                if k==0:
                    apen_mode = False
                else:
                    apen_mode = True
                try:
                    gaincal(vis=visPola, caltable=kcross_mbd, field=polAngleField,
                            spw=str(spw_in_bb[0])+'~'+str(spw_in_bb[-1])+':'+str(lower)+'~'+str(upper),
                            gaintype='KCROSS',
                            solint='inf',
                            combine='scan,spw', 
                            refant=refAnt,
                            gaintable=[''],
                            gainfield=[''],
                            parang=True, 
                            append=apen_mode)
                except:
                    #str2prnt = str(int(spw_in_bb[0])-int(spw_in_bb[0]))+'~'+str(int(spw_in_bb[-1])-int(spw_in_bb[0]))+':'+str(lower)+'~'+str(upper)
                    #print(str2prnt)
                    gaincal(vis=visPola, caltable=kcross_mbd, field=polAngleField,
                        spw=str(int(spw_in_bb[0])-int(spw_in_bb[0]))+'~'+str(int(spw_in_bb[-1])-int(spw_in_bb[0]))+':'+str(lower)+'~'+str(upper),
                        gaintype='KCROSS',
                        solint='inf',
                        combine='scan,spw', 
                        refant=refAnt,
                        gaintable=[''],
                        gainfield=[''],
                        parang=True, 
                        append=apen_mode)                   

            plotms(vis=kcross_mbd, xaxis='frequency', yaxis='delay', antenna = refAnt, coloraxis='corr', plotfile=str(polAngleField)+'_delayvsfreq_kcross_mbd_sol.png', overwrite=True)
            flagmanager(vis=visPola, mode='restore', versionname='pre_kcross_mbd_flagging')
            kcross_spw_map = list(kcross_spw_map_temp)
            kcross_tab = kcross_mbd
        else:
            kcross_spw_map = []
            kcross_tab = kcross_sbd
            
        polLeakFields = polLeakField.rsplit(',')
        print('polLeakField = ', polLeakFields, polLeakFields[0])
        for pl in range(len(leak_polcal_type)):
            task_logprint('Setting model for '+polLeakFields[pl])
            file_spix = np.genfromtxt(visPola.rstrip('_pola_cal.ms/')+'.spix.field'+polLeakFields[pl]+'.band'+band+'.txt', dtype=str)
            file_spix_ndarray = str(file_spix).rsplit(',')
            print('file_spix = ', file_spix_ndarray)
            spix_str_a = file_spix_ndarray[1]
            spix_str_b = file_spix_ndarray[2]
            
            spix_flt_a = float(spix_str_a)
            spix_flt_b = float(spix_str_b)
            
            #spix_flt_a = 0.1707699344315621
            #spix_flt_b = -0.1668844662580119
            #task_logprint('input spectral index = ', spix_flt_a, spix_flt_b)
            file_fd = np.genfromtxt(visPola.rstrip('_pola_cal.ms/')+'.fluxdensities.field'+polLeakFields[pl]+'.band'+band+'.txt', dtype=str)
            Table = []
            Freqs = []
            StokesI = []
            for i in range(len(file_fd)):
                row = file_fd[i].rsplit(',')
                Freqs.append(float(row[0]))
                StokesI.append(float(row[1]))

            Freqs = np.array(Freqs)
            reffreq = refFreqI/(1e9)
            freqs_i = np.where(abs(Freqs-reffreq)==min(abs(Freqs-reffreq)))[0][0]

            task_logprint('reference frequency for setjy() = %s' % Freqs[freqs_i])
            task_logprint('Flux density = %s ' % StokesI[freqs_i])
            popt, pcov = sp.optimize.curve_fit(S, Freqs, StokesI)
            task_logprint('input spectral index = %s ' % str(popt[1])+' , '+str(popt[2]))
            setjy(vis=visPola, standard='manual', field=polLeakFields[pl], 
                                spw=str(spw_start)+'~'+str(spw_end), scalebychan=True, listmodels=False,
                                fluxdensity=[StokesI[freqs_i], 0, 0, 0],
                                spix = [popt[1], popt[2]],
                                reffreq=str(Freqs[freqs_i]*(1e9))+'Hz',
                                usescratch=True,useephemdir=False,interpolation="nearest", ismms=False) 
            
            plotms(vis=visPola,field=polLeakFields[pl],correlation='RR',
            timerange='',antenna='ea01&ea02',
            xaxis='frequency',yaxis='amp',ydatacolumn='model', plotfile='polLeak'+polLeakFields[pl]+'_aftersetjy_visPola.png', overwrite=True)
            
            #print('len(bandSPW[b]) = ',len(bandSPW[b]))
            #print('leak_polcal_type[pl] = ', leak_polcal_type[pl])
            if leak_polcal_type[pl] == 'DfQU':
                use_parang = True
            else:
                use_parang = False

            if use_parang == True:
                task_logprint('Using poltype = DfQU')
                dtab = polLeakFields[pl]+'_'+band+'_band_data.DfQU'
                polcal(vis = visPola, caltable = dtab, field=polLeakFields[pl], 
                       spw = str(spw_start)+'~'+str(spw_end)  , refant=refAnt, poltype='Df+QU', solint='inf,2MHz',
                       combine='scan', gaintable = [kcross_tab], gainfield=[''], spwmap=kcross_spw_map, append=False) #
                    
            else:
                task_logprint('Using poltype = Df')
                dtab = polLeakFields[pl]+'_'+band+'_band_data.Df'
                polcal(vis = visPola, caltable = dtab, field=polLeakFields[pl],
                    spw = str(spw_start)+'~'+str(spw_end)  , refant=refAnt, poltype='Df', solint='inf,2MHz',
                    combine='scan', gaintable = [kcross_tab], gainfield=[''], spwmap=kcross_spw_map, append=False) #len(bandSPW[b])*[spw_start]]

            plotms(vis = dtab, xaxis='freq', yaxis='amp', coloraxis='corr', antenna='ea01', plotfile=polLeakFields[pl]+'_'+band+'_band.ampvsfreq.'+leak_polcal_type[pl]+'solns.png', overwrite=True)

            plotms(vis = dtab, xaxis='chan', yaxis='phase', coloraxis='corr', antenna='ea01', plotrange=[-1,-1,-180,180], plotfile=polLeakFields[pl]+'_'+band+'_band.phasevschan.'+leak_polcal_type[pl]+'solns.png', overwrite=True)


        xtab = polAngleField+"_"+band+"_band_data.Xf"
        polcal(vis=visPola,
               caltable=xtab,
               spw=str(spw_start)+'~'+str(spw_end),
               field=polAngleField,
               solint='inf,2MHz',
               combine='scan',
               poltype='Xf',
               refant = refAnt,
               gaintable=[kcross_tab,dtab],
               gainfield=['',''],
               spwmap=[kcross_spw_map,[]],
               append=False)
        
        plotms(vis=xtab,xaxis='frequency',yaxis='phase',coloraxis='spw', plotfile=str(polAngleField)+'_phasevsfreq_Xf_sol.png', overwrite=True)
        task_logprint('Applying polarization calibration solution tables to %s' % visPola)
        applycal(vis = visPola,
                 field='',
                 gainfield=['', '', ''], 
                 flagbackup=True,
                 interp=['', '', ''],
                 gaintable=[kcross_tab,dtab,xtab],
                 spw=str(spw_start)+'~'+str(spw_end), 
                 calwt=[False, False, False], 
                 applymode='calflagstrict', 
                 antenna='*&*', 
                 spwmap=[kcross_spw_map,[],[]], 
                 parang=True)

        
        
        plotms(vis=visPola,field=polAngleField,correlation='',
               timerange='',antenna='',avgtime='60',
               xaxis='frequency',yaxis='amp',ydatacolumn='corrected',
               coloraxis='corr',
               plotfile='plotms_'+str(polAngleField)+'-corrected-amp-vs-frequency.png')

        plotms(vis=visPola,field=polAngleField,correlation='',
               timerange='',antenna='',avgtime='60',
               xaxis='frequency',yaxis='phase',ydatacolumn='corrected',
               plotrange=[-1,-1,-180,180],coloraxis='corr',
               plotfile='plotms_'+str(polAngleField)+'-corrected-phase-vs-frequency.png')

        for i in range(len(polLeakFields)):
            plotms(vis=visPola,field=polLeakFields[i],correlation='',
                   timerange='',antenna='',avgtime='60',
                   xaxis='frequency',yaxis='amp',ydatacolumn='corrected',
                   plotfile='plotms_'+str(polLeakFields[i])+'-corrected-amp-vs-frequency.png')

            plotms(vis=visPola,field=polLeakFields[i],correlation='RR,LL',
                   timerange='',antenna='',avgtime='60',
                   xaxis='frequency',yaxis='phase',ydatacolumn='corrected',
                   plotrange=[-1,-1,-180,180],coloraxis='corr',
                   plotfile='plotms_'+str(polLeakFields[i])+'-corrected-phase-vs-frequency.png')



        # Image calibrator in full stokes spectral cube at each spw, measure polangle and pol frac
        # across spws and compare with input model:
        # spws_imaging = np.arange(spw_start, spw_end)
        # im_name = polAngleField+'_StokesCubeIm'

    rms_im = 15e-6

    imniter=20000
    ns_thresh = 4.0
    sidelobethresh = [5.5]
    mnbmfrc = [0.3]
    lwnsthresh = [1.5]

    ### Find max baseline length
    # Load your measurement set
    try:
        tb.open(visPola+'ANTENNA')

        # Get antenna positions
        positions = tb.getcol('POSITION')
    finally:
        tb.close()

    # Calculate distances between all pairs of antennas
    num_antennas = positions.shape[1]
    max_baseline = 0.0

    #determine max baseline length
    for i in range(num_antennas):
        for j in range(i + 1, num_antennas):
            # Calculate the baseline length
            baseline_length = np.linalg.norm(positions[:, i] - positions[:, j])
            if baseline_length > max_baseline:
                max_baseline = baseline_length

    print(f'Maximum Baseline Length: {max_baseline} meters')

    
    ## Determine lowest angular resolution from lowest frequency spw
    try:
        tb.open(visPola+'SPECTRAL_WINDOW/')
        ref_frequencies = tb.getcol('REF_FREQUENCY')
    finally:
        tb.close()
        
    freq_diff = np.diff(ref_frequencies) #will be used later for sensitivity calcualtion per spw
    print('freq_diff = ', freq_diff)
    ref_frequencies_GHz = ref_frequencies/(1e9)
    print(ref_frequencies_GHz)
    spws_imaging = np.arange(spw_start, spw_end+1)
    #print(spws_imaging)
    speed_light = 3e8 #m/s
    ref_wavelengths = speed_light/ref_frequencies
    #print('ref_wavelengths = ', ref_wavelengths)

    low_nu_synth_beam = (ref_wavelengths[0]/max_baseline)*(180.0/np.pi)*(60.0*60.0)
    print('synthesized beam = ', low_nu_synth_beam, ' arcsec')


    # Determine PB FWHM from lowest frquency spw
    PB_fwhm = 42/ref_frequencies_GHz[0] #estimate of PB FWHM for VLA (arcmin)
    PB_fwhm_arcsec = PB_fwhm*60
    print('PB FWHM = ', PB_fwhm_arcsec, 'arcsec')

    # Determine cell size (4 cells per synthesized beam)
    cell_size = low_nu_synth_beam/4.0
    print('cell size = ', cell_size, 'arcsec')

    # Determine imsize from PB_fwhm/cell_size, can try 1.5* PB size?
    im_size_temp = 1.5*PB_fwhm_arcsec/cell_size/2
    print('im_size = ', im_size_temp/2)

    # From imsizes reccomended for efficient computation, select one closest to the computed imsize 
    NUMS=[]
    for i in range(0,8):
        for j in range(0,8):
            NUMS.append(5*pow(2,i)*pow(3,j))

    diff_arr = abs(NUMS-im_size_temp)
    im_size = NUMS[np.where(diff_arr==min(diff_arr))[0][0]]


    # Estimate the expected sensitivity from the VLA Observational Status summary page on sensitivity. Use half of time on source for upper limit on sensitivity
    try:
        tb.open(visPola)
        subtable = tb.query('FIELD_ID=='+flux_field_select_string)
        time_on_source = subtable.getcol('TIME')
        subtable.close()
    finally:
        tb.close()
        
    time_diffs = np.diff(time_on_source)
    # print(0.5*np.sum(time_diffs))
    int_time = 0.5*np.sum(time_diffs)

    bandwidth = freq_diff[0]
    print('bandwidth = ', bandwidth , ' Hz')

    ### FIXME: Could make the sensitivity estimate more accurate with a more accurate SEFD and
    ###        determining the true number of antennas used. Assumes all antennas used for now and
    ###        takes upper limits on SEFD from plots from VLA Observational Status summary page on sensitivity
    if refFreqI < 12e9: #if Xband or below, use low-frequency SEFD upper limit
        SEFD = 400
    elif refFreqI>12e9: #if above Xband, use high-frequency SEFD upper limit
        SEFD=1000
    sensitivity = SEFD/(0.93*np.sqrt(2*27*26*int_time*bandwidth))
    print('sensitivity = ', sensitivity, ' Jy/beam')

    spws_imaging = np.arange(spw_start, spw_end)
    
    for i in range(len(spws_imaging)):
        im_name = polAngleField+'_StokesCubeIm_spw'+str(spws_imaging[i])
        spw_list=3*[i]
        chan_list = [0,32,64]
        #list(32*np.ones_like(spw_list))
        weight_list = list(1.0*np.ones_like(spw_list))
        #print(spw_list)
        #print(im_name)
        if i == 0:
            tclean(vis=visPola, selectdata=True, field=polAngleField, datacolumn='corrected', imagename=im_name, spw=str(i), imsize=im_size, cell=str(cell_size)+'arcsec', deconvolver='mtmfs', scales=[0], nterms=2, niter=imniter, weighting='briggs', robust=0.0, gridder='standard', pblimit=-1e-6, stokes='IQUV', threshold=str(sensitivity)+'Jy', interactive=False)
            widebandpbcor(vis=visPola, imagename=im_name, nterms=2, threshold=str(rms_im)+'Jy', action='pbcor', spwlist=spw_list, chanlist=chan_list, weightlist=weight_list)
            #extract restoring beam info from 4.5 GHz image
            head_info = imhead(im_name+'.image.tt0')
            bmj = head_info['perplanebeams']['beams']['*0']['*0']['major']['value']
            bmn = head_info['perplanebeams']['beams']['*0']['*0']['minor']['value']
            bposang = head_info['perplanebeams']['beams']['*0']['*0']['positionangle']['value']

            print('restoring beam info: ', bmj, bmn, bposang)
        else:
            tclean(vis=visPola, selectdata=True, field=polAngleField, datacolumn='corrected', imagename=im_name, spw=str(i), imsize=im_size, cell=str(cell_size)+'arcsec', deconvolver='mtmfs', scales=[0], nterms=2, niter=imniter, weighting='briggs', robust=0.0, gridder='standard', pblimit=-1e-6, stokes='IQUV', threshold=str(sensitivity)+'Jy', interactive=False, restoringbeam=[str(bmj)+'arcsec', str(bmn)+'arcsec', str(bposang)+'deg'])
            widebandpbcor(vis=visPola, imagename=im_name, nterms=2, threshold=str(rms_im)+'Jy', action='pbcor', spwlist=spw_list, chanlist=chan_list, weightlist=weight_list)

    PF = []
    PA = []
    for i in range(len(spws_imaging)):
        im_name = polAngleField+'_StokesCubeIm_spw'+str(spws_imaging[i])
        #immath(outfile=im_name+'.pbcor.poli',mode='poli',imagename=[im_name+'.pbcor.image.tt0'],sigma='0.0Jy/beam')
        # Obtain image for the polarization angle
        #immath(outfile=im_name+'.pbcor.pola',mode='pola',imagename=[im_name+'.pbcor.image.tt0'],sigma='0.0Jy/beam')

        imsubimage(imagename=im_name+'.image.tt0',outfile=im_name+'.I.image',stokes='I')
        imsubimage(imagename=im_name+'.image.tt0',outfile=im_name+'.Q.image',stokes='Q')
        imsubimage(imagename=im_name+'.image.tt0',outfile=im_name+'.U.image',stokes='U')
        
        ia.open(im_name+'.I.image')
        II = ia.getchunk()
        ia.close()
        
        ia.open(im_name+'.Q.image')
        QQ = ia.getchunk()
        ia.close()

        ia.open(im_name+'.U.image')
        UU = ia.getchunk()
        ia.close()
        
        y, x = int(II.shape[0]/2), int(II.shape[1]/2)
        box_size = 10
        sub_II = II[y-box_size:y+box_size,x-box_size:x+box_size,0,0].sum()/(II[y-box_size:y+box_size,x-box_size:x+box_size,0,0].shape[0]*II[y-box_size:y+box_size,x-box_size:x+box_size,0,0].shape[1])
        sub_QQ = QQ[y-box_size:y+box_size,x-box_size:x+box_size,0,0].sum()/(QQ[y-box_size:y+box_size,x-box_size:x+box_size,0,0].shape[0]*QQ[y-box_size:y+box_size,x-box_size:x+box_size,0,0].shape[1])
        sub_UU = UU[y-box_size:y+box_size,x-box_size:x+box_size,0,0].sum()/(UU[y-box_size:y+box_size,x-box_size:x+box_size,0,0].shape[0]*UU[y-box_size:y+box_size,x-box_size:x+box_size,0,0].shape[1])
        
        polfrac = np.sqrt(np.array(sub_UU)*np.array(sub_UU)+np.array(sub_QQ)*np.array(sub_QQ))/sub_II
        print('polfrac = ', polfrac)
        PF.append(polfrac)

        chi_ = 0.5*np.arctan2(np.array(sub_UU),np.array(sub_QQ))
        # chi_ = sub_PA
        print('chi_ = ', chi_)
        PA.append(chi_)


        # ia.open(im_name+'.pbcor.poli')
        # PI = ia.getchunk()
        # ia.close
        # y, x = int(PI.shape[0]/2), int(PI.shape[1]/2)
        # sub_PI = PI[y,x]

        # ia.open(im_name+'.I.image')
        # II = ia.getchunk()
        # ia.close
        # sub_II = II[y,x,0,0]

        # polfrac = sub_PI/sub_II

        # PF.append(polfrac)

        # ia.open(im_name+'.pbcor.pola')
        # PA = ia.getchunk()
        # ia.close
        # # y, x = int(PA.shape[0]/2), int(PI.shape[1]/2)
        # sub_PA = PA[y,x]

        # # chi_ = 0.5*np.arctan2(np.array(sub_UU),np.array(sub_QQ))
        # chi_ = sub_PA
        # #print('chi_ = ', chi_)
        # PA.append(chi_)
        

    # cal_data2013 = np.genfromtxt('/lustre/aoc/students/ksanders/EVLA_SCRIPTED_PIPELINE/EVLA_SP_postfork/evla_scripted_pipeline_postfork/data/PolCals_2013_3C48.3C138.3C147.3C286.dat')

    # freqFitting = cal_data2013[:,0]
    pol_perc = model_pol/100
    # cal_data2013[:,1]
    pol_angle = model_ang
    # cal_data2013[:,2]
    
    flip_factor = 0.0
    if model_flip_plus == True:
        flip_factor = 180.0
    elif model_flip_minus == True:
        flip_factor = -180.0


    interp_func_pf = interp1d(freqFitting, pol_perc/100.0)

    newarr_pf = interp_func_pf(ref_frequencies_GHz[:-1])

    plt.figure()
    plt.title('Overall Title')
    plt.title('Polarization Fraction: Central 100 pixels:')
    plt.xlabel(r'$\nu$ (GHz)')
    plt.ylabel(r'PF')
    plt.scatter(ref_frequencies_GHz[:-1],PF,color='r', label='measured')
    plt.scatter(ref_frequencies_GHz[:-1], newarr_pf, color='b', label='model')
    plt.legend()
    plt.savefig(polAngleField+'_PolFrac_central100pixels.png', bbox_inches="tight", dpi=250)
    plt.show()

    plt.figure()
    plt.title('Overall Title')
    #plt.subplot(221)
    plt.title('PF Residuals: Central 100 pixels:')
    plt.xlabel(r'$\nu$ (GHz)')
    plt.ylabel(r'data-model')
    plt.scatter(ref_frequencies_GHz[:-1],PF-newarr_pf,color='r')
    plt.legend()
    plt.savefig(polAngleField+'_PolFracResiduals_central100pixels.png', bbox_inches="tight", dpi=250)
    plt.show()
        

    interp_func_pa = interp1d(freqFitting, pol_angle)

    newarr_pa = interp_func_pa(ref_frequencies_GHz[:-1])

    plt.figure()
    plt.title('Overall Title')
    #plt.subplot(221)
    plt.title('Polarization Angle: Central 100 pixels:')
    plt.xlabel(r'$\nu$ (GHz)')
    plt.ylabel(r'PA')
    plt.scatter(ref_frequencies_GHz[:-1],np.rad2deg(np.array(PA))+flip_factor,color='r', label='measured')
    plt.scatter(ref_frequencies_GHz[:-1], newarr_pa, color='b', label = 'model')
    plt.legend()
    plt.savefig(polAngleField+'_PolAngle_central100pixels.png', bbox_inches="tight", dpi=250)
    plt.show()

    plt.figure()
    plt.title('Overall Title')
    #plt.subplot(221)
    plt.title('PA Residuals: Central 100 pixels:')
    plt.xlabel(r'$\nu$ (GHz)')
    plt.ylabel(r'data-model')
    plt.scatter(ref_frequencies_GHz[:-1],(np.rad2deg(np.array(PA))+flip_factor)-newarr_pa,color='r')
    plt.legend()
    plt.savefig(polAngleField+'_PolAngleResiduals_central100pixels.png', bbox_inches="tight", dpi=250)
    plt.show()

        
    task_logprint(f"QA2 score: {QA2_polcal}")
    task_logprint("Finished EVLA_pipe_polcal_testing_v2.py")
    time_list = runtiming("polcal", "end")

    pipeline_save()
