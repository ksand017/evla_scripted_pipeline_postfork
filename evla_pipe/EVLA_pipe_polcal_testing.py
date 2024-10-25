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

from casatasks import gaincal, applycal, polcal, setjy, split, fluxscale, casalog, flagdata, flagmanager
from . import pipeline_save
from .utils import runtiming, logprint, find_standards, find_EVLA_band, RefAntHeuristics, MAINLOG

pi = np.pi

def task_logprint(msg):
    logprint(msg, logfileout="logs/polcal.log")

def S(f,S,alpha,beta):
    '''
    Power law model given input alpha, beta, and frequency array
    '''
    return S*(f/3.0)**(alpha+beta*np.log10(f/3.0))

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
    cal_data2013 = np.genfromtxt('/lustre/aoc/students/ksanders/EVLA_SCRIPTED_PIPELINE/EVLA_Scripted_Pipeline/EVLA_SCRIPTED_PIPELINE/data/PolCals_2013_3C48.3C138.3C147.3C286.dat')

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
    popt_pf, pcov_pf = sp.optimize.curve_fit(fitter3, x_data, pol/100.)
    popt_pa, pcov_pa = sp.optimize.curve_fit(fitter3, x_data, ang*(np.pi/180))	

    # refFreq - refFreq = 0
    p_ref = np.polyval(popt_pf[::-1], 0.0)

    #print([popt, p_ref, RM, X_0])
    return popt_pf, popt_pa, p_ref
'''
START OF POLCAL SCRIPT
'''
task_logprint("*** Starting EVLA_pipe_polcal_testing_v2.py ***")
time_list = runtiming("polcal", "start")
QA2_polcal = "Pass"

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
    #refAnt = 'ea10' #casaguide chosen refant
    refAnt = str(RefAntOutput[0])

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
        tb.open(band+'band.ms/')
        unique_scan_nums = np.unique(tb.getcol('SCAN_NUMBER'))
        tb.close()
        if len(unique_scan_nums) <= 3:
            task_logprint('This band does not have enough scans to calibrate!')
            continue

        
        
        ###FIXME: add step to determine the spws perbaseband within a band
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
        tb.close()
        print('SPW_per_bb = ', SPW_per_bb)
        
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
        coeffs_pf, coeffs_pa, p_ref = polyFit(polAngleField, band, refFreqI/1e+09)
        task_logprint("polyFit output:")
        print(coeffs_pf, coeffs_pa, p_ref)
        
        #coeffs = polOut[0]
        task_logprint("polindex input will be: "+str(coeffs_pf))
        task_logprint("polangle input will be: "+str(coeffs_pa))

        #p_ref = polOut[1]
        task_logprint("p_ref is "+str(p_ref))

        #RM = polOut[2]
        #X_0 = polOut[3] # this is in terms of radians

        # calculate Stokes Q and U
        #q_ref = p_ref*i_ref*np.cos(2*X_0)
        #u_ref = p_ref*i_ref*np.sin(2*X_0)

        #task_logprint("Flux Dict passed to setjy is: "+str(i_ref)+" "+str(q_ref)+" "+str(u_ref))

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
                            usescratch=True)

        
        plotms(vis=visPola,field=polAngleField,correlation='RL',
               timerange='',antenna='',
               xaxis='frequency',yaxis='amp',ydatacolumn='model', plotfile=str(polAngleField)+'_ampvsfreq_RL_polarizedmodel.png', overwrite=True)

        plotms(vis=visPola,field=polAngleField,correlation='RL',
               timerange='',antenna='',
               xaxis='frequency',yaxis='phase',ydatacolumn='model', plotfile=str(polAngleField)+'_phasevsfreq_RL_polarizedmodel.png', overwrite=True)

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
        
        tb.open(kcross_sbd)
        delays = np.unique(tb.getcol('FPARAM'))
        #delays = np.array([0.0, 2.0, 3.5, 5.5, 7.0, 10.3, 12.0])
        tb.close()
        print('delays = ', delays)
        d_bool = delays > 10.0
        d_bool_check = True in d_bool
        use_mbd = True
        if d_bool_check == True:
            print('Delays > 10ns found! You may want to check data for RFI!') 
            print('Flagging spws with delay >10ns and determining Kcross solutions using MBD. ')
            flagmanager(vis = visPola, mode='save', versionname='pre_kcross_mbd_flagging')
            bad_spw = np.where(d_bool)[0]
            spw_flag_str = str(bad_spw[0])
            for i in range(1,len(bad_spw)):
                spw_flag_str += ','+str(bad_spw[i])
            use_mbd = True
            flagdata(vis=visPola, mode='manual', spw=spw_flag_str)
        
        else:
            use_mbd = True
            print('No delays > 10ns found!')
        if use_mbd == True:
            # Solving for the Cross Hand Multiband Delays
            for k in range(len(SPW_per_bb)):
                spw_in_bb = SPW_per_bb[k]
                kcross_mbd = polAngleField+'_'+band+'_band_data.mbd.Kcross'
                if k==0:
                    apen_mode = False
                else:
                    apen_mode = True
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

            plotms(vis=kcross_mbd, xaxis='frequency', yaxis='delay', antenna = refAnt, coloraxis='corr', plotfile=str(polAngleField)+'_delayvsfreq_kcross_mbd_sol.png', overwrite=True)
            flagmanager(vis=visPola, mode='restore', versionname='pre_kcross_mbd_flagging')
            kcross_tab = kcross_mbd
            # kcross_spw_map = 
        else:
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
            print('input spectral index = ', spix_flt_a, spix_flt_b)
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

            print('reference frequency for setjy() = ', Freqs[freqs_i])
            print('Flux density = ', StokesI[freqs_i])
            popt, pcov = sp.optimize.curve_fit(S, Freqs, StokesI)
            setjy(vis=visPola, standard='manual', field=polLeakFields[pl], 
                                spw=str(spw_start)+'~'+str(spw_end), scalebychan=True, listmodels=False,
                                fluxdensity=[StokesI[freqs_i], 0, 0, 0],
                                spix = [popt[1], popt[2]],
                                reffreq=str(Freqs[freqs_i]*(1e9))+'Hz',
                                usescratch=True,useephemdir=False,interpolation="nearest", ismms=False) #0.17077,-0.166884  spix_flt_a, spix_flt_b
            
            plotms(vis=visPola,field=polLeakFields[pl],correlation='RR',
            timerange='',antenna='ea01&ea02',
            xaxis='frequency',yaxis='amp',ydatacolumn='model', plotfile='polLeak'+polLeakFields[pl]+'_aftersetjy_visPola.png', overwrite=True)
            
            print('len(bandSPW[b]) = ',len(bandSPW[b]))
            print('leak_polcal_type[pl] = ', leak_polcal_type[pl])
            if leak_polcal_type[pl] == 'DfQU':
                use_parang = True
            else:
                use_parang = False

            if use_parang == True:
                task_logprint('Using poltype = DfQU')
                dtab = polLeakFields[pl]+'_'+band+'_band_data.DfQU'
                polcal(vis = visPola, caltable = dtab, field=polLeakFields[pl], 
                       spw = str(spw_start)+'~'+str(spw_end)  , refant=refAnt, poltype='Df+QU', solint='inf,2MHz',
                       combine='scan', gaintable = [kcross_tab], gainfield=[''], spwmap=[len(bandSPW[b])*[spw_start]], append=False) #
                    
                #len(bandSPW[b])*[2]
            else:
                task_logprint('Using poltype = Df')
                dtab = polLeakFields[pl]+'_'+band+'_band_data.Df'
                polcal(vis = visPola, caltable = dtab, field=polLeakFields[pl],
                    spw = str(spw_start)+'~'+str(spw_end)  , refant=refAnt, poltype='Df', solint='inf,2MHz',
                    combine='scan', gaintable = [kcross_tab], gainfield=[''], spwmap=[len(bandSPW[b])*[spw_start]], append=False)

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
               gaintable=[kcross_mbd,dtab],
               gainfield=['',''],
               spwmap=[len(bandSPW[b])*[spw_start],[]],
               append=False)
        
        plotms(vis=xtab,xaxis='frequency',yaxis='phase',coloraxis='spw', plotfile=str(polAngleField)+'_phasevsfreq_Xf_sol.png', overwrite=True)
        '''
        applycal(vis = visPola,
                 field='',
                 gainfield=['', '', ''], 
                 flagbackup=True,
                 interp=['', '', ''],
                 gaintable=[kcross_mbd,dtab,xtab],
                 spw=str(spw_start)+'~'+str(spw_end), 
                 calwt=[False, False, False], 
                 applymode='calflagstrict', 
                 antenna='*&*', 
                 spwmap=[[0,0,0,0,0,0,0,0],[],[]], 
                 parang=True)

        
        
        plotms(vis=visPola,field=polAngleField,correlation='',
               timerange='',antenna='',avgtime='60',
               xaxis='frequency',yaxis='amp',ydatacolumn='corrected',
               coloraxis='corr',
               plotfile='plotms_'+str(polAngleField)+'-corrected-amp.png')

        plotms(vis=visPola,field=polAngleField,correlation='',
               timerange='',antenna='',avgtime='60',
               xaxis='frequency',yaxis='phase',ydatacolumn='corrected',
               plotrange=[-1,-1,-180,180],coloraxis='corr',
               plotfile='plotms_'+str(polAngleField)+'-corrected-phase.png')

        for i in range(len(polLeakFields)):
            plotms(vis=visPola,field=polLeakFields[i],correlation='',
                   timerange='',antenna='',avgtime='60',
                   xaxis='frequency',yaxis='amp',ydatacolumn='corrected',
                   plotfile='plotms_'+str(polLeakFields[i])+'-corrected-amp.png')

            plotms(vis=visPola,field=polLeakFields[i],correlation='RR,LL',
                   timerange='',antenna='',avgtime='60',
                   xaxis='frequency',yaxis='phase',ydatacolumn='corrected',
                   plotrange=[-1,-1,-180,180],coloraxis='corr',
                   plotfile='plotms_'+str(polLeakFields[i])+'-corrected-phase.png')


        '''
    task_logprint(f"QA2 score: {QA2_polcal}")
    task_logprint("Finished EVLA_pipe_polcal_testing_v2.py")
    time_list = runtiming("polcal", "end")

    pipeline_save()
