import numpy as np
import pypulseq as pp
from pypulseq.opts import Opts
from pSeq_Base import pSeq_Base 
import copy


class pSeq_EPI_base(pSeq_Base):
    def __init__(self, pseq=None, 
                fov = 192e-3,
                nx = 140,
                ny = 140,
                readoutTime = 5e-4,
                RSegment = 1,
                partFourierFactor = 0.75,
                R=2,
                Nmulti  = 1,
                ghost_lines = 3,
                *args, **kwargs,
                
                 ):
        
        super().__init__(pseq, *args, **kwargs)

        self.fov = fov
        self.Nx = nx
        self.Ny = ny
        self.readoutTime = readoutTime
        self.RSegment = RSegment
        self.partFourierFactor = partFourierFactor
        self.R = R
        self.Nmulti = Nmulti
        self.ghost_lines = ghost_lines
        self.ghost_flag = False

        if pseq:
            self.epi_system = pseq.epi_system
        else: 
            self.epi_system = self.system

        #if verbose:
        #   print('EPI: Echo Spacing is ',ESP,'ms')
        #   print('EPI: ADC Bandwidth = ',1/adcDwell/1000,'kHz')
        #   print('Actual RO oversampling factor is  ', deltak/gx.amplitude/adcDwell)
        #   print('EPI: ADC Samples are ',adcS)   

       

    def prep(self,ro_os = 1,pe_enable = 1, ref_polarity = False):

        self.readoutBW = 1/self.readoutTime

        deltak = 1/self.fov
        deltaky=self.RSegment*self.R*deltak
        kWidth = self.Nx*deltak

        blip_dur = np.ceil(2 * np.sqrt(deltaky / self.epi_system.max_slew) / 10e-6 / 2) * 10e-6 * 2  # we round-up the duration to 2x the gradient raster time
        gy = pp.make_trapezoid(channel = 'y',system = self.epi_system, area =-deltaky,duration = blip_dur) # we use negative blips to save one k-space line on our way towards the k-space center

        # Readout gradient is a truncated trapezoid with dead times at the beginning
        # and at the end each equal to a half of blip_dur
        # The area between the blips should be defined by kWidth
        # We do a two-step calculation: we first increase the area assuming maximum
        # slewrate and then scale down the amplitude to fix the area

        extra_area = blip_dur / 2 * blip_dur / 2 * self.epi_system.max_slew  # Check unit!
        gx = pp.make_trapezoid(channel = 'x', system = self.epi_system, area=kWidth + extra_area, duration=self.readoutTime + blip_dur)
        actual_area = gx.area - gx.amplitude / gx.rise_time * blip_dur / 2 * blip_dur / 2 / 2 - gx.amplitude / gx.fall_time * blip_dur / 2 * blip_dur / 2 / 2
        gx.amplitude = gx.amplitude / actual_area * kWidth
        gx.area = gx.amplitude * (gx.flat_time + gx.rise_time / 2 + gx.fall_time / 2)
        gx.flat_area = gx.amplitude * gx.flat_time


        #assert(ro_os>=2) ;
        #adcSamples=Nx*ro_os ;
        #adcDwell=floor(readoutTime/adcSamples*1e7)*1e-7;
        # Calculate ADC
        adc_dwell_nyquist = deltak / gx.amplitude / ro_os
        #adc_dwell_nyquist = 2e-06  # This serves as oversampling, but our primary aim is to match GE's ADC
        # Round-down dwell time to 100 ns
        adc_dwell = np.floor(adc_dwell_nyquist * 1e7) * 1e-7


        # Calculate ADC samples
        adc_samples = np.floor(self.readoutTime / adc_dwell / 4) * 4  # On Siemens, the number of ADC samples needs to be divisible by 4
        #print("# ADC Samples are", adc_samples)
        # MZ: no idea, whether ceil, round or floor is better for the adc_samples...
        adc = pp.make_adc(adc_samples, dwell=adc_dwell, delay=blip_dur / 2)
        self.adc_samples = adc_samples

        # Realign the ADC with respect to the gradient
        time_to_center = adc.dwell * ((adc_samples - 1) / 2 + 0.5)  # I've been told that Siemens samples in the center of the dwell period
        adc.delay = np.round((gx.rise_time + gx.flat_time / 2 - time_to_center) * 1e6) * 1e-6  # We adjust the delay to align the trajectory with the gradient. We have to align the delay to 1us

        # Split the blip into two halves and produce a combined synthetic gradient
        gy_parts = pp.split_gradient_at(gy, blip_dur / 2, self.epi_system)
        gy_blipup, gy_blipdown, _ = pp.align(right=gy_parts[0], left = [gy_parts[1], gx])
        gy_blipdownup = pp.add_gradients([gy_blipdown, gy_blipup], system =self.epi_system)


        # pe_enable support
        gy_blipup.waveform *= pe_enable
        gy_blipdown.waveform *= pe_enable
        gy_blipdownup.waveform *= pe_enable



        # Phase encoding and partial Fourier
        Ny_pre = round((self.partFourierFactor - 1 / 2) * self.Ny - 1)  # PE steps prior to ky=0, excluding the central line
        Ny_pre = round(Ny_pre / self.RSegment / self.R)
        Ny_post = round(self.Ny / 2 + 1)  # PE lines after the k-space center including the central line
        Ny_post = round(Ny_post / self.RSegment / self.R)
        Ny_meas = Ny_pre + Ny_post
        #print(Ny_pre, Ny_post, Ny_meas)

        # Pre-phasing gradients
        gx_pre = pp.make_trapezoid(channel = 'x', system = self.epi_system, area=-gx.area / 2) # (Ny_pre*deltaky-(Nmulti-1)*R*deltak)
        gy_pre = pp.make_trapezoid(channel = 'y', system = self.epi_system, area=(Ny_pre * deltaky - (self.Nmulti - 1)  * deltak* self.R))

        gx_pre, gy_pre = pp.align(right = gx_pre, left= gy_pre)

        # Relax the PE prephaser to reduce stimulation
        gy_pre = pp.make_trapezoid(channel = 'y', system = self.epi_system, area=gy_pre.area, duration=pp.calc_duration(gx_pre, gy_pre))
        gy_pre.amplitude *= pe_enable

        # For after the first echo QL
        gx_pre_post = pp.make_trapezoid(channel = 'x', system = self.epi_system, area=-gx.area / 2)
        gy_pre_post = pp.make_trapezoid(channel = 'y', system = self.epi_system, area=(Ny_post - 1) * deltaky)

        gx_pre_post, gy_pre_post = pp.align(right=gx_pre_post, left=gy_pre_post)

        # Relax the PE prephaser to reduce stimulation
        gy_pre_post = pp.make_trapezoid(channel = 'y', system = self.epi_system, area=gy_pre_post.area, duration=pp.calc_duration(gx_pre_post, gy_pre_post))
        gy_pre_post.amplitude *= pe_enable

        if Ny_meas % 2 == 0:
            gx_pre_post.amplitude = -gx_pre_post.amplitude

        if ref_polarity:
            gy_pre = pp.scale_grad(gy_pre, -1)
            gy_blipdownup = pp.scale_grad(gy_blipdownup, -1)
            gy_blipdown = pp.scale_grad(gy_blipdown, -1)
            gy_blipup = pp.scale_grad(gy_blipup, -1)
            gy_pre_post = pp.scale_grad(gy_pre_post, -1)     

        if pe_enable:
            gy_blipup.last=0
            gy_blipdown.first=0
            gy_blipdownup.last=0
            gy_blipdownup.first=0

        ESP = 1e3 * pp.calc_duration(gx)  # ec
        print('EPI: Echo Spacing is ',ESP,'ms')
        print('EPI: ADC Bandwidth = ',1/adc_dwell/1000,'kHz')
        print('Actual RO oversampling factor is  ', deltak/gx.amplitude/adc_dwell)
        print('EPI: ADC Samples are ',adc_samples)  
            
            
        gy_pre_post.area *= pe_enable


        gx_pre.delay = 0
        gy_pre.delay = 0

        self.ESP = 1e3 * pp.calc_duration(gx)  # echo spacing, ms
        self.Ny_pre = Ny_pre
        self.gxPre = gx_pre
        self.gyPre = gy_pre
        self.gx = gx

        self.gx_pre_post = gx_pre_post
        self.gy_pre_post = gy_pre_post

        self.gy_blipup = gy_blipup
        self.gy_blipdown = gy_blipdown
        self.gy_blipdownup = gy_blipdownup
        self.Ny_meas = Ny_meas
        self.adc = adc
        self.adc_samples = adc_samples

        self.ROpolarity = np.sign(self.gx.amplitude)
        self.blip_dur = blip_dur
       #print("Starting RO polarity",self.ROpolarity)

    def prep_calib(self,pe_enable = 1,Ny_in = None):
        if Ny_in is None:
            Ny_in = self.Ny/self.R
        self.Ny_calib = Ny_in
        self.readoutBW = 1/self.readoutTime

        deltak = 1/self.fov
        deltaky= deltak
        kWidth = self.Nx*deltak

        blip_dur = self.blip_dur  # use the regular blip duration cause want to match
        gy = pp.make_trapezoid(channel = 'y',system = self.epi_system, area =-deltaky,duration = blip_dur) # we use negative blips to save one k-space line on our way towards the k-space center

        # Split the blip into two halves and produce a combined synthetic gradient
        gy_parts = pp.split_gradient_at(gy, blip_dur / 2, self.epi_system)
        gy_blipup, gy_blipdown, _ = pp.align(right=gy_parts[0], left = [gy_parts[1], self.gx])
        gy_blipdownup = pp.add_gradients([gy_blipdown, gy_blipup], system =self.epi_system)

        # pe_enable support
        gy_blipup.waveform *= pe_enable
        gy_blipdown.waveform *= pe_enable
        gy_blipdownup.waveform *= pe_enable

        # Phase encoding and partial Fourier
        Ny_pre = round((self.partFourierFactor - 1 / 2) * self.Ny_calib - 1)  # PE steps prior to ky=0, excluding the central line
        Ny_pre = round(Ny_pre )
        Ny_post = round(self.Ny_calib / 2 + 1)  # PE lines after the k-space center including the central line
        Ny_post = round(Ny_post)
        Ny_meas = Ny_pre + Ny_post
        #print(Ny_pre, Ny_post, Ny_meas)

        # Pre-phasing gradients
        gx_pre = self.gxPre
        gy_pre = pp.make_trapezoid(channel = 'y', system = self.epi_system, area=(Ny_pre * deltaky - (self.Nmulti - 1)  * deltak))
        gx_pre, gy_pre = pp.align(right = gx_pre, left= gy_pre)

        # Relax the PE prephaser to reduce stimulation
        gy_pre = pp.make_trapezoid(channel = 'y', system = self.epi_system, area=gy_pre.area, duration=pp.calc_duration(gx_pre, gy_pre))
        gy_pre.amplitude *= pe_enable

        # For after the first echo QL
        gy_pre_post = pp.make_trapezoid(channel = 'y', system = self.epi_system, area=(Ny_post - 1) * deltaky)

        gx_pre_post, gy_pre_post = pp.align(right=self.gx_pre_post, left=gy_pre_post)

        # Relax the PE prephaser to reduce stimulation
        gy_pre_post = pp.make_trapezoid(channel = 'y', system = self.epi_system, area=gy_pre_post.area, duration=pp.calc_duration(gx_pre_post, gy_pre_post))
        gy_pre_post.amplitude *= pe_enable


        self.calib_Ny_pre = Ny_pre
        self.calib_gxPre = gx_pre
        self.calib_gyPre = gy_pre

        self.calib_gx_pre_post = gx_pre_post
        self.calib_gy_pre_post = gy_pre_post

        self.calib_gy_blipup = gy_blipup
        self.calib_gy_blipdown = gy_blipdown
        self.calib_gy_blipdownup = gy_blipdownup
        self.calib_Ny_meas = Ny_meas









    def get_echoTime_shift(self):
        # Calculate actual echo spacing
        actual_esp = self.gx.rise_time + self.gx.flat_time + self.gx.fall_time
        TEShift = actual_esp / self.RSegment
        TEShift = round(TEShift, 5)
        TEShift_before_echo = (self.Nmulti - 1) * TEShift

        if TEShift_before_echo == 0:
            TEShift_before_echo = 0.00001  # Apply the minimum duration for the no delay case

        TEShift_after_echo = (self.RSegment - (self.Nmulti - 1)) * TEShift
        dETS_before = pp.make_delay(TEShift_before_echo)
        dETS_after = pp.make_delay(TEShift_after_echo)

        return dETS_before, dETS_after
    
    
    def add_ghost_nav(self,pseq0):
        
        #self.gxPre = pp.scale_grad(self.gxPre, -1)
        #self.gx = pp.scale_grad(self.gx, -1)
        #print('Starting GxPre and Gx Amplitudes ', np.sign(self.gxPre.amplitude),np.sign(self.gx.amplitude ))
        center_line = np.floor(self.Ny/2) 
        #if self.R >1:
        #    center_line-=1
        if self.partFourierFactor<1:
            center_line-= (1-self.partFourierFactor)*self.Ny

   
        pseq0.seq.add_block(self.gxPre,pp.make_label(type = 'SET',label = 'NAV',value = 1),
                pp.make_label(type ='SET',label = 'LIN', value = center_line))
        
        #self.gx = pp.scale_grad(self.gx, -1) 
        
        #print('After Prep GxPre and Gx Amplitudes ', np.sign(self.gxPre.amplitude),np.sign(self.gx.amplitude ))
        self.nav_sign_track = []
        for i in range(1,self.ghost_lines+1):
            resultREV = int(np.sign(self.gx.amplitude) == self.ROpolarity)
            resultSEG = int(np.sign(self.gx.amplitude) == self.ROpolarity)
            resultAVG = int(i == self.ghost_lines)

            self.nav_sign_track.append(resultREV)
            

            pseq0.seq.add_block(pp.make_label(type ='SET',label ='REV', value = resultREV),
                    pp.make_label(type = 'SET',label = 'SEG', value = resultSEG), 
                    pp.make_label(type = 'SET',label ='AVG', value = resultAVG ) ) 

            pseq0.seq.add_block(self.gx,self.adc)
            self.gx.amplitude = -self.gx.amplitude #;   % Reverse polarity of read gradient
            
        # 24.08.08 Need to add rewinder back to center of k-space for epi to work
        pseq0.seq.add_block(self.gxPre,pp.make_label(type = 'SET',label = 'NAV',value = 0))
        
        #print('After Prep GxPre and Gx Amplitudes ', np.sign(self.gxPre.amplitude),np.sign(self.gx.amplitude ))
        
        self.gx.amplitude = -self.gx.amplitude #;   % Reverse polarity of read gradient
        #self.gxPre = pp.scale_grad(self.gxPre, -1)


        

    def add_to_seq(self,pseq0,EchoTimeShift=0,ghost_scan = False,
                    ghost_nav = False,mode = 'EPI_v1',label = True,calib= False):
        
        if ghost_nav:
            if label:
                self.add_ghost_nav(pseq0)
                self.ghost_flag = True
                #print(self.gx.amplitude)
            else:
               #print('no nav label')
                self.add_ghost_nav_nolabel(pseq0)
                self.ghost_flag = True
                #print(self.gx.amplitude)
        if calib:
            self.add_calib(pseq0,EchoTimeShift=EchoTimeShift,ghost_scan = ghost_scan)

        else:
            if mode == 'EPI_v1':
                self.add_epi_v1(pseq0,EchoTimeShift=EchoTimeShift,ghost_scan = ghost_scan)



    def get_timeToTE(self,save=True):
        durationToCenter = (self.Ny_pre+0.5)*pp.calc_duration(self.gx)
        pre_duration = pp.calc_duration(self.gxPre,self.gyPre)
        ghost_nav = pp.calc_duration(self.gxPre) + pp.calc_duration(self.gx)*self.ghost_lines
        if save:
            timings = {'durationToCenter': durationToCenter,
                    'pre_duration': pre_duration,
                    'timeToTE': durationToCenter + pre_duration,
                    'ghost_nav': ghost_nav,}
            
            return timings
            
        print('Time to TE is {}'.format(durationToCenter+pre_duration))

        
    
    
    def add_calib(self,pseq0,ghost_scan = False, EchoTimeShift = 0):
        dETS_before, dETS_after = self.get_echoTime_shift()

        if EchoTimeShift:
            pseq0.seq.add_block(self.dETS_before)

        start_sign_y = np.sign(self.calib_gyPre.amplitude)
        start_sign_x = np.sign(self.calib_gxPre.amplitude)

        pseq0.seq.add_block(self.calib_gyPre, self.calib_gxPre)

        

        for i in range(1,int(self.Ny_meas+1)):
            if i ==1:
                pseq0.seq.add_block(self.gx,self.calib_gy_blipup,self.adc, )
            elif i == self.Ny_meas:
                pseq0.seq.add_block(self.gx,self.calib_gy_blipdown,self.adc,)
            else:
                pseq0.seq.add_block(self.gx,self.calib_gy_blipdownup,self.adc, )

            self.gx.amplitude = -self.gx.amplitude #;   % Reverse polarity of read gradient

        if EchoTimeShift:
            pseq0.seq.add_block(self.dETS_after)

        #pseq0.seq.add_block(self.calib_gx_pre_post,self.calib_gy_pre_post)
        if start_sign_y != np.sign(self.calib_gyPre.amplitude):
            self.calib_gyPre = pp.scale_grad(self.calib_gyPre,-1)

        if start_sign_x != np.sign(self.calib_gxPre.amplitude):
            self.calib_gxPre = pp.scale_grad(self.calib_gxPre,-1)

        #self.gyPre = np

       #print(self.gx.amplitude)

    
    
    def add_epi_v1(self,pseq0,ghost_scan = False, EchoTimeShift = 0):   
        dETS_before, dETS_after = self.get_echoTime_shift()

        #if ghost_scan:
        #    self.gy_blipup.last=0
        #    self.gy_blipdown.first=0
        #    self.gy_blipdownup.last=0
        #   self.gy_blipdownup.first=0

        if EchoTimeShift:
            pseq0.seq.add_block(self.dETS_before)

        #if not self.ghost_flag:   
        #    pseq0.seq.add_block(self.gxPre,self.gyPre)

        #if self.ghost_flag:
        #    start_value = self.Ny - int(self.Ny * self.partFourierFactor) -1
           #print('Start value for EPI label is: ', start_value)
        pseq0.seq.add_block(self.gyPre, self.gxPre)
            #pp.make_label(type = 'SET',label = 'LIN', value =start_value),
            #pp.make_label(type = 'SET',label = 'NAV',value =  0),
            #pp.make_label(type = 'SET',label = 'AVG', value = 0) ) # we already are out in k-space in x! 


        for i in range(1,int(self.Ny_meas+1)):
            #lrev = pp.make_label(type ='SET', label = 'REV', value = int(np.sign(self.gx.amplitude) != self.ROpolarity ))
            #lseg = pp.make_label(type = 'SET', label = 'SEG', value = int(np.sign(self.gx.amplitude) != self.ROpolarity ))
            #llin = pp.make_label(type = 'INC', label = 'LIN', value = 1) 

            if i ==1:
                pseq0.seq.add_block(self.gx,self.gy_blipup,self.adc, )
            elif i == self.Ny_meas:
                pseq0.seq.add_block(self.gx,self.gy_blipdown,self.adc,)
            else:
                pseq0.seq.add_block(self.gx,self.gy_blipdownup,self.adc, )

            # why this is failing
            #if self.Ny_meas % 2 == 1:
            self.gx.amplitude = -self.gx.amplitude #;   % Reverse polarity of read gradient

        if EchoTimeShift:
            pseq0.seq.add_block(self.dETS_after)

        #pseq0.seq.add_block(self.gx_pre_post,self.gy_pre_post)
        print('gyPre',np.sign(self.gyPre.amplitude), 'gxPre',np.sign(self.gxPre.amplitude))
       #print(self.gx.amplitude)













    





    