import numpy as np
import pypulseq as pp
from pypulseq.opts import Opts
from pSeq_Base import pSeq_Base 

class pSeq_FatSat(pSeq_Base):
    def __init__(self, pseq=None, 
                sat_ppm= -3.45, 
                use = "saturation", 
                type = "FatSat",
                slice_thickness = 3e-3,
                duration = 3e-3,
                system = None,
                *args, **kwargs,
                 ):
        
        super().__init__(pseq, *args, **kwargs)
        
        self.sat_ppm = sat_ppm
        self.use = use
        self.type = type
        self.slice_thickness = slice_thickness
        self.duration = duration
        self.system = system

        print(self.sat_ppm * 1e-6 * self.system.B0 * self.system.gamma)

        

        print(duration,slice_thickness)
    

    def make_fatsat(self):
        sat_freq = self.sat_ppm * 1e-6 * self.system.B0 * self.system.gamma
        print(sat_freq) # 409 Hz
        self.rf_fs = pp.make_gauss_pulse(110 * np.pi / 180, system=self.system, duration=8e-3,
                                    bandwidth=abs(sat_freq), freq_offset=sat_freq, use=self.use, dwell = 10e-6)
        self.gz_fs = pp.make_trapezoid(channel = 'z', system = self.system, delay=pp.calc_duration(self.rf_fs), area= 1 / 1e-4)  # Spoil up to 0.1mm
        #self.rf_fs.phase_offset = - 2*np.pi*self.rf_fs.freq_offset*pp.calc_rf_center(self.rf_fs)[0]  # Compensate for the frequency-offset induced phase
    
    def make_stir(self,freq_offset = 0): # consider renaming slice_offset to be less confusing
        self.rf_fs, self.gz_fs, __ = pp.make_sinc_pulse(flip_angle = np.pi, system=self.system, duration=self.duration, slice_thickness = self.slice_thickness, apodization=0.5,
                                    time_bw_product = 4,return_gz=True)

        self.rf_fs.freq_offset = self.gz_fs.amplitude * freq_offset
        self.rf_fs.ringdown_time = 1e-5
        self.rf_fs.dead_time = 1e-4
        #self.rf_fs.phase_offset = - 2*np.pi*self.rf_fs.freq_offset*pp.calc_rf_center(self.rf_fs)[0]  # Compensate for the frequency-offset induced phase
        

    def make_spair(self,freq_offset = 0):
        # 'hypsec', # duration is 21 ms , delay from center of peak to excitation is 125 ms 
        sat_freq = -407# self.sat_ppm * 1e-6 * self.system.B0 * self.system.gamma
        self.rf_fs, self.gz_fs, __ = pp.make_adiabatic_pulse(pulse_type = 'hypsec',  system = self.system, 
                duration = self.duration, slice_thickness = self.slice_thickness, 
                freq_offset = sat_freq, return_gz = True,bandwidth = abs(sat_freq))
        
        #self.rf_fs.freq_offset = self.gz_fs.amplitude * freq_offset
        #self.rf_fs.phase_offset = - 2*np.pi*self.rf_fs.freq_offset*pp.calc_rf_center(self.rf_fs)[0]  # Compensate for the frequency-offset induced phase
        print('SPair freq offset',freq_offset)

    def make_spir(self,freq_offset = 0):
        self.make_fatsat()

    def make_fatsat3(self):
        print('Make Fat Sat Debugging! ')
        # Design RF Pulse
        sat_ppm = -3.3 #; % frequency offset is -407 Hz in Siemens product sequence
        sat_freq= sat_ppm*1e-6*self.system.B0*self.system.gamma #; % gamma: [Hz/T]
        print('Sat Frequency is', sat_freq)
        rf_fs = pp.make_gauss_pulse(110*np.pi/180,system = self.system,duration = 5120e-6,dwell = 10e-6,
            bandwidth = abs(sat_freq),freq_offset = sat_freq,use = 'saturation')
        rf_fs.phase_offset=np.pi/2 -2*np.pi*rf_fs.freq_offset*pp.calc_rf_center(rf_fs)[0] #; % compensate for the frequency-offset induced phase    
        # rf_fs.phaseOffset=-2*pi*rf_fs.freqOffset*mr.calcRfCenter(rf_fs) ; % compensate for the frequency-offset induced phase    
        #self.phase_offset= np.pi/2 - 2*np.pi*self.rf.freq_offset*pp.calc_rf_center(self.rf)[0] #; % compensate for the slice-offset induced phase

        # Design the crushers 
        spoiler_amp_crush = 3 * 8 * 1e-3 * self.system.gamma #; % fatSat gradient spoiler amplitude [mT/m] -> [Hz/m]
        rise_time_crush = 500e-6 #; % ramp up and ramp down time
        flat_time_crush = 2500e-6 #;
        gx_crush_pre =pp.make_trapezoid(channel = 'x',amplitude =  spoiler_amp_crush,rise_time = rise_time_crush,flat_time = flat_time_crush,system = self.system) 
        gy_crush_pre = pp.make_trapezoid(channel = 'y',amplitude = -spoiler_amp_crush,rise_time = rise_time_crush,flat_time = flat_time_crush,system = self.system) 
        gz_crush_pre = pp.make_trapezoid(channel = 'z',amplitude = spoiler_amp_crush,rise_time =rise_time_crush,flat_time =flat_time_crush,system = self.system) 
        gx_crush_post = pp.scale_grad(gx_crush_pre, -1) 
        gy_crush_post = pp.scale_grad(gy_crush_pre, -1) 
        gz_crush_post = pp.scale_grad(gz_crush_pre, -1) 

        # Assign Variables
        self.rf_fs = rf_fs

        self.gx_crush_pre = gx_crush_pre
        self.gy_crush_pre = gy_crush_pre
        self.gz_crush_pre = gz_crush_pre

        self.gx_crush_post = gx_crush_post
        self.gy_crush_post = gy_crush_post
        self.gz_crush_post = gz_crush_post





    def prep_fatsat(self,verbose = True, freq_offset = 0):
        if self.type == "FatSat" or self.type == "FatSat2":
            self.make_fatsat()
            if verbose:
                print('Fat Sat: The peak rf_fs Amplitude = ',np.max(abs(self.rf_fs.signal))/self.system.gamma, 'uT')

        elif self.type == "FatSat3" :
            self.make_fatsat3()


        elif self.type == "STIR":
            self.make_stir(freq_offset = freq_offset)
            print("Making STIR Pulse")
            
        elif self.type == "SPAIR":
            self.make_spair()
            print("Making SPAIR Pulse")

        elif self.type == "SPIR":
            self.make_spir()
            print("Making SPIR Pulse")
            

    def prep_spoiler(self, offset = 0):
        spoiler_amp = 3 * 8 * 42.58 * 10e2
        spoiler_amp = 3 * 8 * 1e-3 * self.system.gamma #; % fatSat gradient spoiler amplitude [mT/m] -> [Hz/m]
        est_rise = 500e-6  # ramp time 280 us
        est_flat = 2500e-6  # duration 600 us

        gp_r = pp.make_trapezoid('x', amplitude=spoiler_amp, rise_time=est_rise, flat_time=est_flat, system=self.system)
        gp_p = pp.make_trapezoid('y', amplitude=spoiler_amp, rise_time=est_rise, flat_time=est_flat, system=self.system)
        gp_s = pp.make_trapezoid('z', amplitude=spoiler_amp, rise_time=est_rise, flat_time=est_flat, system=self.system)

        gn_r = pp.make_trapezoid('x', amplitude=-spoiler_amp, delay=pp.calc_duration(self.rf_fs), rise_time=est_rise, flat_time=est_flat, system=self.system)
        gn_p = pp.make_trapezoid('y', amplitude=-spoiler_amp, delay=pp.calc_duration(self.rf_fs), rise_time=est_rise, flat_time=est_flat, system=self.system)
        gn_s = pp.make_trapezoid('z', amplitude=-spoiler_amp, delay=pp.calc_duration(self.rf_fs), rise_time=est_rise, flat_time=est_flat, system=self.system)


        gp_r =pp.make_trapezoid(channel = 'x',amplitude =  spoiler_amp,rise_time = est_rise,flat_time = est_flat,system = self.system) 
        gp_p = pp.make_trapezoid(channel = 'y',amplitude = -spoiler_amp,rise_time = est_rise,flat_time = est_flat,system = self.system) 
        gp_s = pp.make_trapezoid(channel = 'z',amplitude = spoiler_amp,rise_time =est_rise,flat_time =est_flat,system = self.system) 
        gn_r = pp.scale_grad(gp_r, -1) 
        gn_p = pp.scale_grad(gp_p, -1) 
        gn_s = pp.scale_grad(gp_s, -1) 


        self.gp_r = gp_r
        self.gp_p = gp_p
        self.gp_s = gp_s
        
        self.gn_r = gn_r
        self.gn_p = gn_p
        self.gn_s = gn_s

    def add_to_seq(self,pseq0,delay = 0,offset = 0):
        if self.type == "FatSat3":
            pseq0.seq.add_block(self.gx_crush_pre, self.gy_crush_pre, self.gz_crush_pre) 
            pseq0.seq.add_block(self.rf_fs) 
            pseq0.seq.add_block(self.gx_crush_post, self.gy_crush_post, self.gz_crush_post) 

        
        if self.type == "FatSat2":
            if delay > 0:
                rf_dur_to_center = self.rf.delay +  pp.calc_rf_center(self.rf_fs)[0]
                delay_block= pp.make_delay(delay + rf_dur_to_center)
            else:
                delay = pp.make_delay(0)


            pseq0.seq.add_block(self.rf_fs, self.gz_fs, delay_block)

        if self.type == "FatSat":
            if delay > 0:
                rf_dur_to_center = self.rf_fs.delay +  pp.calc_rf_center(self.rf_fs)[0]
                delay_block= pp.make_delay(delay + rf_dur_to_center)
            
            else:
                delay_block = pp.make_delay(0)
            pseq0.seq.add_block(self.gp_r,self.gp_p,self.gp_s)
            pseq0.seq.add_block(self.rf_fs, self.gn_r,self.gn_p,self.gn_s,delay_block)


        elif self.type == "STIR" :
            
            self.rf_fs.freq_offset = self.gz_fs.amplitude * offset
            #self.phase_offset = np.pi - 2*np.pi * self.rf_fs.freq_offset * pp.calc_rf_center(self.rf_fs)[0]
            print('STIR Freq offset = {}Hz, delay = {}ms, phase offset = {}'.format( self.rf_fs.freq_offset,delay, self.rf_fs.phase_offset))
            pseq0.seq.add_block(pp.make_delay(delay), self.rf_fs, self.gz_fs)

        elif self.type == "SPIR" or self.type == "SPAIR":
            pseq0.seq.add_block(self.gp_r,self.gp_p,self.gp_s)
            pseq0.seq.add_block(self.rf_fs, self.gn_r,self.gn_p,self.gn_s)
            
            pulse_dur = self.get_duration()
            rf_dur_to_center = pp.calc_rf_center(self.rf_fs)[0]
            raster_delay = np.ceil((delay - pulse_dur + rf_dur_to_center)/self.system.grad_raster_time)*self.system.grad_raster_time
            pseq0.seq.add_block(pp.make_delay(raster_delay ))
            print('DELAY FAT SAT', raster_delay)


    def get_duration(self):
        dur = pp.calc_duration(self.rf_fs, self.gz_fs)
        
        
        return dur

        

        
        



        



        

        