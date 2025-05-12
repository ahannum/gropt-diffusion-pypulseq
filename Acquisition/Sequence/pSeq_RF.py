import numpy as np
import pypulseq as pp
from pypulseq.opts import Opts
from pSeq_Base import pSeq_Base 


class pSeq_RF(pSeq_Base):
    def __init__(self, pseq=None, 
                flip_angle= np.pi/2, 
                use = "excitation", 
                duration = 3e-3, 
                do_refocus = True,
                thickness=1.5e-3,
                system  = None,

                *args, **kwargs,
                 ):
        
        super().__init__(pseq, *args, **kwargs)
        
        self.flip_angle = flip_angle
        self.use = use
        self.duration = duration
        self.do_refocus = do_refocus
        self.thickness = thickness
        if system == None:
            self.system = pseq.seq.system
        else:
            system = self.system
  

    
    def prep_trig(self,seq_type = None):
        if seq_type == "Skope":
            self.trig = pp.make_digital_output_pulse('ext1', duration=100e-6)  # possible channels: 'osc0','osc1','ext1'
        #elif seq_type == "Cardiac":
        #    self.trig = pp.make_digital_output_pulse('trig', duration=100e-6)  # possible channels: 'osc0','osc1','ext1'
        else:
            self.trig = pp.make_digital_output_pulse('osc0', duration=100e-6)
    

    def prep(self,tbw=4,apodization=0.5,phase_offset = 0, type = None,verbose = True, default = 'sinc', SLR_params = None,ftype = 'ms',inner_volume = False,custom_slew=None):
        
        if self.use == "refocusing":
            phase_offset = np.pi/2
            self.phase_offset= phase_offset
        else:
            self.phase_offset = 0
            phase_offset = 0

        #if custom_slew == None:
        #    slew_adj = self.system.max_slew
        #else:
        #    slew_adj  =pp.convert.convert(from_value = custom_slew,from_unit = 'T/m/s',to_unit = 'Hz/m/s')
            
        
        if default == 'sinc' or 'Sinc':
            print('Making Sinc Pulse with flip angle',self.flip_angle)
            self.rf, self.gz, self.gzReph = pp.make_sinc_pulse(flip_angle = self.flip_angle, 
                                                system=self.system, duration=self.duration,
                                                slice_thickness=self.thickness, apodization = apodization, 
                                                phase_offset = phase_offset,
                                                time_bw_product=tbw,use=self.use,return_gz = True)
        
        if default == 'SLR':
            print('Making SLR Pulse with flip angle',self.flip_angle)
            from pypulseq import make_sigpy_pulse, sigpy_pulse_opts

            
            apodization =apodization
            time_bw_product = tbw
            
            sigpy_config = sigpy_pulse_opts.SigpyPulseOpts(ftype = ftype)
            rfp,gz,gzr,pulse = make_sigpy_pulse.sigpy_n_seq(flip_angle = self.flip_angle,  freq_offset = 0, phase_offset = self.phase_offset, 
                                         slice_thickness = self.thickness, system = self.system, plot = False,duration = self.duration,
                                        time_bw_product = time_bw_product, use = self.use,return_gz =True, pulse_cfg = sigpy_config)
            
            if inner_volume:
                gz.channel = 'x'
            
            self.rf = rfp
            self.gz = gz
            self.gzReph = gzr
            self.pulse = pulse
        
        
        if inner_volume and self.use =="refocusing":
            print('Making Innner-volume post, double check params')
            if default == 'sinc':
                self.rf, self.gz, self.gzReph = pp.make_sinc_pulse(flip_angle = self.flip_angle, 
                                                system=self.system, duration=self.duration,
                                                slice_thickness=self.thickness, apodization = apodization, 
                                                phase_offset = phase_offset, 
                                                time_bw_product=tbw,use=self.use,return_gz = True)
            self.gz.channel = 'x'
        
        if verbose:
            gamma_H1 = 42.58
            rf_peak = max(abs(self.rf.signal))/gamma_H1 #; % [uT]

            print(self.use, ': The peak rf Amplitude = ',rf_peak, 'uT') 
        
        self.default = default
        
        #rf180, gz180, _ = pp.make_sinc_pulse(np.pi, system=lims, duration=tRFref,
                                            #slice_thickness=thickness, apodization=0.5, time_bw_product=4, phase_offset=np.pi / 2, use='refocusing',return_gz= True)
        
        # Note on inner-volume --> 
        
    def make_default_seq(self):
        self.init_seq()
        self.seq.add_block(self.rf, self.gz,)
        if self.do_refocus:
            self.seq.add_block(self.gzReph)

    def offset_rf(self,offset= 0):
        self.rf.freq_offset = self.gz.amplitude * offset

        if self.use == "excitation":    
            self.phase_offset= np.pi/2 - 2*np.pi*self.rf.freq_offset*pp.calc_rf_center(self.rf)[0] #; % compensate for the slice-offset induced phase


        if self.use == 'refocusing':
            self.phase_offset = - 2*np.pi * self.rf.freq_offset * pp.calc_rf_center(self.rf)[0]
            #self.phase_offset = np.pi/2
            
        self.rf.phase_offset =self.phase_offset



    def add_to_seq(self,pseq0,channel = 'z'): 
        if self.use == "refocusing":
            pseq0.seq.add_block(self.rf, self.gz)
            

        if self.use == "excitation":
            pseq0.seq.add_block(self.trig)
            pseq0.seq.add_block(self.rf, self.gz)
        
        if self.do_refocus:
            pseq0.seq.add_block(self.gzReph)

        if self.use == "saturation":
            #self.use = "excitation"
            self.gz.channel = channel
            pseq0.seq.add_block(self.rf, self.gz)
            
            #est_rise = 500e-6  # ramp time 280 us
            #est_flat = 2500e-6  # duration 600 us
            #gz_fs = pp.make_trapezoid(channel = 'z',system = self.system, rise_time=est_rise, flat_time=est_flat,area = 0.1/1e-4)#; % spoil up to 0.1mm
            #gx_fs = pp.make_trapezoid(channel = 'y',system = self.system,  rise_time=est_rise, flat_time=est_flat, area = 0.1/1e-4)#; % spoil up to 0.1mm
            #pseq0.seq.add_block(gz_fs,gx_fs)

            #print(gz_fs.area)
            
            


    def get_duration(self):
        dur = pp.calc_duration(self.rf, self.gz)
        
        if self.do_refocus:
            dur += pp.calc_duration(self.gzReph)
        
        return dur
    
    def get_timing(self):
        """
        This is the timing for GrOpt also optionally returns the dictionary of all timings 
        """
        timing = self.get_duration()
        center_incl_delay = self.rf.delay +  pp.calc_rf_center(self.rf)[0]
        if self.use == "excitation":
            timing -= center_incl_delay

        #if self.use == "refocusing":
        #    timing += self.rf.delay

        time_dict = {'timing': timing ,
                     'center_incl_delay': center_incl_delay}
        
        return timing, time_dict


    





            
