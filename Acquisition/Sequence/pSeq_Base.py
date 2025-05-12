import numpy as np
import pypulseq as pp
from pypulseq.opts import Opts
import os
from pypulseq.utils.siemens.readasc import readasc
from pypulseq.utils.siemens.asc_to_hw import asc_to_hw, asc_to_acoustic_resonances


class pSeq_Base:
    def __init__(self, pseq=None, channels=("x","y","z"),  **kwargs):
    
        if pseq is not None:
            self.system = pseq.system
            self.channels = pseq.channels

        else:
            # Some defaults 
            opt_args = {"grad_unit": "mT/m", 
                        "slew_unit": "T/m/s",
                        "rf_ringdown_time": 10e-6,#10e-6,
                        "rf_dead_time": 100e-6,
                        "adc_dead_time": 10e-6,
                        "B0":  2.89}
            
            
            # Add any Opts arguments from kwargs to the self Opts
            for key, val in kwargs.items():
                if key in Opts.default.__dict__:
                    opt_args[key] = val
                    
            self.system = Opts(**opt_args)
            self.channels = channels

        self.seq = pp.Sequence(system = self.system)

    def init_epi_system(self, pseq=None, channels=("x","y","z"),  **kwargs):
        if pseq is not None:
            self.epi_system = pseq.system

        else:
            # Some defaults 
            opt_args = {"grad_unit": "mT/m", 
                        "slew_unit": "T/m/s",
                        "rf_ringdown_time": 10e-6,#10e-6,
                        "rf_dead_time": 100e-6,
                        "adc_dead_time": 10e-6,
                        "B0":  2.89}
            
            # Add any Opts arguments from kwargs to the self Opts
            for key, val in kwargs.items():
                if key in Opts.default.__dict__:
                    opt_args[key] = val
                    
            self.epi_system = Opts(**opt_args)
            self.channels = channels


    
    def init_seq(self):
        self.rf_spoil_idx = 0
        self.rf_spoil_phase = 0
        self.seq = pp.Sequence(system = self.system)
        
    def add_pseq_to_self(self, pseq, *args, **kwargs):
        pseq.add_to_seq(self, *args, **kwargs)

    def add_TR(self,TR = 1):
        self.seq.add_block(pp.make_delay(TR))


    def add_ecgTrig(self, duration = 100e-3):
        self.trigECG=pp.make_trigger(channel = 'physio1',duration= duration) #; %  duration after trig to acquire in diastole 500 ms
        self.seq.add_block(self.trigECG)

        
    def get_seq_time(self):
        wave_data, tfp_excitation, tfp_refocusing, t_adc, fp_adc = self.seq.waveforms_and_times(append_RF=True)
        return wave_data, tfp_excitation, tfp_refocusing, t_adc, fp_adc  

    def get_report(self):
        ok, error_report = self.seq.check_timing()
        if ok:
            print("Timing check passed successfully")
        else:
            print("Timing check failed. Error listing follows:")
            [print(e) for e in error_report]


    def write_seq(self,fov = 220,slicePositions = 0,thickness = 3e-3, 
        sliceGap = 0, ro_os = 1,Ny = 80,Nx=80,trap_grid_params ={},seq_name = 'test.seq',gx = pp.make_trapezoid(channel='x',system = pp.Opts(), amplitude=  965406, duration = 1e-3),adc = pp.make_adc(80,system =  pp.Opts(),dwell=1e-5)):

        self.seq.set_definition('Name', 'epi')
        self.seq.set_definition('FOV', [fov, fov, max(slicePositions)-min(slicePositions)+thickness])
        self.seq.set_definition('SlicePositions', slicePositions)
        self.seq.set_definition('SliceThickness', thickness)
        self.seq.set_definition('SliceGap', sliceGap)
        self.seq.set_definition('ReceiverGainHigh', 1)
        self.seq.set_definition('ReadoutOversamplingFactor',ro_os)
        self.seq.set_definition('TargetGriddedSamples', Nx*ro_os)
        self.seq.set_definition('TrapezoidGriddingParameters', [gx.rise_time, gx.flat_time, gx.fall_time, adc.delay-gx.delay, adc.duration])
        self.seq.set_definition('kSpaceCenterLine', Ny/2)
        self.seq.set_definition('PhaseResolution', 1) 
        self.seq.set_definition('BaseResolution', Nx) 
        

        self.seq.write(seq_name)
    
    
    def check_safety_lims(self,acs_file ='/Users/ariel/Documents/PhD/Projects/asc_files/MP_GradSys_P034_X60.asc', cns = False, time_range = [0,0.1], do_plot= True ):
        
        # Need to run seperately CNS vs. PNS 
        asc, extra = readasc(os.path.join(acs_file)) #load asc file
        hw = asc_to_hw(asc, cardiac_model = False) #convert to hardware object
        # Calculate PNS and CNS responses
        ok, pns_norm, pns_comp, t = self.seq.calculate_pns(hardware = hw,
                        time_range = time_range, do_plots = False)
        print('PNS is{} OK'.format('' if ok else ' NOT'))

        if cns:
            asc, extra = readasc(os.path.join(acs_file)) #load asc file
            hw = asc_to_hw(asc, cardiac_model = True) #convert to hardware object
            # Calculate PNS and CNS responses
            ok, cns_norm, cns_comp, t_cns = self.seq.calculate_pns(hardware = hw,
                            time_range = time_range, do_plots = False)
            print('CNS is{} OK'.format('' if ok else ' NOT'))

        # Calculate Gradient Spectrum
        freq = asc_to_acoustic_resonances(asc=asc)
        spectrograms, spectrogram_rss, frequencies, times = self.seq.calculate_gradient_spectrum(acoustic_resonances=freq, plot = False)

        if do_plot:
            from matplotlib import pyplot as plt
            plt.figure(10,figsize = (20,6))
            plt.rcParams.update({'font.size': 36, 'lines.linewidth': 4}) 
            pns_norm2 = 100*pns_norm
            pns_comp2 = 100*pns_comp
            if not cns:
                plt.subplot(121)
            if cns: 
                plt.subplot(131)
            plt.plot(t, pns_comp2[:,0], 'r-',
                        t, pns_comp2[:,1], 'g-',
                        t, pns_comp2[:,2], 'b-',
                        t, pns_norm2 , 'k-')
                
            plt.ylim([0, 120])
            plt.xlim([min(t), max(t)])

            plt.title(f'Predicted PNS ({max(pns_norm2):0.0f}%)',fontsize = 20)
            plt.ylabel('Relative stimulation [%]',fontsize = 20)
            plt.xlabel('Time [ms]',fontsize = 20)
            plt.plot([0, max(t)], [max(pns_norm2), max(pns_norm2)], 'k:')
            plt.legend([f'X ({max(pns_comp2[:,0]):0.0f}%)',
                    f'Y ({max(pns_comp2[:,1]):0.0f}%)',
                    f'Z ({max(pns_comp2[:,2]):0.0f}%)',
                    f'nrm ({max(pns_norm2):0.0f}%)'], loc='best',fontsize = 15)
            plt.tick_params(axis='both', which='major', labelsize=18)
            if not cns:
                plt.subplot(122)
            if cns:
                plt.subplot(132)
                cns_norm2 = 100*cns_norm
                cns_comp2 = 100*cns_comp
                plt.plot(t, cns_comp2[:,0], 'r-',
                            t, cns_comp2[:,1], 'g-',
                            t, cns_comp2[:,2], 'b-',
                            t, cns_norm2 , 'k-')
                    
                plt.ylim([0, 120])
                plt.xlim([min(t), max(t)])

                plt.title(f'Predicted CNS ({max(cns_norm2):0.0f}%)',fontsize = 20)
                plt.ylabel('Relative stimulation [%]',fontsize = 20)
                plt.xlabel('Time [ms]',fontsize = 20)
                plt.plot([0, max(t)], [max(cns_norm2), max(cns_norm2)], 'k:')
                plt.legend([f'X ({max(cns_comp2[:,0]):0.0f}%)',
                        f'Y ({max(cns_comp2[:,1]):0.0f}%)',
                        f'Z ({max(cns_comp2[:,2]):0.0f}%)',
                        f'nrm ({max(cns_norm2):0.0f}%)'], loc='best',fontsize = 15)

                plt.tick_params(axis='both', which='major', labelsize=18)
                
            if cns:
                plt.subplot(133)
            # Plot  Gradient Spectrum Checks 
            plt.xlabel('Frequency (Hz)',fontsize = 20)
            plt.title('Gradient Waveform in Hz/m',fontsize = 20)
            # According to spectrogram documentation y unit is (Hz/m)^2 / Hz = Hz/m^2, is this meaningful?
            for s in spectrograms:
                plt.plot(frequencies, s)
                plt.plot(frequencies, spectrogram_rss)
                plt.legend(['x', 'y', 'z', 'rss'],fontsize = 15)

            for res in freq:
                plt.axvline(res['frequency'], color='k', linestyle='-')
                plt.axvline(res['frequency'] - res['bandwidth']/2, color='k', linestyle='--')
                plt.axvline(res['frequency'] + res['bandwidth']/2, color='k', linestyle='--')

            plt.tick_params(axis='both', which='major', labelsize=18)
            plt.subplots_adjust(wspace=0.3)  # Adjust the width space as needed

            plt.show()


    

