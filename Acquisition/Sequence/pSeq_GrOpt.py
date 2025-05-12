import numpy as np
import pypulseq as pp
from pypulseq.opts import Opts
from pSeq_Base import pSeq_Base 
from matplotlib import pyplot as plt
import sys
import copy



import os
from pypulseq.utils.siemens.readasc import readasc
from pypulseq.utils.siemens.asc_to_hw import asc_to_hw, asc_to_acoustic_resonances


sys.path.append('/Users/ariel/Documents/PhD/Projects/gropt-diffusion-pulseq/utils/gropt/python')
import gropt
from helper_utils import *



class pSeq_GrOpt(pSeq_Base):
    def __init__(self, pseq=None, 
                t_90= 3,
                t_180 = 6,
                dt = 8e-5,
                T_readout = 10,
                durationToCenter = 9.8,
                pre_duration = 0.2,
                b = 1000,
                mmt = 0,
                pns_thresh = 0.8,
                gmax = 200,
                smax = 200,
                tol = 0.01,
                rf_90dict = {},
                rf_180dict = {},
                *args, **kwargs,
                 ):
        
        super().__init__(pseq, *args, **kwargs)
        

        self.timings = {"T_90": t_90,
                        "T_180": t_180,
                        "dt": dt,
                        "T_readout": T_readout,
                        "durationToCenter": durationToCenter,
                        "pre_duration": pre_duration,
                        "b": b,
                        "MMT": mmt,
                        "pns_thresh": pns_thresh,
                        "gmax": gmax-1,
                        "smax": smax-1,
                        "tol":tol,
                        'mode': 'diff_bval', 

                        
                        }
        
        self.rf_90 = rf_90dict
        self.rf_180 = rf_180dict
        self.TE = -1
        self.grad_amplitude = self.timings['gmax']

        

    def get_delay(self):
            #self.get_duration()
            # Calculate first TE delay (TE1)
            self.delayTE1 = np.ceil((self.TE/2 - self.rf_90['timing'] - self.rf_180['center_incl_delay'])/self.system.grad_raster_time)*self.system.grad_raster_time 
            
            # Calculate second TE delay (TE2)
            self.delayTE2= np.ceil((self.TE/2 - self.rf_180['timing'] + self.rf_180['center_incl_delay'] - self.timings['T_readout']*1e-3)/self.system.grad_raster_time)*self.system.grad_raster_time 
            #self.delayTE2 -= self.timings['pre_duration']*1e-3

    def get_min_delay(self):
        delayTE1 = -1
        delayTE2 = -1


        while delayTE1 < 0 or delayTE2 < 0:
            self.get_delay()
            delayTE1 = self.delayTE1 - pp.calc_duration(self.gz180_crusher_1)
            delayTE2 = self.delayTE2 - pp.calc_duration(self.gz180_crusher_1)

            self.TE += self.system.grad_raster_time

        self.min_delayTE1 = self.delayTE1
        self.min_delayTE2 = self.delayTE2

        return self.TE
    
    def make_gropt(self,custom_T180=None):
        if custom_T180 is not None:
            self.timings['T_180'] = custom_T180
        
        # Make GrOPt waveform given constraints and timings
        self.timings['dt'] = 8e-5
        
        G, TE =  get_min_TE(self.timings.copy(),verbose = True , bval = self.timings['b'])
        print(self.timings,G.shape)
        self.timings['TE'] = TE
        gropt_bval = get_bval(G,self.timings)
        print("GrOpt B-Value: {:.2f} s/mm^2 ".format(gropt_bval))
        self.TE = TE*1e-3

        # Rasterize waveform to system raster time
        self.timings['dt_out'] = self.system.grad_raster_time
        self.timings['TE'] = self.TE*1e3
        self.TE = self.timings['TE'] * 1e-3

        
        G_new,_ = gropt.gropt(self.timings.copy())
        self.timings['dt'] =self.system.grad_raster_time
        print(self.timings,G_new.shape)
        bval_check = get_bval(G_new, self.timings.copy())
        self.G = np.squeeze(G_new)
        self.time_array = np.linspace(0,len(self.G)*self.timings['dt'],len(self.G))
        print("GrOpt High Raster B-val: {:.2f} s/mm^2 ".format(bval_check))

        # Find Zero Sequences
        self.sequences = self.find_zero_sequences(self.G)
        print('Zero Sequences', self.sequences)

        del self.timings['dt_out']
        self.output = {'Gradient': self.G, 'TE': self.TE, 'b': bval_check,'Time':self.time_array}

      

        
        
    def check_gropt(self, do_plot = True):
        TT_INV = self.timings['TE'] / 2 * 1e-3
        preCenter180_dur = self.rf_180['center_incl_delay'] + pp.calc_duration(self.gz180_crusher_1)
        postCenter180_dur = self.timings['T_180']*1e-3 - preCenter180_dur #pseq_diff.rf_180['center_incl_delay'] #+ pp.calc_duration(pseq_diff.gz180_crusher_1)
        
        plt.figure()
        plt.subplot(211)
        plt.plot(self.time_array,self.G,linewidth = 2)
        
        # Plot red and purple lines
        plt.axvline(x=TT_INV - preCenter180_dur, color='red')
        plt.axvline(x=TT_INV, color='black')
        plt.axvline(x=TT_INV + postCenter180_dur, color='purple')

        # Find indices to split the gradient based on the time values
        idx_pre = np.where(self.time_array <= TT_INV - preCenter180_dur)[0][-1]  # Last index before red line
        if len(self.sequences) >2:
            idx_post = np.where(self.time_array >= TT_INV + postCenter180_dur)[0][0]  +45 # First index after purple line
        else:
            idx_post = None

        # Split the gradient array
        gradient_before =self.G [:idx_pre + 1]  # Everything before the red line
        gradient_after = self.G [idx_post:]      # Everything after the purple line

        # Optional: Plot the segments for visualization
        plt.subplot(212)
        plt.axhline(y=0,color='black')
        plt.plot(self.time_array[:idx_pre + 1], gradient_before, color='red',linewidth = 2)
        plt.plot(self.time_array[idx_post:], gradient_after, color='purple',linewidth = 2)
        
        plt.axvline(x=TT_INV - preCenter180_dur, color='red')
        plt.axvline(x=TT_INV, color='black')
        
        plt.axvline(x=TT_INV + postCenter180_dur, color='purple')

        plt.show()

        self.TT_INV = TT_INV

        return idx_pre, idx_post



    def split_gropt(self):
        idx_start180, idx_end180 = self.check_gropt()

        if self.G[idx_start180] != 0:
            difference1 = abs(self.sequences[1][0] - idx_start180)
        else:
            difference1=0

        if idx_end180 is not None and self.G[idx_end180] != 0:
            difference2 = abs(self.sequences[1][0] - idx_end180)
        else:
            difference2=0
        factor = 10
        T180_adj = self.timings['T_180']
        num_loop = 10
        if difference1 + difference2 == 0:
            print('Timing of 180 needs padding',difference2 + difference1)
            original_T180 = copy.deepcopy(self.timings['T_180'])
            T180_adj = (difference2 + difference1 + factor )*1e-2 +self.timings['T_180'] 
            T180_adj = np.ceil(T180_adj*1e2)/1e2
            print('new T_180 is:', T180_adj, 'Original is', original_T180)
            
            #Remake Waveform
            self.timings['T_180'] = float(T180_adj)
            print(self.timings['T_180'])
            self.make_gropt()
            idx_start180, idx_end180 = self.check_gropt()

            self.timings['T_180'] = original_T180
            self.check_gropt()
            print('Gradient at start of diffusion is now:  ',self.G[idx_start180])
            factor +=15
            num_loop+=1

            
            


        self.diff1 = self.G[int(self.timings['T_90']*1e2):idx_start180]
        self.diff2 = self.G[idx_end180:]


        gap_pre_180 =  self.TT_INV - self.time_array[idx_start180] - pp.calc_duration(self.gz180_crusher_1) - self.rf_180['center_incl_delay']
        gap_post_180 =  self.time_array[idx_end180]  - self.time_array[idx_start180] -  pp.calc_duration(self.gz180_crusher_1)*2 - self.rf_180['timing'] - gap_pre_180

        
        self.gap_pre_180 = np.ceil(gap_pre_180/self.system.grad_raster_time) * self.system.grad_raster_time
        self.gap_post_180 = np.ceil(gap_post_180/self.system.grad_raster_time) * self.system.grad_raster_time

        print(gap_pre_180,gap_post_180)




    def check_timings(self):
        delayTE1_noCrush = self.delayTE1 - pp.calc_duration(self.gz180_crusher_1, self.gz180_crusher_2, self.gz180_crusher_3)   
        delayTE2_noCrush = self.delayTE2 - pp.calc_duration(self.gz180_crusher_1, self.gz180_crusher_2, self.gz180_crusher_3)
        
        print("Original size of diffusion 1",self.diff1.shape[0],"delay",delayTE1_noCrush/self.system.grad_raster_time)
        print("Original size of diffusion 2",self.diff2.shape[0],"delay",delayTE2_noCrush/self.system.grad_raster_time)
        
        # Compute zeros to pad for 90
        #pad_90 =  int(delayTE1_noCrush/self.system.grad_raster_time - self.diff1.shape[0]) #self.sequences[0][1] - np.floor(self.timings['T_90']*1e2)
        pad_90 =  int(self.sequences[0][1]) - int(np.floor(self.timings['T_90']*1e2))
        #self.diff1 = np.concatenate((np.zeros([pad_90]),self.diff1))
        print("New size of diffusion 1",self.diff1.shape[0],"delay",delayTE1_noCrush/self.system.grad_raster_time)

        #pad_180= int(delayTE2_noCrush/self.system.grad_raster_time - self.diff2.shape[0]) #self.sequences[0][1] - np.floor(self.timings['T_90']*1e2)
        pad_180 = int(( int(self.sequences[1][1]) - int(np.floor(self.timings['T_180']*1e2))) / 2)
        #self.diff1 = np.concatenate((self.diff1, np.zeros((pad_180))),)
        #self.diff2 = np.concatenate((np.zeros((pad_180)),self.diff2))

        if len(self.sequences) > 2:
            self.diff2 = np.concatenate((self.diff2,np.zeros((self.sequences[2][1]+2))))
        

        
        print("New size of diffusion 2",self.diff2.shape[0],"delay",delayTE2_noCrush/self.system.grad_raster_time)

        self.updated_g = np.concatenate((np.zeros(int(self.timings['T_90']*1e2)),self.diff1,np.zeros((int(self.timings['T_180']*1e2))),self.diff2))
        moments = self.get_moments1(self.updated_g[:,np.newaxis], self.timings['T_readout'], self.timings['dt_out'])
        print
        count = 0
        for mm in moments:
            print('M{:.2f}: {:.12f}'.format(count,mm[-1]))
            count+=1
        
        
    def prep_GrOpt(self,diff_vec = [0,0,0]):
        g1_convert = pp.convert.convert(from_value = np.array(self.diff1)*1e3, from_unit = 'mT/m', to_unit = 'Hz/m')
        g2_convert = pp.convert.convert(from_value = np.array(self.diff2)*1e3, from_unit = 'mT/m', to_unit = 'Hz/m')

        gDiff1_x = g1_convert * diff_vec[0]
        gDiff1_y = g1_convert * diff_vec[1]
        gDiff1_z = g1_convert * diff_vec[2]

        self.gDiff1_x = pp.make_arbitrary_grad(channel='x',waveform = gDiff1_x,system = self.system)
        self.gDiff1_y = pp.make_arbitrary_grad(channel='y',waveform = gDiff1_y,system = self.system)
        self.gDiff1_z = pp.make_arbitrary_grad(channel='z',waveform = gDiff1_z,system = self.system)

        gDiff2_x = g2_convert * diff_vec[0]
        gDiff2_y = g2_convert * diff_vec[1]
        gDiff2_z = g2_convert * diff_vec[2]
        if self.diff2.shape[0] != 0:
            self.gDiff2_x = pp.make_arbitrary_grad(channel='x',waveform = gDiff2_x,system = self.system)
            self.gDiff2_y = pp.make_arbitrary_grad(channel='y',waveform = gDiff2_y,system = self.system)
            self.gDiff2_z = pp.make_arbitrary_grad(channel='z',waveform = gDiff2_z,system = self.system)
        
        self.diff_vec = diff_vec

    def prep_crusher(self, custom_slew = None):
        import copy
        crusher_d=0.95e-3
        system = copy.deepcopy(self.system)
        print('crusher max slew',system.max_slew, )
        if custom_slew is not None:
           system.max_slew = pp.convert.convert(from_value = custom_slew,from_unit = 'T/m/s',to_unit = 'Hz/m/s')

        self.gz180_crusher_1=pp.make_trapezoid(channel='z',system=system,amplitude=19.05*42.58*10e2,duration=crusher_d) # that's what we used for Siemens
        self.gz180_crusher_2=pp.make_trapezoid(channel='y',system=system,amplitude=19.05*42.58*10e2,duration=crusher_d) # that's what we used for Siemens
        self.gz180_crusher_3=pp.make_trapezoid(channel='x',system=system,amplitude=19.05*42.58*10e2,duration=crusher_d) # that's what we used for Siemens

        #if self.timings['MMT'] == 1:
        #    self.rf_180['timing'] += pp.calc_duration(self.gz180_crusher_1)*2

        self.timings['T_180'] += 2*pp.calc_duration(self.gz180_crusher_1)*1e3

    def add_to_seq(self,pseq0,flag = "diff1",diff_mode = 'GrOpt'):
        if flag == "diff1":
            if diff_mode == 'GrOpt':
                self.get_delay()
                delay = self.delayTE1 #- pp.calc_duration(self.gz180_crusher_1)
                # If all amplitudes 0 add crusher
                if self.diff_vec[0] == 0 and self.diff_vec[1] == 0 and self.diff_vec[2] == 0:   
                    
                    #pseq0.seq.add_block(pp.make_delay(pp.calc_duration(self.gDiff1_x,self.gDiff1_y,self.gDiff1_z)),pp.make_delay(delayTE1))
                    #pseq0.seq.add_block(self.gap_pre_180)
                    delay = self.delayTE1  - self.gap_pre_180 
                    #pseq0.seq.add_block(pp.make_delay(delay))
                    pseq0.seq.add_block(pp.make_delay(pp.calc_duration(self.gDiff1_x,self.gDiff1_y,self.gDiff1_z)),pp.make_delay(delay))
                    pseq0.seq.add_block(pp.make_delay(self.gap_pre_180))
                    ##pseq0.seq.add_block(self.gz180_crusher_1,self.gz180_crusher_2,self.gz180_crusher_3)
                    return
                else:
                    #delayTE1 = self.delayTE1 - len(self.diff1)*self.system.grad_raster_time - self.gap_pre_180 
                    # Add Diffusion Lobes and Delay
                    #pseq0.seq.add_block(self.gDiff1_x,self.gDiff1_y,self.gDiff1_z,pp.make_delay(delayTE1))
                    #pseq0.seq.add_block(self.gap_pre_180)

                    delay = self.delayTE1  - self.gap_pre_180 - pp.calc_duration(self.gDiff1_x,self.gDiff1_y,self.gDiff1_z)
                    if delay > 0:
                        pseq0.seq.add_block(pp.make_delay(delay))

                    pseq0.seq.add_block(self.gDiff1_x,self.gDiff1_y,self.gDiff1_z)
                    #pseq0.seq.add_block(pp.make_delay(pp.calc_duration(delay) - pp.calc_duration(self.gDiff1_x,self.gDiff1_y,self.gDiff1_z)))
                    pseq0.seq.add_block(pp.make_delay(self.gap_pre_180))

            if diff_mode == 'Trap':
                if self.timings['MMT'] == 0:
                    self.get_delay()
                    pseq0.seq.add_block(self.gDiffA_x,self.gDiffA_y,self.gDiffA_z,self.delayTE1)
                    #pseq0.seq.add_block(self.delayPre180)

                if self.timings['MMT'] == 1:
                    self.get_delay()
                    
                    pseq0.seq.add_block(self.gDiffA_x, self.gDiffA_y,self.gDiffA_z,)
                    pseq0.seq.add_block(self.gDiffB_x, self.gDiffB_y,self.gDiffB_z,)
                    pseq0.seq.add_block(self.delayTE1 - pp.calc_duration(self.gDiffA_x) - pp.calc_duration(self.gDiffB_x))
                    pseq0.seq.add_block(pp.make_delay(self.system.grad_raster_time*10))
                

                if self.timings['MMT'] == 2:
                    self.get_delay()
                    pseq0.seq.add_block(self.gap)
                    pseq0.seq.add_block(self.gDiffA_x,self.gDiffA_y,self.gDiffA_z)
                    pseq0.seq.add_block(self.gDiffB_x,self.gDiffB_y,self.gDiffB_z)
                    pseq0.seq.add_block(pp.make_delay(pp.calc_duration(self.delayTE1) - pp.calc_duration(self.gDiffA_x) - pp.calc_duration(self.gDiffB_x)-pp.calc_duration(self.gap)  ))

                if self.timings['MMT'] == 3:
                    if np.sign(self.gDiffA_x.amplitude) <0 or np.sign(self.gDiffA_y.amplitude) <0 or np.sign(self.gDiffA_z.amplitude) <0:
                        self.gDiffA_x = pp.scale_grad(self.gDiffA_x,-1)
                        self.gDiffA_y = pp.scale_grad(self.gDiffA_y,-1)
                        self.gDiffA_z = pp.scale_grad(self.gDiffA_z,-1)

                    if np.sign(self.gDiffB_x.amplitude) > 0 or np.sign(self.gDiffB_y.amplitude) > 0 or np.sign(self.gDiffB_z.amplitude) > 0:
                        self.gDiffB_x = pp.scale_grad(self.gDiffB_x,-1)
                        self.gDiffB_y = pp.scale_grad(self.gDiffB_y,-1)
                        self.gDiffB_z = pp.scale_grad(self.gDiffB_z,-1)

                    if np.sign(self.gDiffC_x.amplitude) or np.sign(self.gDiffC_y.amplitude) or np.sign(self.gDiffC_z.amplitude) < 0:
                        self.gDiffC_x = pp.scale_grad(self.gDiffC_x,-1)
                        self.gDiffC_y = pp.scale_grad(self.gDiffC_y,-1)
                        self.gDiffC_z = pp.scale_grad(self.gDiffC_z,-1)

        
                    self.get_delay()
                    pseq0.seq.add_block(self.delayTE1 - pp.calc_duration(self.gDiffA_x) - pp.calc_duration(self.gDiffB_x) -  pp.calc_duration(self.gDiffC_x))
                    pseq0.seq.add_block(self.gDiffA_x, self.gDiffA_y,self.gDiffA_z,)
                    pseq0.seq.add_block(self.gDiffB_x, self.gDiffB_y,self.gDiffB_z,)
                    pseq0.seq.add_block(self.gDiffC_x, self.gDiffC_y,self.gDiffC_z,)
                    
                
        elif flag == "diff2":
            if diff_mode == 'GrOpt':
                print(diff_mode)
                self.get_delay()
                delay = self.delayTE2 #- pp.calc_duration(self.gz180_crusher_1)
                #delayTE2 = self.delayTE2 - len(self.diff2)*self.system.grad_raster_time - self.gap_post_180 
                
                delay = self.delayTE2 #- pp.calc_duration(self.gz180_crusher_1)
                if self.diff2.shape[0] == 0:
                    #pseq0.seq.add_block(pp.make_delay(self.gap_post_180))
                    #pseq0.seq.add_block(pp.make_delay(delayTE2))
                    #pseq0.seq.add_block(self.gz180_crusher_1,self.gz180_crusher_2,self.gz180_crusher_3)
                    delay = self.delayTE2  - self.gap_post_180 
                    
                    pseq0.seq.add_block(pp.make_delay(self.gap_post_180 ))
                    #pseq0.seq.add_block(pp.make_delay(delay))
                    pseq0.seq.add_block(pp.make_delay(delay))
                    
                    return

                # If all amplitudes 0 add crusher
                if self.diff_vec[0] == 0 and self.diff_vec[1] == 0 and self.diff_vec[2] == 0:   
                    #pseq0.seq.add_block(self.gz180_crusher_1,self.gz180_crusher_2,self.gz180_crusher_3)
                    #pseq0.seq.add_block(pp.make_delay(self.delayTE2 - pp.calc_duration(self.gz180_crusher_1, self.gz180_crusher_2, self.gz180_crusher_3)))
                    #pseq0.seq.add_block(pp.make_delay(self.gap_post_180))
                    #pseq0.seq.add_block(pp.make_delay(pp.calc_duration(self.gDiff2_x,self.gDiff2_y,self.gDiff2_z)),pp.make_delay(delayTE2))
                    delay = self.delayTE2  - self.gap_post_180 - pp.calc_duration(self.gDiff2_x,self.gDiff2_y,self.gDiff2_z)
                    
                    pseq0.seq.add_block(pp.make_delay(self.gap_post_180))
                    #pseq0.seq.add_block(pp.make_delay(delay))
                    pseq0.seq.add_block(pp.make_delay(pp.calc_duration(self.gDiff2_x,self.gDiff2_y,self.gDiff2_z)))
                    if delay > 0:
                        pseq0.seq.add_block(pp.make_delay(delay))
                    return
                
                else:
                    # Add Diffusion Lofbes and Delay
                    delay = self.delayTE2  - self.gap_post_180 - pp.calc_duration(self.gDiff2_x,self.gDiff2_y,self.gDiff2_z)
                    
                    pseq0.seq.add_block(pp.make_delay(self.gap_post_180))
                    #pseq0.seq.add_block(pp.make_delay((pp.calc_duration(delay) - pp.calc_duration(self.gDiff2_x,self.gDiff2_y,self.gDiff2_z))))
                    pseq0.seq.add_block(self.gDiff2_x,self.gDiff2_y,self.gDiff2_z)
                    if delay > 0:
                        pseq0.seq.add_block(pp.make_delay(delay))
                    #pseq0.seq.add_block(pp.make_delay(self.gap_post_180))
                    #pseq0.seq.add_block(self.gDiff2_x,self.gDiff2_y,self.gDiff2_z,pp.make_delay(delay),pp.make_delay(delayTE2))
            
            elif diff_mode == 'Trap':
                if self.timings['MMT'] == 0:
                    #pseq0.seq.add_block(self.delayPost180)
                    pseq0.seq.add_block(self.gDiffA_x,self.gDiffA_y,self.gDiffA_z,self.delayTE2)

                if self.timings['MMT'] == 1:
                    pseq0.seq.add_block(pp.make_delay(self.system.grad_raster_time*10))
                    pseq0.seq.add_block(self.delayTE2 - pp.calc_duration(self.gDiffA_x) - pp.calc_duration(self.gDiffB_x))
                    pseq0.seq.add_block(self.gDiffA_x, self.gDiffA_y,self.gDiffA_z,)
                    pseq0.seq.add_block(self.gDiffB_x, self.gDiffB_y,self.gDiffB_z,)

                if self.timings['MMT'] == 2:
                    #pseq0.seq.add_block(self.delayPost180)
                    pseq0.seq.add_block(pp.make_delay(self.delayTE2 - pp.calc_duration(self.gDiffA_x) - pp.calc_duration(self.gDiffB_x) ))
                    pseq0.seq.add_block(self.gDiffB_x,self.gDiffB_y,self.gDiffB_z)
                    pseq0.seq.add_block(self.gDiffA_x,self.gDiffA_y,self.gDiffA_z)

                if self.timings['MMT'] == 3:
                    pseq0.seq.add_block(self.delayTE2 - pp.calc_duration(self.gDiffA_x) - pp.calc_duration(self.gDiffB_x) -  pp.calc_duration(self.gDiffC_x))

                    if np.sign(self.gDiffA_x.amplitude) > 0 or np.sign(self.gDiffA_y.amplitude) > 0 or np.sign(self.gDiffA_z.amplitude) > 0:
                        self.gDiffA_x = pp.scale_grad(self.gDiffA_x,-1)
                        self.gDiffA_y = pp.scale_grad(self.gDiffA_y,-1)
                        self.gDiffA_z = pp.scale_grad(self.gDiffA_z,-1)

                    if np.sign(self.gDiffB_x.amplitude) < 0  or np.sign(self.gDiffB_y.amplitude) < 0 or np.sign(self.gDiffB_z.amplitude) < 0 :
                        self.gDiffB_x = pp.scale_grad(self.gDiffB_x,-1)
                        self.gDiffB_y = pp.scale_grad(self.gDiffB_y,-1)
                        self.gDiffB_z = pp.scale_grad(self.gDiffB_z,-1)

                    if np.sign(self.gDiffC_x.amplitude) > 0 or np.sign(self.gDiffC_y.amplitude) > 0 or np.sign(self.gDiffC_z.amplitude) > 0:
                        self.gDiffC_x = pp.scale_grad(self.gDiffC_x,-1)
                        self.gDiffC_y = pp.scale_grad(self.gDiffC_y,-1)
                        self.gDiffC_z = pp.scale_grad(self.gDiffC_z,-1)


                    pseq0.seq.add_block(self.gDiffC_x, self.gDiffC_y,self.gDiffC_z,)
                    pseq0.seq.add_block(self.gDiffB_x, self.gDiffB_y,self.gDiffB_z,)
                    pseq0.seq.add_block(self.gDiffA_x, self.gDiffA_y,self.gDiffA_z,)

                    if np.sign(self.gDiffA_x.amplitude) <0 or np.sign(self.gDiffA_y.amplitude) <0 or np.sign(self.gDiffA_z.amplitude) <0:
                        self.gDiffA_x = pp.scale_grad(self.gDiffA_x,-1)
                        self.gDiffA_y = pp.scale_grad(self.gDiffA_y,-1)
                        self.gDiffA_z = pp.scale_grad(self.gDiffA_z,-1)

                    if np.sign(self.gDiffB_x.amplitude) > 0  or np.sign(self.gDiffB_y.amplitude) > 0 or np.sign(self.gDiffB_z.amplitude) > 0:
                        self.gDiffB_x = pp.scale_grad(self.gDiffB_x,-1)
                        self.gDiffB_y = pp.scale_grad(self.gDiffB_y,-1)
                        self.gDiffB_z = pp.scale_grad(self.gDiffB_z,-1)

                    if np.sign(self.gDiffC_x.amplitude) < 0 or np.sign(self.gDiffC_y.amplitude) < 0 or np.sign(self.gDiffC_z.amplitude) < 0:
                        self.gDiffC_x = pp.scale_grad(self.gDiffC_x,-1)
                        self.gDiffC_y = pp.scale_grad(self.gDiffC_y,-1)
                        self.gDiffC_z = pp.scale_grad(self.gDiffC_z,-1)
                    

                    

    def make_trap(self,tol=0.01, smax_custom = None,gmax_custom = None):
        # Pilot making trap
        
        gmax = self.timings['gmax']
        if smax_custom is not None:
            smax = smax_custom
            smax_og = smax_custom
        else:
            smax = self.timings['smax']
            smax_og = self.timings['smax']

        if gmax_custom is not None:
            gmax = gmax_custom
        
        
            
        
        pns_thresh = self.timings['pns_thresh']
        pns = 100

        print('Gmax {} and Smax {}'.format(gmax,smax))

        while pns >= pns_thresh:

            if self.timings['MMT'] == 0:
                output = self.monopolar(gmax = gmax, smax = smax,tol= tol)

            elif self.timings['MMT'] == 1:
                output = self.bipolar(gmax = gmax, smax = smax,tol= tol)

            elif self.timings['MMT'] == 2:
                output = self.asymm(gmax = gmax, smax = smax,tol= tol)
            
            elif self.timings['MMT'] == 3:
                output = self.m3(gmax = gmax, smax = smax,tol= tol)

         
            pns_all = get_stim(output['Gradient'][np.newaxis,:],self.timings['dt'])
            pns = np.max(pns_all)
            print('PNS: {:.2f}, B-val: {:.2f}, TE = {:.2f}, Smax = {:.2f},Gmax = {:.2f}'.format(pns,output['b'],output['TE'],smax,gmax))
            self.pns = pns_all
            if pns <= pns_thresh:
                break
            if smax == 1:
                gmax -= 1
                smax = smax_og #self.timings['smax']
            else:    
                smax -= 1

        self.TE = self.timings['TE']*1e-3
        self.trap_params = output
        self.G  = output['Gradient']
        self.output = output
    

        


    def prep_trap(self,diff_vec = [0,0,0]):
        self.diff_vec = diff_vec
        if self.timings['MMT'] == 0   :
            amplitude = pp.convert.convert(from_value =self.trap_params['Amplitude']*1e3, from_unit = 'mT/m', to_unit = 'Hz/m')
            self.trapA = pp.make_trapezoid(channel='z',system=self.system, amplitude = amplitude,rise_time = self.trap_params['rise_time'],
                                        flat_time = self.trap_params['trapA_flat'],fall_time = self.trap_params['rise_time']) 

            self.gap = pp.make_delay(np.ceil(self.trap_params['gap']/self.system.grad_raster_time)*self.system.grad_raster_time)
            
            sequences = self.find_zero_sequences(self.trap_params['Gradient'])
            TT_inv = self.trap_params['TE']/2*1e-3
            index_inversion = np.argmin(np.abs(self.trap_params['Time'] - TT_inv))
            start_index_gapP180 = sequences[1][0]
            
            zeros_to_inv = index_inversion- start_index_gapP180
            zeros_to_inv_time = zeros_to_inv * (self.trap_params['Time'][1]-self.trap_params['Time'][0])
            print(zeros_to_inv_time)

            gap_to_180 = zeros_to_inv_time  - self.rf_180['center_incl_delay']
            gap_post_180 = self.trap_params['gap_p_180'] - (gap_to_180 + self.rf_180['timing'])
            self.delayPre180 = pp.make_delay(np.ceil(gap_to_180/self.system.grad_raster_time)*self.system.grad_raster_time)
            self.delayPost180 =  pp.make_delay(np.ceil(gap_post_180/self.system.grad_raster_time)*self.system.grad_raster_time)


            self.gDiffA_x = copy.deepcopy(self.trapA)
            self.gDiffA_x.amplitude *= diff_vec[0]
            self.gDiffA_y = copy.deepcopy(self.trapA)
            self.gDiffA_y.amplitude *= diff_vec[1]
            self.gDiffA_z = copy.deepcopy(self.trapA)
            self.gDiffA_z.amplitude *= diff_vec[2]

            self.gDiffA_x.channel = 'x'
            self.gDiffA_y.channel = 'y'
            self.gDiffA_z.channel = 'z'



        if self.timings['MMT'] == 1:
            amplitude = pp.convert.convert(from_value =self.trap_params['Amplitude']*1e3, from_unit = 'mT/m', to_unit = 'Hz/m')
            self.trapA = pp.make_trapezoid(channel='z',system=self.system, amplitude = amplitude,rise_time = self.trap_params['rise_time'],
                                        flat_time = self.trap_params['trapA_flat'],fall_time = self.trap_params['rise_time']) 

            self.gap = pp.make_delay(np.ceil(self.trap_params['gap']/self.system.grad_raster_time)*self.system.grad_raster_time)
            
            sequences = self.find_zero_sequences(self.trap_params['Gradient'])
            TT_inv = self.trap_params['TE']/2*1e-3
            index_inversion = np.argmin(np.abs(self.trap_params['Time'] - TT_inv))
            start_index_gapP180 = sequences[2][0]
            
            zeros_to_inv = index_inversion- start_index_gapP180
            zeros_to_inv_time = zeros_to_inv * (self.trap_params['Time'][1]-self.trap_params['Time'][0])
            print(zeros_to_inv_time,sequences,self.rf_180)

            gap_to_180 = zeros_to_inv_time  - self.rf_180['center_incl_delay']
            
            #self.delayPre180 = pp.make_delay(np.ceil(gap_to_180/self.system.grad_raster_time)*self.system.grad_raster_time)
            #self.delayPost180 =  pp.make_delay(np.ceil((self.delayTE2 - pp.calc_duration(self.trapA)*2)/self.system.grad_raster_time)*self.system.grad_raster_time)


            self.gDiffA_x = copy.deepcopy(self.trapA)
            self.gDiffA_x.amplitude *= diff_vec[0]
            self.gDiffA_y = copy.deepcopy(self.trapA)
            self.gDiffA_y.amplitude *= diff_vec[1]
            self.gDiffA_z = copy.deepcopy(self.trapA)
            self.gDiffA_z.amplitude *= diff_vec[2]

            self.gDiffA_x.channel = 'x'
            self.gDiffA_y.channel = 'y'
            self.gDiffA_z.channel = 'z'

            self.gDiffB_x = copy.deepcopy(self.gDiffA_x)
            self.gDiffB_x.amplitude *= -1
            self.gDiffB_y = copy.deepcopy(self.gDiffA_y)
            self.gDiffB_y.amplitude *= -1
            self.gDiffB_z = copy.deepcopy(self.gDiffA_z)
            self.gDiffB_z.amplitude  *= -1

        if self.timings['MMT'] == 2:
            amplitude = pp.convert.convert(from_value =self.trap_params['Amplitude']*1e3, from_unit = 'mT/m', to_unit = 'Hz/m')
            self.trapA = pp.make_trapezoid(channel='z',system=self.system, amplitude = amplitude,rise_time = self.trap_params['rise_time'],
                                        flat_time = self.trap_params['trapA_flat'],fall_time = self.trap_params['rise_time']) 


            self.trapB = pp.make_trapezoid(channel='z',system=self.system, amplitude = amplitude,rise_time = self.trap_params['rise_time'],
                                        flat_time = self.trap_params['trapB_flat'],fall_time = self.trap_params['rise_time']) 

            self.gap = pp.make_delay(np.ceil(self.trap_params['gap']/self.system.grad_raster_time)*self.system.grad_raster_time)
            
            sequences = self.find_zero_sequences(self.trap_params['Gradient'])
            TT_inv = self.trap_params['TE']/2*1e-3
            index_inversion = np.argmin(np.abs(self.trap_params['Time'] - TT_inv))
            start_index_gapP180 = sequences[2][0]
            zeros_to_inv = index_inversion- start_index_gapP180
            zeros_to_inv_time = zeros_to_inv * (self.trap_params['Time'][1]-self.trap_params['Time'][0])

            gap_to_180 = zeros_to_inv_time  - self.rf_180['center_incl_delay']
            gap_post_180 = self.trap_params['gap_p_180'] - (gap_to_180 + self.rf_180['timing'])
            if np.ceil(gap_to_180/self.system.grad_raster_time)*self.system.grad_raster_time < 0:
                gap_to_180 = 0#abs(gap_to_180)
                gap_post_180 += gap_to_180
            self.delayPre180 = pp.make_delay(np.ceil(gap_to_180/self.system.grad_raster_time)*self.system.grad_raster_time)
            self.delayPost180 =  pp.make_delay(np.ceil(gap_post_180/self.system.grad_raster_time)*self.system.grad_raster_time)

            self.gDiffA_x = copy.deepcopy(self.trapA)
            self.gDiffA_x.amplitude *= diff_vec[0]
            self.gDiffA_y = copy.deepcopy(self.trapA)
            self.gDiffA_y.amplitude *= diff_vec[1]
            self.gDiffA_z = copy.deepcopy(self.trapA)
            self.gDiffA_z.amplitude *= diff_vec[2]

            self.gDiffA_x.channel = 'x'
            self.gDiffA_y.channel = 'y'
            self.gDiffA_z.channel = 'z'


            self.gDiffB_x = copy.deepcopy(self.trapB)
            self.gDiffB_x.amplitude *= -1*diff_vec[0]
            self.gDiffB_y = copy.deepcopy(self.trapB)
            self.gDiffB_y.amplitude *= -1*diff_vec[1]
            self.gDiffB_z =copy.deepcopy(self.trapB)
            self.gDiffB_z.amplitude *= -1*diff_vec[2]

            self.gDiffB_x.channel = 'x'
            self.gDiffB_y.channel = 'y'
            self.gDiffB_z.channel = 'z'

        if self.timings['MMT'] == 3:

            amplitudeA = pp.convert.convert(from_value =self.trap_params['G1']*1e3, from_unit = 'mT/m', to_unit = 'Hz/m')
            amplitudeB = pp.convert.convert(from_value =self.trap_params['G2']*1e3, from_unit = 'mT/m', to_unit = 'Hz/m')
            amplitudeC = pp.convert.convert(from_value =self.trap_params['G3']*1e3, from_unit = 'mT/m', to_unit = 'Hz/m')

            print('AmplitudeA',self.trap_params['G1']*1e3,'AmplitudeB',self.trap_params['G2']*1e3, self.trap_params['G3']*1e3)
            
            
            self.trapA = pp.make_trapezoid(channel='z',system=self.system, amplitude = amplitudeA,rise_time = self.trap_params['rise_time'],
                                        flat_time = self.trap_params['trapA_flat'],fall_time = self.trap_params['rise_time']) 
            

            self.trapB = pp.make_trapezoid(channel='z',system=self.system, amplitude = -amplitudeB,rise_time = self.trap_params['rise_time'],
                                        flat_time = self.trap_params['trapA_flat'],fall_time = self.trap_params['rise_time']) 
            

            self.trapC = pp.make_trapezoid(channel='z',system=self.system, amplitude = amplitudeC,rise_time = self.trap_params['rise_time'],
                                        flat_time = self.trap_params['trapA_flat'],fall_time = self.trap_params['rise_time']) 
                        

            self.gap = pp.make_delay(np.ceil(self.trap_params['gap']/self.system.grad_raster_time)*self.system.grad_raster_time)
            
            
            self.gDiffA_x = copy.deepcopy(self.trapA)
            self.gDiffA_x.amplitude *= diff_vec[0]
            self.gDiffA_y = copy.deepcopy(self.trapA)
            self.gDiffA_y.amplitude *= diff_vec[1]
            self.gDiffA_z = copy.deepcopy(self.trapA)
            self.gDiffA_z.amplitude *= diff_vec[2]

            self.gDiffA_x.channel = 'x'
            self.gDiffA_y.channel = 'y'
            self.gDiffA_z.channel = 'z'


            self.gDiffB_x = copy.deepcopy(self.trapB)
            self.gDiffB_x.amplitude *= -1*diff_vec[0]
            self.gDiffB_y = copy.deepcopy(self.trapB)
            self.gDiffB_y.amplitude *= -1*diff_vec[1]
            self.gDiffB_z =copy.deepcopy(self.trapB)
            self.gDiffB_z.amplitude *= -1*diff_vec[2]

            self.gDiffB_x.channel = 'x'
            self.gDiffB_y.channel = 'y'
            self.gDiffB_z.channel = 'z'

            self.gDiffC_x = copy.deepcopy(self.trapC)
            self.gDiffC_x.amplitude *= -1*diff_vec[0]
            self.gDiffC_y = copy.deepcopy(self.trapC)
            self.gDiffC_y.amplitude *= -1*diff_vec[1]
            self.gDiffC_z =copy.deepcopy(self.trapC)
            self.gDiffC_z.amplitude *= -1*diff_vec[2]

            self.gDiffC_x.channel = 'x'
            self.gDiffC_y.channel = 'y'
            self.gDiffC_z.channel = 'z'


        

    


        








        

    def asymm(self,gmax = 200,smax = 200,tol=0.01):
        # Define waveform parameters
        T_90 = self.timings['T_90'] * 1e-3
        T_180 = self.timings['T_180'] * 1e-3
        dt = self.timings['dt']
        Gmax = gmax / 1000
        Smax = smax / 1000
        GAM = 2 * np.pi * 42.58e3
        zeta = (Gmax / Smax) * 1e-3
        target_bval = self.timings['b']
        b = 0
        flat = 0e-3
        motion_comp = self.timings['MMT']
        prev_t = np.array([0])
        prev_g = np.array([0])

        while flat <= 100000000e-3 :
            if 2*zeta+ flat < T_180:
                diff = T_180 - 2*zeta
                flat = flat+diff

            
            if motion_comp == 2: 
                
                t = []
                g = []
                gap = self.timings['T_readout'] * 1e-3 - self.rf_90['center_incl_delay'] #T_90/2 
                
                # Add in the time spent to slew up 
                gap_p_180=  2*zeta + flat
                
                flat_time = 2*flat + zeta #2*flat + zeta #3*zeta + 2*flat #flat*2 + zeta 
                intervals = [T_90, gap, zeta, flat, zeta, zeta,flat,flat, zeta, T_180, zeta, flat,flat, zeta,zeta,flat,zeta]
                
                time = [
                    np.linspace(0, T_90 + gap, int((T_90 + gap) / dt)),
                    np.linspace(T_90 + gap, T_90 + gap + zeta, int(zeta / dt)),
                    np.linspace(T_90 + gap + zeta, T_90 + gap + zeta + flat, int(flat / dt)),
                    np.linspace(T_90 + gap + zeta + flat, T_90 + gap + zeta + flat + zeta, int(zeta / dt)),
                    np.linspace(T_90 + gap + zeta + flat + zeta, T_90 + gap + zeta + flat + zeta + zeta, int(zeta / dt)),
                    np.linspace(T_90 + gap + zeta + flat + zeta + zeta, T_90 + gap + zeta + flat + zeta + zeta + flat_time, int(flat_time / dt)),
                    np.linspace(T_90 + gap + zeta + flat + zeta + zeta + flat_time, T_90 + gap + zeta + flat + zeta + zeta + flat_time + zeta, int(zeta / dt)),
                    np.linspace(T_90 + gap + zeta + flat + zeta + zeta + flat_time + zeta, T_90 + gap + zeta + flat + zeta + zeta + flat_time + zeta + gap_p_180, int(gap_p_180 / dt)),
                    np.linspace(T_90 + gap + zeta + flat + zeta + zeta + flat_time + zeta + gap_p_180, T_90 + gap + zeta + flat + zeta + zeta + flat_time + zeta + gap_p_180 + zeta, int(zeta / dt)),
                    np.linspace(T_90 + gap + zeta + flat + zeta + zeta + flat_time + zeta + gap_p_180 + zeta, T_90 + gap + zeta + flat + zeta + zeta + flat_time + zeta + gap_p_180 + zeta + flat_time, int(flat_time / dt)),
                    np.linspace(T_90 + gap + zeta + flat + zeta + zeta + flat_time + zeta + gap_p_180 + zeta + flat_time, T_90 + gap + zeta + flat + zeta + zeta + flat_time + zeta + gap_p_180 + zeta + flat_time + zeta, int(zeta / dt)),
                    np.linspace(T_90 + gap + zeta + flat + zeta + zeta + flat_time + zeta + gap_p_180 + zeta + flat_time + zeta, T_90 + gap + zeta + flat + zeta + zeta + flat_time + zeta + gap_p_180 + zeta + flat_time + zeta + zeta, int(zeta / dt)),
                    np.linspace(T_90 + gap + zeta + flat + zeta + zeta + flat_time + zeta + gap_p_180 + zeta + flat_time + zeta + zeta, T_90 + gap + zeta + flat + zeta + zeta + flat_time + zeta + gap_p_180 + zeta + flat_time + zeta + zeta + flat, int(flat / dt)),
                    np.linspace(T_90 + gap + zeta + flat + zeta + zeta + flat_time + zeta + gap_p_180 + zeta + flat_time + zeta + zeta + flat, T_90 + gap + zeta + flat + zeta + zeta + flat_time + zeta + gap_p_180 + zeta + flat_time + zeta + zeta + flat + zeta, int(zeta / dt)),
                ]
                gradient = [
                    np.zeros_like(time[0]),
                    np.linspace(0,Gmax,num = len(time[1])),
                    Gmax*np.ones_like(time[2]),
                    np.linspace(Gmax,0,num = len(time[3])),
                    np.linspace(0,-Gmax,num = len(time[4])),
                    -Gmax*np.ones_like(time[5]),
                    np.linspace(-Gmax,0,num = len(time[6])),
                    np.zeros_like(time[7]),
                    np.linspace(0,-Gmax,num = len(time[8])),
                    -Gmax*np.ones_like(time[9]),
                    np.linspace(-Gmax,0,num = len(time[10])),
                    np.linspace(0,Gmax,num = len(time[11])),
                    Gmax*np.ones_like(time[12]),
                    np.linspace(Gmax,0,num = len(time[13])),
                    ]

                
                t = np.concatenate(time,axis = 0)
                g = np.concatenate(gradient,axis = 0)
                

            else: 
                print('Invalid Motion Compensation Parameter')
                break

            TE = t[-1] * 1e3 + self.timings['T_readout']
            self.timings['TE'] = TE

            
            b = get_bval(g[np.newaxis,:],self.timings)

            if (np.abs(b - target_bval) <= target_bval*tol): # Stop if the calculated b-value is within 1% of the target
                break
            
                
            else:
                flat += dt
            prev_t = t
            prev_g = g
            #flat += dt   
            waveform = {'Gradient':g,
                        'Time':t,
                        'TE':TE,
                        'b':b,
                        'rise_time':np.ceil(zeta/self.system.grad_raster_time)*self.system.grad_raster_time,
                        'trapA_flat':np.ceil(flat/self.system.grad_raster_time)*self.system.grad_raster_time,
                        'trapB_flat': np.ceil(flat_time/self.system.grad_raster_time)*self.system.grad_raster_time,
                        'gap': np.ceil(gap/self.system.grad_raster_time)*self.system.grad_raster_time,
                        'gap_p_180':gap_p_180,
                        'Amplitude': Gmax,
                        }
        return waveform

            

    def monopolar(self,gmax = 200,smax = 200,tol=0.01):
        # Define waveform parameters
        T_90 = self.timings['T_90'] * 1e-3
        T_180 = self.timings['T_180'] * 1e-3
        dt = self.timings['dt']
        Gmax = gmax / 1000
        Smax = smax / 1000
        GAM = 2 * np.pi * 42.58e3
        zeta = (Gmax / Smax) * 1e-3
        target_bval = self.timings['b']
        b = 0
        flat = 0e-3
        motion_comp = self.timings['MMT']
        
        while flat <= 100000000e-3 :
            
            if motion_comp == 0: 
                
                t = []
                g = []
                offset_90 = self.rf_90['center_incl_delay']
                gap_p_180 = self.timings['T_readout'] * 1e-3 - offset_90 + T_180
                gap = self.timings['T_readout'] * 1e-3 - offset_90
                intervals = [T_90, zeta, flat, zeta, gap_p_180, zeta, flat, zeta]


                time = [
                    np.linspace(0, T_90, int(T_90/dt)),
                    np.linspace(T_90, T_90+zeta, int(zeta/dt)),
                    np.linspace(T_90+zeta, T_90+zeta+flat, int(flat/dt)),
                    np.linspace(T_90+zeta+flat, T_90+zeta+flat+zeta, int(zeta/dt)),
                    np.linspace(T_90+zeta+flat+zeta, T_90+zeta+flat+zeta+gap_p_180, int(gap_p_180/dt)),
                    np.linspace(T_90+zeta+flat+zeta+gap_p_180, T_90+zeta+flat+zeta+gap_p_180+zeta, int(zeta/dt)),
                    np.linspace(T_90+zeta+flat+zeta+gap_p_180+zeta, T_90+zeta+flat+zeta+gap_p_180+zeta+flat, int(flat/dt)),
                    np.linspace(T_90+zeta+flat+zeta+gap_p_180+zeta+flat, T_90+zeta+flat+zeta+gap_p_180+zeta+flat+zeta, int(zeta/dt)),
                ]

                gradient = [
                    np.zeros_like(time[0]),
                    np.linspace(0,Gmax,num = len(time[1])),
                    Gmax*np.ones_like(time[2]),
                    np.linspace(Gmax,0,num = len(time[3])),
                    np.zeros_like(time[4]),
                    np.linspace(0,Gmax,num = len(time[5])),
                    Gmax*np.ones_like(time[6]),
                    np.linspace(Gmax,0,num = len(time[7])),
            
                    ]
                t = np.concatenate(time,axis = 0)
                g = np.concatenate(gradient,axis = 0)

            else: 
                print('Invalid Motion Compensation Parameter')
                break

            TE = t[-1] * 1e3 + self.timings['T_readout']
            self.timings['TE'] = TE
            delta = flat + zeta
            
            Delta = self.timings['TE'] - T_90/2 - delta - zeta - self.timings['T_readout']*1e-3
            b = get_bval(g[np.newaxis,:],self.timings)

            if (np.abs(b - target_bval) <= target_bval*tol): # Stop if the calculated b-value is within 1% of the target
                break
            elif b > (target_bval + target_bval*tol) and flat > 0 :
                print('\tNot within tolerance but bval exceeded: bval = {:.2f},flat = {:.2f}'.format(b,flat))
                prev_bval = get_bval(prev_g[np.newaxis,:-1],self.timings)
                if prev_bval - target_bval < b- target_bval:
                    g=prev_g
                    t=prev_t
                break
            flat +=dt
            prev_t = t
            prev_g = g
            waveform = {'Gradient':g,
                        'Time':t,
                        'TE':TE,
                        'b':b,
                        'rise_time': np.ceil(zeta/self.system.grad_raster_time)*self.system.grad_raster_time,
                        'trapA_flat': np.ceil(flat/self.system.grad_raster_time)*self.system.grad_raster_time,
                        'trapB_flat': None,
                        'gap': np.ceil(gap/self.system.grad_raster_time)*self.system.grad_raster_time,
                        'gap_p_180':gap_p_180,
                        'Amplitude': Gmax,
                        }
        return waveform

    def bipolar(self,gmax = 200,smax = 200,tol=0.01):
        T_90 = self.timings['T_90'] * 1e-3
        T_180 = self.timings['T_180'] * 1e-3
        dt = self.timings['dt']
        Gmax = gmax / 1000
        Smax = smax / 1000
        GAM = 2 * np.pi * 42.58e3
        zeta = (Gmax / Smax) * 1e-3
        target_bval = self.timings['b']
        b = 0
        flat = 0e-3
        motion_comp = self.timings['MMT']
        
        while flat <= 100000e-3 :
            
            if motion_comp == 1: 
                T_90 = self.timings['T_90'] * 1e-3
                
                t = []
                g = []
                offset_90 = self.rf_90['center_incl_delay']
                
                gap = self.timings['T_readout'] * 1e-3 - offset_90

                #T_90 +=gap
                gap_p_180 =  T_180 + gap
                intervals = [T_90, zeta, flat, zeta, zeta, flat, zeta, gap_p_180, zeta, flat, zeta,zeta, flat, zeta,]


                time = [
                    np.linspace(0, T_90, int(T_90/dt)),
                    np.linspace(T_90, T_90+zeta, int(zeta/dt)),
                    np.linspace(T_90+zeta, T_90+zeta+flat, int(flat/dt)),
                    np.linspace(T_90+zeta+flat, T_90+zeta+flat+zeta, int(zeta/dt)),
                    np.linspace(T_90+zeta+flat+zeta, T_90+zeta+flat+zeta+zeta, int(zeta/dt)),
                    np.linspace(T_90+zeta+flat+zeta+zeta, T_90+zeta+flat+zeta+zeta+flat, int(flat/dt)),
                    np.linspace(T_90+zeta+flat+zeta+zeta+flat, T_90+zeta+flat+zeta+zeta+flat+zeta, int(zeta/dt)),
                    np.linspace(T_90+zeta+flat+zeta+zeta+flat+zeta, T_90+zeta+flat+zeta+zeta+flat+zeta+gap_p_180, int(gap_p_180/dt)),
                    np.linspace(T_90+zeta+flat+zeta+zeta+flat+zeta+gap_p_180, T_90+zeta+flat+zeta+zeta+flat+zeta+gap_p_180+zeta, int(zeta/dt)),
                    np.linspace(T_90+zeta+flat+zeta+zeta+flat+zeta+gap_p_180+zeta, T_90+zeta+flat+zeta+zeta+flat+zeta+gap_p_180+zeta+flat, int(flat/dt)),
                    np.linspace(T_90+zeta+flat+zeta+zeta+flat+zeta+gap_p_180+zeta+flat, T_90+zeta+flat+zeta+zeta+flat+zeta+gap_p_180+zeta+flat+zeta, int(zeta/dt)),
                    np.linspace(T_90+zeta+flat+zeta+zeta+flat+zeta+gap_p_180+zeta+flat+zeta, T_90+zeta+flat+zeta+zeta+flat+zeta+gap_p_180+zeta+flat+zeta+zeta, int(zeta/dt)),
                    np.linspace(T_90+zeta+flat+zeta+zeta+flat+zeta+gap_p_180+zeta+flat+zeta+zeta, T_90+zeta+flat+zeta+zeta+flat+zeta+gap_p_180+zeta+flat+zeta+zeta+flat, int(flat/dt)),
                    np.linspace(T_90+zeta+flat+zeta+zeta+flat+zeta+gap_p_180+zeta+flat+zeta+zeta+flat, T_90+zeta+flat+zeta+zeta+flat+zeta+gap_p_180+zeta+flat+zeta+zeta+flat+zeta, int(zeta/dt)),
                
                    
                ]

                gradient = [
                    np.zeros_like(time[0]),
                    np.linspace(0,Gmax,num = len(time[1])),
                    Gmax*np.ones_like(time[2]),
                    np.linspace(Gmax,0,num = len(time[3])),
                    
                    np.linspace(0,-Gmax,num = len(time[4])),
                    -Gmax*np.ones_like(time[5]),
                    np.linspace(-Gmax,0,num = len(time[6])),
                    
                    np.zeros_like(time[7]),

                    
                    np.linspace(0,Gmax,num = len(time[8])),
                    Gmax*np.ones_like(time[9]),
                    np.linspace(Gmax,0,num = len(time[10])),

                    np.linspace(0,-Gmax,num = len(time[11])),
                    -Gmax*np.ones_like(time[12]),
                    np.linspace(-Gmax,0,num = len(time[13])),
            
                    ]
                t = np.concatenate(time,axis = 0)
                g = np.concatenate(gradient,axis = 0)

            else: 
                print('Invalid Motion Compensation Parameter')
                break

            TE = t[-1] * 1e3 + self.timings['T_readout']
            self.timings['TE'] = TE
            delta = flat + zeta
            
            Delta = self.timings['TE'] - T_90/2 - delta - zeta - self.timings['T_readout']*1e-3
            b = get_bval(g[np.newaxis,:],self.timings)

            if (np.abs(b - target_bval) <= target_bval*tol): # Stop if the calculated b-value is within 1% of the target
                break
            elif b > (target_bval + target_bval*tol) and flat > 0 :
                print('\tNot within tolerance but bval exceeded: bval = {:.2f},flat = {:.2f}'.format(b,flat))
                prev_bval = get_bval(prev_g[np.newaxis,:-1],self.timings)
                if prev_bval - target_bval < b- target_bval:
                    g=prev_g
                    t=prev_t
                break
            flat +=dt
            prev_t = t
            prev_g = g
            waveform = {'Gradient':g,
                        'Time':t,
                        'TE':TE,
                        'b':b,
                        'rise_time': np.ceil(zeta/self.system.grad_raster_time)*self.system.grad_raster_time,
                        'trapA_flat': np.ceil(flat/self.system.grad_raster_time)*self.system.grad_raster_time,
                        'trapB_flat': None,
                        'gap': np.ceil(gap/self.system.grad_raster_time)*self.system.grad_raster_time,
                        'gap_p_180':gap_p_180,
                        'Amplitude': Gmax,
                        }
        return waveform
        





    
    
    def find_zero_sequences(self,arr):
        sequences = []
        i = 0
        while i < len(arr):
            if arr[i] == 0:
                start_index = i
                length = 0
                while i < len(arr) and arr[i] == 0:
                    length += 1
                    i += 1
                sequences.append((start_index, length))
            else:
                i += 1
        return sequences
    
    def get_moments1(self,G, T_readout, dt):
        G=np.squeeze(G)
        TE = G.size*dt*1e3 + T_readout
        tINV = int(np.floor(TE/dt/1.0e3/2.0))
        GAMMA   = 42.58e3; 
        INV = np.ones(G.size)
        INV[tINV:] = -1
        Nm = 5
        tvec = np.arange(G.size)*dt
        tMat = np.zeros((Nm, G.size))
        for mm in range(Nm):
            tMat[mm] = tvec**mm

        moments = np.abs(GAMMA*dt*tMat@(G*INV))
        mm = GAMMA*dt*tMat * (G*INV)[np.newaxis,:]

        mmt = np.cumsum(mm[0])
        m0=mmt/np.abs(mmt).max()
        mmt = np.cumsum(mm[1])
        m1=mmt/np.abs(mmt).max()
        mmt = np.cumsum(mm[2])
        m2=mmt/np.abs(mmt).max()
        mmt = np.cumsum(mm[3])
        m3=mmt/np.abs(mmt).max()

        return m0,m1,m2, m3
    

    def m3(self,gmax = 200,smax = 200,tol=0.01):
        # Define waveform parameters
        T_90 = self.timings['T_90'] * 1e-3
        T_180 = self.timings['T_180'] * 1e-3
        dt = self.timings['dt']
        Gmax = gmax / 1000
        Smax = smax / 1000
        GAM = 2 * np.pi * 42.58e3
        zeta = (Gmax / Smax) * 1e-3
        target_bval = self.timings['b']
        b = 0
        flat = 0e-3
        motion_comp = self.timings['MMT']
        prev_t = np.array([0])
        prev_g = np.array([0])

        while flat <= 10000000e-3 :
            #if 2*zeta+ flat < T_180:
            #    diff = T_180 - 2*zeta
            #    flat = flat+diff


            
            
            if motion_comp == 3: 

                
                
                t = []
                g = []
                gap = self.timings['T_readout'] * 1e-3 - self.rf_90['center_incl_delay'] #T_90/2 
                # Add in the time spent to slew up 
                gap_p_180=  T_180
                
                flat_time = flat #2*flat + zeta #3*zeta + 2*flat #flat*2 + zeta 
                intervals = [T_90, gap, zeta, flat, zeta, zeta,flat,flat, zeta, T_180, zeta, flat,flat, zeta,zeta,flat,zeta]
                
                time = [
                    np.linspace(0, T_90 + gap, int((T_90 + gap) / dt)),
                    np.linspace(T_90 + gap, T_90 + gap + zeta, int(zeta / dt)),
                    np.linspace(T_90 + gap + zeta, T_90 + gap + zeta + flat, int(flat / dt)),
                    np.linspace(T_90 + gap + zeta + flat, T_90 + gap + zeta + flat + zeta, int(zeta / dt)),
                    np.linspace(T_90 + gap + zeta + flat + zeta, T_90 + gap + zeta + flat + zeta + zeta, int(zeta / dt)),
                    np.linspace(T_90 + gap + zeta + flat + zeta + zeta, T_90 + gap + zeta + flat + zeta + zeta + flat, int(flat / dt)),
                    np.linspace(T_90 + gap + zeta + flat + zeta + zeta + flat, T_90 + gap + zeta + flat + zeta + zeta + flat + zeta, int(zeta / dt)),
                    np.linspace(T_90 + gap + zeta + flat + zeta + zeta + flat + zeta, T_90 + gap + zeta + flat + zeta + zeta + flat + zeta + zeta, int(zeta / dt)),
                    np.linspace(T_90 + gap + zeta + flat + zeta + zeta + flat + zeta + zeta, T_90 + gap + zeta + flat + zeta + zeta + flat + zeta + zeta + flat, int(flat / dt)),
                    np.linspace(T_90 + gap + zeta + flat + zeta + zeta + flat + zeta + zeta + flat, T_90 + gap + zeta + flat + zeta + zeta + flat + zeta + zeta + flat + zeta, int(zeta / dt)),
                    np.linspace(T_90 + gap + zeta + flat + zeta + zeta + flat + zeta + zeta + flat + zeta, T_90 + gap + zeta + flat + zeta + zeta + flat + zeta + zeta + flat + zeta + gap_p_180, int(gap_p_180 / dt)),
                    np.linspace(T_90 + gap + zeta + flat + zeta + zeta + flat + zeta + zeta + flat + zeta + gap_p_180, T_90 + gap + zeta + flat + zeta + zeta + flat + zeta + zeta + flat + zeta + gap_p_180 + zeta, int(zeta / dt)),
                    np.linspace(T_90 + gap + zeta + flat + zeta + zeta + flat + zeta + zeta + flat + zeta + gap_p_180 + zeta, T_90 + gap + zeta + flat + zeta + zeta + flat + zeta + zeta + flat + zeta + gap_p_180 + zeta + flat, int(flat / dt)),
                    np.linspace(T_90 + gap + zeta + flat + zeta + zeta + flat + zeta + zeta + flat + zeta + gap_p_180 + zeta + flat, T_90 + gap + zeta + flat + zeta + zeta + flat + zeta + zeta + flat + zeta + gap_p_180 + zeta + flat + zeta, int(zeta / dt)),
                    np.linspace(T_90 + gap + zeta + flat + zeta + zeta + flat + zeta + zeta + flat + zeta + gap_p_180 + zeta + flat + zeta, T_90 + gap + zeta + flat + zeta + zeta + flat + zeta + zeta + flat + zeta + gap_p_180 + zeta + flat + zeta + zeta, int(zeta / dt)),
                    np.linspace(T_90 + gap + zeta + flat + zeta + zeta + flat + zeta + zeta + flat + zeta + gap_p_180 + zeta + flat + zeta + zeta, T_90 + gap + zeta + flat + zeta + zeta + flat + zeta + zeta + flat + zeta + gap_p_180 + zeta + flat + zeta + zeta + flat, int(flat / dt)),
                    np.linspace(T_90 + gap + zeta + flat + zeta + zeta + flat + zeta + zeta + flat + zeta + gap_p_180 + zeta + flat + zeta + zeta + flat, T_90 + gap + zeta + flat + zeta + zeta + flat + zeta + zeta + flat + zeta + gap_p_180 + zeta + flat + zeta + zeta + flat + zeta, int(zeta / dt)),
                    np.linspace(T_90 + gap + zeta + flat + zeta + zeta + flat + zeta + zeta + flat + zeta + gap_p_180 + zeta + flat + zeta + zeta + flat + zeta, T_90 + gap + zeta + flat + zeta + zeta + flat + zeta + zeta + flat + zeta + gap_p_180 + zeta + flat + zeta + zeta + flat + zeta + zeta, int(zeta / dt)),
                    np.linspace(T_90 + gap + zeta + flat + zeta + zeta + flat + zeta + zeta + flat + zeta + gap_p_180 + zeta + flat + zeta + zeta + flat + zeta + zeta, T_90 + gap + zeta + flat + zeta + zeta + flat + zeta + zeta + flat + zeta + gap_p_180 + zeta + flat + zeta + zeta + flat + zeta + zeta + flat, int(flat / dt)),
                    np.linspace(T_90 + gap + zeta + flat + zeta + zeta + flat + zeta + zeta + flat + zeta + gap_p_180 + zeta + flat + zeta + zeta + flat + zeta + zeta + flat, T_90 + gap + zeta + flat + zeta + zeta + flat + zeta + zeta + flat + zeta + gap_p_180 + zeta + flat + zeta + zeta + flat + zeta + zeta + flat + zeta, int(zeta / dt)),
                    
                    ]
                
                delta = 2*zeta + flat 
                Delta = gap_p_180 + delta*3
                G2 = Gmax
                G1 = G2 *  (Delta - delta) / (2*Delta) 
                G3 = G2-G1

                
                
                gradient = [
                    np.zeros_like(time[0]),
                    np.linspace(0,G1,num = len(time[1])),
                    G1*np.ones_like(time[2]),
                    np.linspace(G1,0,num = len(time[3])),

                    np.linspace(0,-G2,num = len(time[4])),
                    -G2*np.ones_like(time[5]),
                    np.linspace(-G2,0,num = len(time[6])),

                    np.linspace(0,G3,num = len(time[7])),
                    G3*np.ones_like(time[8]),
                    np.linspace(G3,0,num = len(time[9])),

                    np.zeros_like(time[10]),


                    np.linspace(0,-G3,num = len(time[11])),
                    -G3*np.ones_like(time[12]),
                    np.linspace(-G3,0,num = len(time[13])),

                    np.linspace(0,G2,num = len(time[14])),
                    G2*np.ones_like(time[15]),
                    np.linspace(G2,0,num = len(time[16])),

                    np.linspace(0,-G1,num = len(time[17])),
                    -G1*np.ones_like(time[18]),
                    np.linspace(-G1,0,num = len(time[19])),

                    ]

         
                t = np.concatenate(time,axis = 0)
                g = np.concatenate(gradient,axis = 0)

            else: 
                print('Invalid Motion Compensation Parameter')
                break

            TE = t[-1] * 1e3 + self.timings['T_readout']
            self.timings['TE'] = TE

            
            b = get_bval(g[np.newaxis,:],self.timings)

            if (np.abs(b - target_bval) <= target_bval*tol): # Stop if the calculated b-value is within 1% of the target
                break
            
                
            else:
                flat += dt
            prev_t = t
            prev_g = g
            #flat += dt   
            waveform = {'Gradient':g,
                        'Time':t,
                        'TE':TE,
                        'b':b,
                        'rise_time':np.ceil(zeta/self.system.grad_raster_time)*self.system.grad_raster_time,
                        'trapA_flat':np.ceil(flat/self.system.grad_raster_time)*self.system.grad_raster_time,
                        'trapB_flat': np.ceil(flat_time/self.system.grad_raster_time)*self.system.grad_raster_time,
                        'gap': np.ceil(gap/self.system.grad_raster_time)*self.system.grad_raster_time,
                        'gap_p_180':gap_p_180,
                        'Amplitude': Gmax,
                        'delta':delta,
                        'Delta':Delta,
                        'G1':G1,
                        'G2':G2,
                        'G3':G3,
                        }
        return waveform




        
         





        
         
