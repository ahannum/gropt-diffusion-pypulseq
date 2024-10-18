import numpy as np 
import sys
sys.path.append("/Users/arielhannum/Documents/GitHub/gropt/python")
import gropt
from helper_utils import *


def monopolar(params, tol):
    # Define waveform parameters
    T_90 = params['T_90'] * 1e-3
    T_180 = params['T_180'] * 1e-3
    dt = params['dt']
    Gmax = params['gmax'] / 1000
    Smax = params['smax'] / 1000
    GAM = 2 * np.pi * 42.58e3
    zeta = (Gmax / Smax) * 1e-3
    target_bval = params['b']
    b = 0
    flat = 0e-3
    motion_comp = params['MMT']
    
    while flat <= 100e-3 :
        
        if motion_comp == 0: 
            
            t = []
            g = []
            gap_p_180 = params['T_readout'] * 1e-3 - T_90/2 + T_180
            intervals = [T_90, zeta, flat, zeta, gap_p_180, zeta, flat, zeta]


            time = [
                    np.arange(0,T_90,dt),
                    np.arange(T_90,T_90+zeta,dt),
                    np.arange(T_90+zeta,T_90+zeta+flat,dt),
                    np.arange(T_90+zeta+flat,T_90+zeta+flat+zeta,dt),
                    np.arange(T_90+zeta+flat+zeta,T_90+zeta+flat+zeta+gap_p_180,dt),
                    np.arange(T_90+zeta+flat+zeta+gap_p_180,T_90+zeta+flat+zeta+gap_p_180+zeta,dt),
                    np.arange(T_90+zeta+flat+zeta+gap_p_180+zeta,T_90+zeta+flat+zeta+gap_p_180+zeta+flat,dt),
                    np.arange(T_90+zeta+flat+zeta+gap_p_180+zeta+flat,T_90+zeta+flat+zeta+gap_p_180+zeta+flat+zeta,dt),
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

        TE = t[-1] * 1e3 + params['T_readout']
        params['TE'] = TE*1e-3
        delta = flat + zeta
        
        Delta = params['TE'] - T_90/2 - delta - zeta - params['T_readout']*1e-3
        b = get_bval(g,params)

        if (np.abs(b - target_bval) <= target_bval*tol): # Stop if the calculated b-value is within 1% of the target
            break
        elif b > (target_bval + target_bval*tol) and flat > 0 :
            print('\tNot within tolerance but bval exceeded: bval = {:.2f},flat = {:.2f}'.format(b,flat))
            prev_bval = get_bval(g[:-1],params)
            if prev_bval - target_bval < b- target_bval:
                g=prev_g
                t=prev_t
            break
        flat +=dt
        prev_t = t
        prev_g = g
    return g,t,TE,b,
    

def bipolar(params, tol):
    # Define waveform parameters
    T_90 = params['T_90'] * 1e-3
    T_180 = params['T_180'] * 1e-3
    dt = params['dt']
    Gmax = params['gmax'] / 1000
    Smax = params['smax'] / 1000
    GAM = 2 * np.pi * 42.58e3
    zeta = (Gmax / Smax) * 1e-3
    target_bval = params['b']
    b = 0
    flat = 0e-3
    motion_comp = params['MMT']
    prev_t = np.array([0])
    prev_g = np.array([0])

    while flat <= 100e-3 :
        
        if motion_comp == 1: 
            
            t = []
            g = []
            gap = params['T_readout'] * 1e-3 - T_90/2 
            intervals = [T_90, gap, zeta, flat, zeta, zeta,flat,zeta, T_180, zeta, flat, zeta,zeta,flat,zeta]

            time = [
                np.arange(0,T_90+gap,dt),
                np.arange(T_90+gap,T_90+gap+zeta,dt),
                np.arange(T_90+gap+zeta,T_90+gap+zeta+flat,dt),
                np.arange(T_90+gap+zeta+flat,T_90+gap+zeta+flat+zeta,dt),
                np.arange(T_90+gap+zeta+flat+zeta,T_90+gap+zeta+flat+zeta+zeta,dt),
                np.arange(T_90+gap+zeta+flat+zeta+zeta,T_90+gap+zeta+flat+zeta+zeta+flat,dt),
                np.arange(T_90+gap+zeta+flat+zeta+zeta+flat,T_90+gap+zeta+flat+zeta+zeta+flat+zeta,dt),
                np.arange(T_90+gap+zeta+flat+zeta+zeta+flat+zeta,T_90+gap+zeta+flat+zeta+zeta+flat+zeta+T_180,dt),
                np.arange(T_90+gap+zeta+flat+zeta+zeta+flat+zeta+T_180,T_90+gap+zeta+flat+zeta+zeta+flat+zeta+T_180+zeta,dt),
                np.arange(T_90+gap+zeta+flat+zeta+zeta+flat+zeta+T_180+zeta,T_90+gap+zeta+flat+zeta+zeta+flat+zeta+T_180+zeta+flat,dt),
                np.arange(T_90+gap+zeta+flat+zeta+zeta+flat+zeta+T_180+zeta+flat,T_90+gap+zeta+flat+zeta+zeta+flat+zeta+T_180+zeta+flat+zeta,dt),
                np.arange(T_90+gap+zeta+flat+zeta+zeta+flat+zeta+T_180+zeta+flat+zeta,T_90+gap+zeta+flat+zeta+zeta+flat+zeta+T_180+zeta+flat+zeta+zeta,dt),
                np.arange(T_90+gap+zeta+flat+zeta+zeta+flat+zeta+T_180+zeta+flat+zeta+zeta,T_90+gap+zeta+flat+zeta+zeta+flat+zeta+T_180+zeta+flat+zeta+zeta+flat,dt),
                np.arange(T_90+gap+zeta+flat+zeta+zeta+flat+zeta+T_180+zeta+flat+zeta+zeta+flat,T_90+gap+zeta+flat+zeta+zeta+flat+zeta+T_180+zeta+flat+zeta+zeta+flat+zeta,dt),

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

        TE = t[-1] * 1e3 + params['T_readout']
        params['TE'] = TE*1e-3
        delta = flat + zeta
        
        Delta = params['TE'] - T_90/2 - delta - zeta - params['T_readout']*1e-3

                             # Delta and delta are greater than or equal to zero, compute the b-value
        #b = GAM ** 2 * Gmax ** 2 * (delta ** 2 * (Delta - delta / 3) + zeta ** 3 / 30 - delta * zeta ** 2 / 6)
        b = get_bval(g,params)
        if (np.abs(b - target_bval) <= target_bval*tol): # Stop if the calculated b-value is within 1% of the target
            break
        elif b > (target_bval + target_bval*tol) :
            print('\tNot within tolerance but bval exceeded: bval = {:.2f},flat = {:.2f}'.format(b,flat))
            prev_bval = get_bval(g[:-1],params)
            if prev_bval - target_bval < b- target_bval:
                g=prev_g
                t=prev_t
            break
        flat += dt
        prev_t = t
        prev_g = g
    return g,t,TE,b,


def asymm(params, tol):
    # Define waveform parameters
    T_90 = params['T_90'] * 1e-3
    T_180 = params['T_180'] * 1e-3
    dt = params['dt']
    Gmax = params['gmax'] / 1000
    Smax = params['smax'] / 1000
    GAM = 2 * np.pi * 42.58e3
    zeta = (Gmax / Smax) * 1e-3
    target_bval = params['b']
    b = 0
    flat = 0e-3
    motion_comp = params['MMT']
    prev_t = np.array([0])
    prev_g = np.array([0])

    while flat <= 100e-3 :
        if 2*zeta+ flat < T_180:
            diff = T_180 - 2*zeta
            flat = flat+diff
        
        if motion_comp == 2: 
            
            t = []
            g = []
            gap = params['T_readout'] * 1e-3 - T_90/2 
            # Add in the time spent to slew up 
            gap_p_180=  2*zeta +flat 
            
            flat_time = flat*2 + zeta 
            intervals = [T_90, gap, zeta, flat, zeta, zeta,flat,flat, zeta, T_180, zeta, flat,flat, zeta,zeta,flat,zeta]

            time = [
                            np.arange(0,T_90+gap,dt),
                            np.arange(T_90+gap,T_90+gap+zeta,dt),
                            np.arange(T_90+gap+zeta,T_90+gap+zeta+flat,dt),
                            np.arange(T_90+gap+zeta+flat,T_90+gap+zeta+flat+zeta,dt),
                            np.arange(T_90+gap+zeta+flat+zeta,T_90+gap+zeta+flat+zeta+zeta,dt),
                            np.arange(T_90+gap+zeta+flat+zeta+zeta,T_90+gap+zeta+flat+zeta+zeta+flat_time,dt),
                            np.arange(T_90+gap+zeta+flat+zeta+zeta+flat_time,T_90+gap+zeta+flat+zeta+zeta+flat_time+zeta,dt),
                            np.arange(T_90+gap+zeta+flat+zeta+zeta+flat_time+zeta,T_90+gap+zeta+flat+zeta+zeta+flat_time+zeta+gap_p_180,dt),
                            np.arange(T_90+gap+zeta+flat+zeta+zeta+flat_time+zeta+gap_p_180,T_90+gap+zeta+flat+zeta+zeta+flat_time+zeta+gap_p_180+zeta,dt),
                            np.arange(T_90+gap+zeta+flat+zeta+zeta+flat_time+zeta+gap_p_180+zeta,T_90+gap+zeta+flat+zeta+zeta+flat_time+zeta+gap_p_180+zeta+flat_time,dt),
                            np.arange(T_90+gap+zeta+flat+zeta+zeta+flat_time+zeta+gap_p_180+zeta+flat_time,T_90+gap+zeta+flat+zeta+zeta+flat_time+zeta+gap_p_180+zeta+flat_time+zeta,dt),
                            np.arange(T_90+gap+zeta+flat+zeta+zeta+flat_time+zeta+gap_p_180+zeta+flat_time+zeta,T_90+gap+zeta+flat+zeta+zeta+flat_time+zeta+gap_p_180+zeta+flat_time+zeta+zeta,dt),
                            np.arange(T_90+gap+zeta+flat+zeta+zeta+flat_time+zeta+gap_p_180+zeta+flat_time+zeta+zeta,T_90+gap+zeta+flat+zeta+zeta+flat_time+zeta+gap_p_180+zeta+flat_time+zeta+zeta+flat,dt),
                            np.arange(T_90+gap+zeta+flat+zeta+zeta+flat_time+zeta+gap_p_180+zeta+flat_time+zeta+zeta+flat,T_90+gap+zeta+flat+zeta+zeta+flat_time+zeta+gap_p_180+zeta+flat_time+zeta+zeta+flat+zeta,dt),

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

        TE = t[-1] * 1e3 + params['T_readout']
        params['TE'] = TE*1e-3

        
        b = get_bval(g,params)
        if (np.abs(b - target_bval) <= target_bval*tol): # Stop if the calculated b-value is within 1% of the target
            break
        elif b > (target_bval + target_bval*tol) :
            print('\tNot within tolerance but bval exceeded: bval = {:.2f},flat = {:.2f},te={:.2f}'.format(b,flat,TE))
            prev_bval = get_bval(g[:-1],params)
            if prev_bval - target_bval < b- target_bval:
                g=prev_g
                t=prev_t
            break
        else:
            flat += dt
        prev_t = t
        prev_g = g
        flat += dt
    return g,t,TE,b,

def calc_trap(input_params,lim,pns_thresh):
    pns = 0
    initial_gmax = input_params['gmax']
    target_b = input_params['b']
    # Initial Iteration
    if input_params['MMT'] == 0:
        # Traditional Waveforms 
        G,Time,echoT,b = monopolar(input_params.copy(),lim)
        pns= np.abs(get_stim(G[np.newaxis,:], input_params['dt']))

    elif input_params['MMT'] == 1:
        # Traditional Waveforms 
        G,Time,echoT,b = bipolar(input_params.copy(),lim)
        pns= np.abs(get_stim(G[np.newaxis,:], input_params['dt']))

    elif input_params['MMT'] == 2:
        # Traditional Waveforms 
        G,Time,echoT,b = asymm(input_params.copy(),lim)
        pns= np.abs(get_stim(G[np.newaxis,:], input_params['dt']))

    # Iterate until fall below PNS Threshold 
    while np.nanmax(pns)>pns_thresh:
        print('\tPNS above thresh,pns at',np.nanmax(pns) )
        input_params['smax']-=5
        if input_params['MMT'] == 0:
            # Traditional Waveforms 
            G,Time,echoT,b = monopolar(input_params.copy(),lim)
            pns= np.abs(get_stim(G[np.newaxis,:], input_params['dt']))

        elif input_params['MMT'] == 1:
            # Traditional Waveforms 
            G,Time,echoT,b = bipolar(input_params.copy(),lim)
            pns= np.abs(get_stim(G[np.newaxis,:], input_params['dt']))

        elif input_params['MMT'] == 2:
            # Traditional Waveforms 
            G,Time,echoT,b = asymm(input_params.copy(),lim)
            pns= np.abs(get_stim(G[np.newaxis,:], input_params['dt']))
    
    print('\tPNS is',np.round(np.max(pns),2))
    # check that b-value is correct - this should rarely be used  
    TE_prev = 100000000
    prev_gmax = 10000000

    while abs(b - target_b) > target_b*lim and abs(b-target_b)>5:
        if abs(b-target_b)<5:
            break
        print('\tBvalue not match, reducing gmax, difference is:',np.round(b - input_params['b'],2), 'Bval',np.round(b,2),'TE',np.round(echoT,2))
        input_params['TE'] = echoT
        b = get_bval(G,input_params)
        
        TE_prev = echoT
        prev_gmax = input_params['gmax']
        if b-input_params['b'] > 5:
            input_params['gmax']-=5
        else:
            input_params['gmax']-=1
        

        if input_params['MMT'] == 0:
            # Traditional Waveforms 
            G,Time,echoT,b = monopolar(input_params.copy(),lim)
            pns= np.abs(get_stim(G[np.newaxis,:], input_params['dt']))    

        elif input_params['MMT'] == 1:
            # Traditional Waveforms 
            G,Time,echoT,b = bipolar(input_params.copy(),lim)
            pns= np.abs(get_stim(G[np.newaxis,:], input_params['dt']))

        elif input_params['MMT'] == 2:
            # Traditional Waveforms 
            G,Time,echoT,b = asymm(input_params.copy(),lim)
            pns= np.abs(get_stim(G[np.newaxis,:], input_params['dt']))

        if abs(b-input_params['b']) < target_b*lim:
            break

        if TE_prev - echoT < 0:
            echoT = TE_prev
            input_params['gmax'] = prev_gmax
            if input_params['MMT'] == 0:
                # Traditional Waveforms 
                G,Time,echoT,b = monopolar(input_params.copy(),lim)
                pns= np.abs(get_stim(G[np.newaxis,:], input_params['dt']))    

            elif input_params['MMT'] == 1:
                # Traditional Waveforms 
                G,Time,echoT,b = bipolar(input_params.copy(),lim)
                pns= np.abs(get_stim(G[np.newaxis,:], input_params['dt']))

            elif input_params['MMT'] == 2:
                # Traditional Waveforms 
                G,Time,echoT,b = asymm(input_params.copy(),lim)
                pns= np.abs(get_stim(G[np.newaxis,:], input_params['dt']))
            break

        
    return G,Time,echoT,b,pns




