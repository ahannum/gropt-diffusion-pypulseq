import numpy as np 
import sys
sys.path.append("utils/gropt/python")
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
    
    while flat <= 100000e-3 :
        
        if motion_comp == 0: 
            
            t = []
            g = []
            gap_p_180 = params['T_readout'] * 1e-3 - T_90/2 + T_180
            gap = params['T_readout'] * 1e-3 - T_90/2 
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

        TE = t[-1] * 1e3 + params['T_readout']
        params['TE'] = TE
        delta = flat + zeta
        
        Delta = params['TE'] - T_90/2 - delta - zeta - params['T_readout']*1e-3
        b = get_bval(g[np.newaxis,:],params)

        if (np.abs(b - target_bval) <= target_bval*tol): # Stop if the calculated b-value is within 1% of the target
            break
        elif b > (target_bval + target_bval*tol) and flat > 0 :
            print('\tNot within tolerance but bval exceeded: bval = {:.2f},flat = {:.2f}'.format(b,flat))
            prev_bval = get_bval(prev_g[np.newaxis,:-1],params)
            if prev_bval - target_bval < b- target_bval:
                g=prev_g
                t=prev_t
            break
        flat +=dt
        prev_t = t
        prev_g = g
    return g,t,TE,b,zeta,flat, 
    

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

    while flat <= 1000e-3 :
        
        if motion_comp == 1: 
            T_180 = params['T_180'] * 1e-3
            
            t = []
            g = []
            gap = params['T_readout'] * 1e-3 - T_90/2 
            intervals = [T_90, gap, zeta, flat, zeta, zeta,flat,zeta, T_180, zeta, flat, zeta,zeta,flat,zeta]
            T_180 = T_180+ gap

            time = [
                np.linspace(0, T_90, int(T_90 / dt)),  # 90 pulse

                np.linspace(T_90, T_90 + zeta, int(zeta / dt)),  # zeta
                np.linspace(T_90 + zeta, T_90 + zeta + flat, int(flat / dt)),  # flat
                np.linspace(T_90 + zeta + flat, T_90 + zeta + flat + zeta, int(zeta / dt)),  # zeta

                np.linspace(T_90 + zeta + flat + zeta, T_90 + zeta + flat + zeta + zeta, int(zeta / dt)),  # zeta
                np.linspace(T_90 + zeta + flat + zeta + zeta, T_90 + zeta + flat + zeta + zeta + flat, int(flat / dt)),  # flat
                np.linspace(T_90 + zeta + flat + zeta + zeta + flat, T_90 + zeta + flat + zeta + zeta + flat + zeta, int(zeta / dt)),  # zeta

                np.linspace(T_90 + zeta + flat + zeta + zeta + flat + zeta, T_90 + zeta + flat + zeta + zeta + flat + zeta + T_180, int(T_180 / dt)),  # 180 pulse

                np.linspace(T_90 + zeta + flat + zeta + zeta + flat + zeta + T_180, T_90 + zeta + flat + zeta + zeta + flat + zeta + T_180 + zeta, int(zeta / dt)),  # zeta
                np.linspace(T_90 + zeta + flat + zeta + zeta + flat + zeta + T_180 + zeta, T_90 + zeta + flat + zeta + zeta + flat + zeta + T_180 + zeta + flat, int(flat / dt)),  # flat
                np.linspace(T_90 + zeta + flat + zeta + zeta + flat + zeta + T_180 + zeta + flat, T_90 + zeta + flat + zeta + zeta + flat + zeta + T_180 + zeta + flat + zeta, int(zeta / dt)),  # zeta
                np.linspace(T_90 + zeta + flat + zeta + zeta + flat + zeta + T_180 + zeta + flat + zeta, T_90 + zeta + flat + zeta + zeta + flat + zeta + T_180 + zeta + flat + zeta + zeta, int(zeta / dt)),  # zeta
                np.linspace(T_90 + zeta + flat + zeta + zeta + flat + zeta + T_180 + zeta + flat + zeta + zeta, T_90 + zeta + flat + zeta + zeta + flat + zeta + T_180 + zeta + flat + zeta + zeta + flat, int(flat / dt)),  # flat
                np.linspace(T_90 + zeta + flat + zeta + zeta + flat + zeta + T_180 + zeta + flat + zeta + zeta + flat, T_90 + zeta + flat + zeta + zeta + flat + zeta + T_180 + zeta + flat + zeta + zeta + flat + zeta, int(zeta / dt)),  # zeta
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
        params['TE'] = TE
        delta = flat + zeta
        
        Delta = params['TE'] - T_90/2 - delta - zeta - params['T_readout']*1e-3

                             # Delta and delta are greater than or equal to zero, compute the b-value
        #b = GAM ** 2 * Gmax ** 2 * (delta ** 2 * (Delta - delta / 3) + zeta ** 3 / 30 - delta * zeta ** 2 / 6)
        b = get_bval(g[np.newaxis,:],params)
        if (np.abs(b - target_bval) <= target_bval*tol): # Stop if the calculated b-value is within 1% of the target
            break
        elif b > (target_bval + target_bval*tol) :
            print('\tNot within tolerance but bval exceeded: bval = {:.2f},flat = {:.2f}'.format(b,flat))
            prev_bval = get_bval(prev_g[np.newaxis,:-1],params)
            if prev_bval - target_bval < b- target_bval:
                g=prev_g
                t=prev_t
            break
        flat += dt
        prev_t = t
        prev_g = g
    return g,t,TE,b,zeta,flat,


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

    while flat <= 100000e-3 :
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

        TE = t[-1] * 1e3 + params['T_readout']
        params['TE'] = TE

        
        b = get_bval(g[np.newaxis,:],params)
        if (np.abs(b - target_bval) <= target_bval*tol): # Stop if the calculated b-value is within 1% of the target
            break
        elif b > (target_bval + target_bval*tol) :
            print('\tNot within tolerance but bval exceeded: bval = {:.2f},flat = {:.2f},te={:.2f}'.format(b,flat,TE))
            
            prev_bval = get_bval(prev_g[np.newaxis,:-1],params)
            print(prev_bval)
            if prev_bval - target_bval < b- target_bval:
                g=prev_g
                t=prev_t
                break


                

            
            

            
        else:
            flat += dt
        prev_t = t
        prev_g = g
        flat += dt

        
        
    return g,t,TE,b,zeta,flat,



def calc_trap(input_params,lim,pns_thresh):
    pns = 0

    target_b = input_params['b']
    # Initial Iteration
    if input_params['MMT'] == 0:
        # Traditional Waveforms 
        G,Time,echoT,b,zeta,flat = monopolar(input_params.copy(),lim)
        pns= np.abs(get_stim(G[np.newaxis,:], input_params['dt']))

    elif input_params['MMT'] == 1:
        # Traditional Waveforms 
        G,Time,echoT,b,zeta,flat = bipolar(input_params.copy(),lim)
        pns= np.abs(get_stim(G[np.newaxis,:], input_params['dt']))

    elif input_params['MMT'] == 2:
        # Traditional Waveforms 
        G,Time,echoT,b,zeta,flat = asymm(input_params.copy(),lim)
        pns= np.abs(get_stim(G[np.newaxis,:], input_params['dt']))

    # Iterate until fall below PNS Threshold
    print(len(pns))
    print(flat)
    while pns.size > 0 and np.nanmax(pns)>pns_thresh:
        
        print('\tPNS above thresh,pns at',np.nanmax(pns) )
        input_params['smax']-=1
        if input_params['MMT'] == 0:
            # Traditional Waveforms 
            G,Time,echoT,b,zeta,flat = monopolar(input_params.copy(),lim)
            pns= np.abs(get_stim(G[np.newaxis,:], input_params['dt']))

        elif input_params['MMT'] == 1:
            # Traditional Waveforms 
            G,Time,echoT,b,zeta,flat = bipolar(input_params.copy(),lim)
            pns= np.abs(get_stim(G[np.newaxis,:], input_params['dt']))

        elif input_params['MMT'] == 2:
            # Traditional Waveforms 
            G,Time,echoT,b,zeta,flat = asymm(input_params.copy(),lim)
            pns= np.abs(get_stim(G[np.newaxis,:], input_params['dt']))
            pns
    
        print(len(G))
        print('\tPNS is',np.round(np.max(pns),2))
    

    while abs(b - target_b) > target_b*lim and abs(b-target_b)>5:
        input_params['gmax']-=1
        if abs(b-target_b)<5:
            break
        print('\tBvalue not match, reducing gmax, difference is:',np.round(b - input_params['b'],2), 'Bval',np.round(b,2),'TE',np.round(echoT,2))
        input_params['TE'] = echoT
        input_params['lim'] = lim
        b = get_bval(G[np.newaxis,:],input_params)
        
        TE_prev = echoT
        prev_gmax = input_params['gmax']

    
        G,Time,echoT,b,zeta,flat = which_waveform(input_params.copy())
        pns= np.abs(get_stim(G[np.newaxis,:], input_params['dt']))    


        if abs(b-input_params['b']) < target_b*lim:
            break

        if TE_prev - echoT < 0:
            echoT = TE_prev
            input_params['gmax'] = prev_gmax
            G,Time,echoT,b,zeta,flat = which_waveform(input_params.copy())
            pns= np.abs(get_stim(G[np.newaxis,:], input_params['dt']))    
            break

    gap = input_params['T_readout'] * 1e-3 - (input_params['T_90']  * 1e-3)/2 
    
    return G,Time,echoT,b,pns, gap, zeta, flat




def which_waveform(input_params):
    if input_params['MMT'] == 0:
        G,Time,echoT,b,zeta,flat = monopolar(input_params.copy(),input_params['lim'])

    elif input_params['MMT'] == 1:
        G,Time,echoT,b,zeta,flat = bipolar(input_params.copy(),input_params['lim'])

    elif input_params['MMT'] == 2:
        G,Time,echoT,b,zeta,flat = asymm(input_params.copy(),input_params['lim'])
        
    else:
        raise Exception("Invalid Moment Nulling level")
    
    gap = input_params['T_readout'] * 1e-3 - (input_params['T_90']  * 1e-3)/2 
    return G,Time,echoT,b,zeta,flat
