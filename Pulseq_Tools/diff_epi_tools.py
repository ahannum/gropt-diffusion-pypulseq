import numpy as np
import pypulseq as pp
import matplotlib.pyplot as plt


def make_diff_epirs(plot: bool = False, timings_only: bool = False,
                    do_fatsat: bool = False,
                    fovx: float = 288e-3, Nx: int = 96, 
                    fovy: float = 288e-3, Ny: int = 96, partial_fourier: float = 0.8, R: float = 1.0,
                    slice_thickness: float = 6e-3, refocus_in_spoiler: bool = False,
                    TE: float = 200e-3,
                    pe_enable: bool = True,
                    extra_readout_time: float = 2e-5, ro_os: float = 1.0,
                    t_RF_ex: float = 2e-3, t_RF_ref: float = 2e-3, spoil_factor: float = 1.5,
                    gmax: float = 36, smax: float = 120,
                    gmax_full: float = 42, smax_full: float = 185,
                    B0: float = 2.89, sat_ppm: float = -3.45,
                    Navg: int = 10, delay_TR: float = 3,
                    diff_dirs: list = [[0,0,0], [1,0,0], [0,1,0], [0,0,1]],
                    diff_grad1 = None, diff_grad2 = None,
                    seq_filename: str = "diff_epirs_temp.seq", write_seq: bool = False,):
    
    

    # Set system limits
    system = pp.Opts(
        max_grad=gmax,
        grad_unit="mT/m",
        max_slew=smax,
        slew_unit="T/m/s",
        rf_ringdown_time=30e-6,
        rf_dead_time=100e-6,
    )

    # Set system limits to use with custom waveforms
    system_full = pp.Opts(
        max_grad=gmax_full,
        grad_unit="mT/m",
        max_slew=smax_full,
        slew_unit="T/m/s",
        rf_ringdown_time=30e-6,
        rf_dead_time=100e-6,
    )

    seq = pp.Sequence()

    # -------------------------
    # ----- Fat Sat Pulse -----

    sat_freq = sat_ppm * 1e-6 * B0 * system.gamma
    rf_fs = pp.make_gauss_pulse(
        flip_angle=110 * np.pi / 180,
        system=system,
        duration=8e-3,
        bandwidth=np.abs(sat_freq),
        freq_offset=sat_freq,
    )
    gz_fs = pp.make_trapezoid(
        channel="z", system=system, delay=pp.calc_duration(rf_fs), area=1 / 1e-4
    )

    # -------------------------
    # ----- Excitation -----

    rf, gz, gz_reph = pp.make_sinc_pulse(
        flip_angle=np.pi / 2,
        system=system,
        duration=t_RF_ex,
        slice_thickness=slice_thickness,
        apodization=0.5,
        time_bw_product=4,
        use="excitation",
        return_gz=True,
    )

    # -------------------------
    # ----- 180 -----

    rf180, gz180, _ = pp.make_sinc_pulse(
        flip_angle=np.pi,
        system=system,
        duration=t_RF_ref,
        slice_thickness=slice_thickness,
        apodization=0.5,
        time_bw_product=4,
        phase_offset=np.pi / 2,
        use="refocusing",
        return_gz=True,
    )
    _, gzr1_t, gzr1_a = pp.make_extended_trapezoid_area(
        channel="z",
        grad_start=0,
        grad_end=gz180.amplitude,
        area=spoil_factor * gz.area,
        system=system,
    )

    if refocus_in_spoiler:
        second_spoiler_area = -gz_reph.area + spoil_factor * gz.area
    else:
        second_spoiler_area = spoil_factor * gz.area
    _, gzr2_t, gzr2_a = pp.make_extended_trapezoid_area(
        channel="z",
        grad_start=gz180.amplitude,
        grad_end=0,
        area=second_spoiler_area,
        system=system,
    )
    if gz180.delay > (gzr1_t[3] - gz180.rise_time):
        gz180.delay -= gzr1_t[3] - gz180.rise_time
    else:
        rf180.delay += (gzr1_t[3] - gz180.rise_time) - gz180.delay
    gz180n = pp.make_extended_trapezoid(
        channel="z",
        system=system,
        times=np.array([*gzr1_t, *gzr1_t[3] + gz180.flat_time + gzr2_t]) + gz180.delay,
        amplitudes=np.array([*gzr1_a, *gzr2_a]),
    )


    # -------------------------
    # ----- PE blips -----

    # Phase blip in shortest possible time.
    # Round up the duration to 2x gradient raster time
    delta_ky_R = R / fovy  # Size for the biggest blip if we are undersampling
    delta_ky = 1.0 / fovy
    blip_duration = (
        np.ceil(2 * np.sqrt(delta_ky_R / system.max_slew) / 10e-6 / 2) * 10e-6 * 2
    )
    # Use negative blips to save one k-space line on our way to center of k-space
    # TODO: We need an undersampled and a fully sampled blip here
    gy = pp.make_trapezoid(
        channel="y", system=system, area=-delta_ky, duration=blip_duration
    )

    # print(f'{1e3*blip_duration = :.02f}')
    print('blip_duration',blip_duration)

    # -------------------------
    # ----- Readout -----

    delta_kx = 1 / fovx
    kx_width = Nx * delta_kx

    # Readout gradient is a truncated trapezoid with dead times at the beginning and at the end each equal to a half of
    # blip duration. The area between the blips should be defined by k_width. We do a two-step calculation: we first
    # increase the area assuming maximum slew rate and then scale down the amplitude to fix the area
    extra_area = blip_duration / 2 * blip_duration / 2 * system.max_slew
    
    gx_fastest = pp.make_trapezoid(
        channel="x",
        system=system,
        area=kx_width + extra_area,
    )
    print('kx_width',kx_width,)
    readout_time = pp.calc_duration(gx_fastest) + extra_readout_time

    print(f'{kx_width = :.02f}')
    print(f'{extra_area = :.02f}')
    print(f'{1000*pp.calc_duration(gx_fastest) = :.02f}')
    print(f'{1000*readout_time = :.02f}')

    
    gx = pp.make_trapezoid(
        channel="x",
        system=system,
        area=kx_width + extra_area,
        duration=readout_time + blip_duration,
    )

    print(f'{1000*(readout_time + blip_duration) = :.02f}')
    print(f'{1000*pp.calc_duration(gx) = :.02f}')

    actual_area = (
        gx.area
        - gx.amplitude / gx.rise_time * blip_duration / 2 * blip_duration / 2 / 2
    )
    actual_area -= (
        gx.amplitude / gx.fall_time * blip_duration / 2 * blip_duration / 2 / 2
    )
    gx.amplitude = gx.amplitude / actual_area * kx_width
    gx.area = gx.amplitude * (gx.flat_time + gx.rise_time / 2 + gx.fall_time / 2)
    gx.flat_area = gx.amplitude * gx.flat_time

    # Calculate ADC
    # We use ramp sampling, so we have to calculate the dwell time and the number of samples, which will be quite
    # different from Nx and readout_time/Nx, respectively.
    adc_dwell_nyquist = delta_kx / gx.amplitude / ro_os
    # Round-down dwell time to 100 ns
    adc_dwell = np.floor(adc_dwell_nyquist * 1e7) * 1e-7
    # Number of samples on Siemens needs to be divisible by 4
    adc_samples = np.floor(readout_time / adc_dwell / 4) * 4
    adc = pp.make_adc(num_samples=adc_samples, dwell=adc_dwell, delay=blip_duration / 2)
    # Realign the ADC with respect to the gradient
    # Supposedly Siemens samples at center of dwell period
    time_to_center = adc_dwell * ((adc_samples - 1) / 2 + 0.5)
    # Adjust delay to align the trajectory with the gradient. We have to align the delay to 1us
    adc.delay = round((gx.rise_time + gx.flat_time / 2 - time_to_center) * 1e6) * 1e-6
    # This rounding actually makes the sampling points on odd and even readouts to appear misaligned. However, on the
    # real hardware this misalignment is much stronger anyways due to the gradient delays


    # -------------------------
    # ----- Blip Halfs -----

    # Split the blip into two halves and produce a combined synthetic gradient
    gy_parts = pp.split_gradient_at(
        grad=gy, time_point=blip_duration / 2, system=system
    )
    gy_blipup, gy_blipdown, _ = pp.align(right=gy_parts[0], left=[gy_parts[1], gx])
    gy_blipdownup = pp.add_gradients((gy_blipdown, gy_blipup), system=system)

    # pe_enable support
    gy_blipup.waveform = gy_blipup.waveform * pe_enable
    gy_blipdown.waveform = gy_blipdown.waveform * pe_enable
    gy_blipdownup.waveform = gy_blipdownup.waveform * pe_enable


    # -------------------------
    # ----- Number of PE -----

    # Phase encoding and partial Fourier
    # PE steps prior to ky=0, excluding the central line
    Ny_pre = round(partial_fourier * Ny / 2 - 1)
    # PE lines after the k-space center including the central line
    Ny_post = round(Ny / 2 + 1)
    Ny_meas = Ny_pre + Ny_post
    
    # print(f'{Ny_pre = }  {Ny_post = }  {Ny_meas = }')


    # -------------------------
    # ----- Prephasers -----

    # Pre-phasing gradients
    gx_pre = pp.make_trapezoid(channel="x", system=system, area=-gx.area / 2)
    gy_pre = pp.make_trapezoid(channel="y", system=system, area=Ny_pre * delta_ky)

    pre_duration = pp.calc_duration(gx_pre, gy_pre)

    # recalculate with matched durations
    gx_pre = pp.make_trapezoid(channel="x", system=system, area=-gx.area / 2, duration=pre_duration)
    gy_pre = pp.make_trapezoid("y", system=system, area=gy_pre.area, duration=pre_duration)

    gy_pre.amplitude = gy_pre.amplitude * pe_enable


    # -------------------------
    # ----- Timings -----
    print(pp.calc_duration(gx))
    # Calculate delay times
    duration_to_center = (Ny_pre + 0.5) * pp.calc_duration(gx)
    rf_center_incl_delay = rf.delay + pp.calc_rf_center(rf)[0]
    rf180_center_incl_delay = rf180.delay + pp.calc_rf_center(rf180)[0]

    # --------------------
    # These are the GrOpt timings
    t_90 = pp.calc_duration(rf, gz) - rf_center_incl_delay
    if not refocus_in_spoiler:
        t_90 += pp.calc_duration(gz_reph)
    t_180 = pp.calc_duration(rf180, gz180n)
    t_readout = duration_to_center + pre_duration

    delay_TE1 = (
        np.ceil(
            (
                TE / 2
                - t_90
                - rf180_center_incl_delay
            )
            / system.grad_raster_time
        )
        * system.grad_raster_time
    )
    delay_TE2 = (
        np.ceil(
            (
                TE / 2
                - pp.calc_duration(rf180, gz180n)
                + rf180_center_incl_delay
                - duration_to_center
                - pre_duration
            )
            / system.grad_raster_time
        )
        * system.grad_raster_time
    )
    assert delay_TE1 >= 0
    assert delay_TE2 >= 0

    # print(f'{1000*duration_to_center = :.02f}')

    timings = {'t_90':t_90,
               't_180':t_180,
               't_readout':t_readout,
               'delay_TE1':delay_TE1,
               'delay_TE2':delay_TE2}
    
    if timings_only:
        return seq, timings

    # ======================
    # CONSTRUCT SEQUENCE
    # ======================

    print('BANDWIDTH',1/adc.dwell/Nx)

    # Define sequence blocks
    for ddir in diff_dirs:
        if diff_grad1 is not None:
            g1_convert = pp.convert.convert(from_value = 1e3*diff_grad1, from_unit = 'mT/m', to_unit = 'Hz/m')
            diff_gx1 = pp.make_arbitrary_grad.make_arbitrary_grad(channel='x', waveform = ddir[0]*g1_convert, system=system_full)
            diff_gy1 = pp.make_arbitrary_grad.make_arbitrary_grad(channel='y', waveform = ddir[1]*g1_convert, system=system_full)
            diff_gz1 = pp.make_arbitrary_grad.make_arbitrary_grad(channel='z', waveform = ddir[2]*g1_convert, system=system_full)

            if pp.calc_duration(diff_gx1) != delay_TE1:
                print('WARNING: diff_grad1 duration = {:.2f} ms  !=   delay_TE1 = {:.2f}'.format(1000*pp.calc_duration(diff_gx1), 1000*delay_TE1))

        if diff_grad2 is not None:
            g2_convert = pp.convert.convert(from_value = 1e3*diff_grad2, from_unit = 'mT/m', to_unit = 'Hz/m')
            diff_gx2 = pp.make_arbitrary_grad.make_arbitrary_grad(channel='x', waveform = ddir[0]*g2_convert, system=system_full)
            diff_gy2 = pp.make_arbitrary_grad.make_arbitrary_grad(channel='y', waveform = ddir[1]*g2_convert, system=system_full)
            diff_gz2 = pp.make_arbitrary_grad.make_arbitrary_grad(channel='z', waveform = ddir[2]*g2_convert, system=system_full)

            if pp.calc_duration(diff_gx2) != delay_TE2:
                print('WARNING: diff_grad2 duration = {:.2f} ms  !=   delay_TE2 = {:.2f}'.format(1000*pp.calc_duration(diff_gx2), 1000*delay_TE2))

        for i_avg in range(Navg):
            if do_fatsat:
                seq.add_block(rf_fs, gz_fs)
            seq.add_block(rf, gz)
            if not refocus_in_spoiler:
                seq.add_block(gz_reph)
            if diff_grad1 is not None:
                seq.add_block(diff_gx1, diff_gy1, diff_gz1)
            else:
                seq.add_block(pp.make_delay(delay_TE1))
            seq.add_block(rf180, gz180n)
            if diff_grad2 is not None:
                seq.add_block(diff_gx2, diff_gy2, diff_gz2)
            else:
                seq.add_block(pp.make_delay(delay_TE2))
            seq.add_block(gx_pre, gy_pre)
            for i in range(1, Ny_meas + 1):
                if i == 1:
                    # Read the first line of k-space with a single half-blip at the end
                    seq.add_block(gx, gy_blipup, adc)
                elif i == Ny_meas:
                    # Read the last line of k-space with a single half-blip at the beginning
                    seq.add_block(gx, gy_blipdown, adc)
                else:
                    # Read an intermediate line of k-space with a half-blip at the beginning and a half-blip at the end
                    seq.add_block(gx, gy_blipdownup, adc)
                gx.amplitude = -gx.amplitude  # Reverse polarity of read gradient
            seq.add_block(pp.make_delay(delay_TR))

    # print(f'{1e3*pp.calc_duration(gx, gy_blipdownup, adc) = :.02f}')

    # Check whether the timing of the sequence is correct
    ok, error_report = seq.check_timing()
    if ok:
        print("Timing check passed successfully")
    else:
        print("Timing check failed. Error listing follows:")
        [print(e) for e in error_report]

    # ======
    # VISUALIZATION
    # ======
    if plot:
        seq.plot()

    # Very optional slow step, but useful for testing during development e.g. for the real TE, TR or for staying within
    # slew-rate limits
    # rep = seq.test_report()
    # print(rep)

    # =========
    # WRITE .SEQ
    # =========
    if write_seq:
        # Prepare the sequence output for the scanner
        seq.set_definition(key="FOV", value=[fovx, fovy, slice_thickness])
        seq.set_definition(key="Name", value="epi")

        seq.write(seq_filename)

    return seq, timings


def plot_trajectory(seq):
    k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspacePP()

    traj_lim = 1.1*np.abs(k_traj_adc[:2]).max()
    print(traj_lim)
    plt.figure(10,figsize=(12,12))
    plt.plot(k_traj[0,], k_traj[1,])
    plt.plot(k_traj_adc[0,], k_traj_adc[1,], marker = '.', linestyle='None', markeredgecolor='r')
    plt.ylim(-traj_lim, traj_lim)
    plt.xlim(-traj_lim, traj_lim)
