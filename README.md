# gropt-diffusion-pypulseq
Pypulseq Implementation of PNS-constrianed Diffusion Waveforms created with the MRI Gradient Optimization Toolbox (GrOpt)

# Overview
This repository is a python implementation of integrating optimized gradient waveform design in Pulseq, for accessible pulse sequence programming and implementation. In this Repository there are the following main folders: 

- **Simulation**: Code to generate optimized (GrOpt) and Pulsed-gradient spin-echo (PGSE) waveforms, given different system parameters. The script to run a demonstration is Demo_ISMRM2025.ipynb, allowing the ability to create waveforms using optimized and conventional approaches. 


This repository will continue to be updated with code to generate and assess waveforms using the MRI Gradient Optimization Toolbox and convert to Pulseq files for capabilities of generating waveforms on a scanner. 


# Demo
Once the environment is installed (see installation section) you can follow these demos: 
**Simulation**: Refer to [demo_ISMRM.ipynb] notebook  to generate PGSE and GrOpt waveforms based on system inputs. 


# Installation
**Prerequisite:** Make sure you have [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.  
If you're not using Conda, ensure Python and `pip` are installed, and consider converting the `.yaml` file to `requirements.txt`.
This environment requires **Python 3.11.8**. You can check your Python version with:
```bash
python --version
```
If you're not using Conda, ensure Python 3.11.8 and `pip` are installed, and consider converting the `.yaml` file to a `requirements.txt`.

To create the required Python environment from the provided `.yaml` file:

### 1. Open Terminal or Anaconda Prompt

### 2. Navigate to the folder containing `pulseq-gropt.yaml`
```bash
cd /path/to/your/yaml/file
conda env create -f gropt-pulseq.yaml
conda activate gropt-pulseq
conda list
```


#  References
This repository has been associated with the following abstracts:


Hannum AJ, Loecher M, Setsompop K, Ennis DB. Mitigation of Peripheral-Nerve Stimulation with Arbitrary Gradient Waveform Design for Diffusion-Weighted MRI.


# Resources
This toolbox builds upon and utilizes tools from the following open-source projects:

**GrOpt**: Loecher M, Middione MJ, Ennis DB. A gradient optimization toolbox for general purpose time-optimal MRI gradient waveform design. Magn Reson Med. 2020 Dec;84(6):3234-3245. [https://github.com/mloecher/gropt]

**Pulseq**: Layton KJ, Kroboth S, Jia F, et al. Pulseq: A rapid and hardware-independent pulse sequence prototyping framework. Magn Reson Med. 2017;77(4):1544-1552. [https://pulseq.github.io/]

**Pypulseq**: [https://github.com/imr-framework/pypulseq] 




