## Reduced-order adaptive synchronization in a chaotic neural network with parameter mismatch: A dynamical system vs. machine learning approach


This repo contains the source code for the paper:

Reduced-order adaptive synchronization in a chaotic neural network with parameter mismatch: A dynamical system vs. machine learning approach

Authors: Jan Kobiolka, Jens Habermann, Marius E. Yamakou

Abstract: In this paper, we address the reduced-order synchronization problem between two chaotic memristive Hindmarsh-Rose (HR) neurons of different orders using two distinct methods. The first method employs the Lyapunov active control technique. Through this technique, we develop appropriate control functions to synchronize a 4D chaotic HR neuron (response system) with the canonical projection of a 5D chaotic HR neuron (drive system). Numerical simulations are provided to demonstrate the effectiveness of this approach. The second method is data-driven and leverages a machine learning-based control technique. Our technique utilizes a \textit{ad hoc} combination of reservoir computing (RC) algorithms, incorporating reservoir observer (RO), online control (OC), and online predictive control (OPC) algorithms. We anticipate our effective heuristic RC adaptive control algorithm to guide the development of more formally structured and systematic, data-driven RC control approaches to chaotic synchronization problems, and to inspire more data-driven neuromorphic methods for controlling and achieving synchronization in chaotic neural networks *in vivo*.

## Setting Up


## Prerequisites
- Python version 3.8 to 3.11
- Fortran compiler (gfortran recommended)
  - Install from: [Fortran Installation Guide](https://fortran-lang.org/learn/os_setup/install_gfortran/)


## Installation
1. Set up a virtual environment (recommended).
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run preprocessing:

```bash
python preprocessing.py
```
This creates two required folders: <span style="background-color: #fff3; padding: 2px 4px; border-radius: 3px;">images</span> and
<span style="background-color: #fff3; padding: 2px 4px; border-radius: 3px;">data</span>. (You can also create these manually if preferred.)

4. For some functions, namely (Fig. 2, Fig. 3a, and Fig. 6), require data from Fortran-generated .txt files.
 Please run the following Fortran scripts:
   1. (drive_lya_exp_s.f, drive_lya_exp_sigma.f, drive_lya_exp_k_1.f, drive_lya_exp_k_2.f) -> Fig_2 
   2.  (drive_lya_exp_k_1_k_2.f) -> Fig_3_a (takes around 18 hours)
   3. (synchronization_k_1_k_2.f95) -> Fig_6 (takes around 5-6 days)

- Compile Fortran scripts:
```bash
gfortran file_name -o name_of_executable
```
- Run executables:

```bash
./name_of_executable
```

- Move the generated .txt files into the data folder.

5. Finally, all functions within main.py can be executed.
6. Should you have difficulties or should you require the .txt files feel free to contact me. 




