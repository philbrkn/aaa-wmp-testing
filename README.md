# AAA - wmp environment

OpenMC is included with addition: openmc/openmc/data/aaa.py.
This replaces vectfit. I pickled njoy at 5e-4 since its expensive to run NJOY each time for u238. so we just import the pickle. 

the pickle is 128 mb, so it cant be added to github. you will have to run njoy the first time.


### install python environment with: 
```bash
conda create -n aaa-wmp-env -c conda-forge openmc scipy h5py numpy matplotlib njoy2016
conda activate aaa-wmp-env
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_OPENMC=0.0+local python -m pip install -e ./openmc
# no need to make c++
```

### run with file:
run_aaa.py


different files:
run_aaa.py: runs the aaa code. saves plots named by piece and channel.

plot_aaa.py: assembles pieces of aaa to plot (work in progress)

run_miaaa.py: what i used to run miaaa code adapted from Monzon et al, miaaa not included at the moment.

plot_vf_code.py: plots the VF output that ran on HPC for U238 on ENDFviii and was pickled. useful for comparing errors to AAA.

current_wmp.py: tests WMP library's h5 file for number of windows and poles.

test_window.py: runs current OpenMC windowing 