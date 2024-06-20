The files in this repo can be used to run simulations using the python implementation of  Meep (https://meep.readthedocs.io/en/latest/) for a 3x3 meta-atom neighborhood.

For background, see the following: https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.15.054039. Under local phase approximation (LPA), a single dielectric pillar exhibits 0-2$\pi$ 
phase delay for pillars with radii between 75 and 250 nm when excited by a 1550 nm source.

This work drops the LPA but retains the 75 - 250 micron radii range shown in the paper. Using pillars in this range, we extend the simulation space to include 9 randomly chosen pillars on a 3x3 grid. 
The simulations are run at a resolution of 80. The simulation environment has pml in the +/- z direction, and periodic boundary conditions in x and y. The y-polarized Gaussian source is incident on 
the pillars, and dft fields are collected for the entire non-pml volume.

The folder `3x3_pillar_sims` contains the `_3x3PillarSim()` class, which serves as a wrapper of the Meep code for the specific task of simulating 3x3 neighborhoods of pillars.

## Instructions for executing the code

This code is cpu-core intensive, and is ideally run on a machine with 32 to 48 cores using mpi. It should be run on the docker image, `kovaleskilab/meep:v2_with_torch` 
which can be found at https://hub.docker.com/repository/docker/kovaleskilab/meep/general  [login: kovaleskilab, password: mindful1024] 

This image and the code are set up for the following container mounts: "/develop/data" "/develop/results" "/develop/code"

The command `mpirun -np 48 --allow-run-as-root python3 run_sim.py -index 0` will execute the simulation code.
    - mpirun allows us to run in parallel. the -np flag indicates number of processes - At a resolution of 80, 32 to 48 cores is ideal. if you lower the resolution (in config.yaml) to below 30, you can
        reasonably run the code on a single core.
    - --allow-run-as-root is required to run the code in a docker container.
    - the -index flag indicates which group of randomly generated pillars will be populated for the simulation. This can be any integer value between 0 and 5000.
        - run_sim.py gets these groups of pillars from the file `neighbors_library_allrandom.pkl`, which is just a list with len 5000 of randomly generated lists of len 9 of values between 0.075 and 0.250

The output of `run_sim.py` is a .pkl file containing the dft fields for each wavelength of interest defined in the gaussian source (1060 nm, 1300 nm, 1550 nm, 1650 nm, 2881 nm). The size of this file is
just under 2 GB

## Some other notes

`mapping.py` allows the user to map a single pillar radius to its corresponding LPA phase delay, as shown in the Raeker paper referenced above (we also validated this independently using meep). This is not
used to simulate data but included in case the user needs it for their application.

`config.yaml` contains all simulation variables and paths.

The `utils` folder contains a file called `parameter_manager.py` which reads in the config file and allows us to set variables which require us to do math. It also organizes the params into groups.

Important: The output file size, 2GB, is prohibitively large for any reasonably sized dataset. The file `preprocess_data.py` organizes the contents of the .pkl file and grabs a slice of the dft fields 
at a distance about 775 nm downstream of the pillars.

The dataset contained here is the output of `preprocess_data.py` NOT `run_sim.py`! The raw outputs of `run_sim.py` are stored in a persistent volume claim (PVC) on the Nautilus cluster. 
The PVC name is meep-results.
