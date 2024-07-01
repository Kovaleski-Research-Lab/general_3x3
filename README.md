# general_3x3

A wrapper for [meep code](https://meep.readthedocs.io/en/latest/) that generates a dataset of electric field data for 3x3 meta-atom pillars. The radii of the pillars are generated randomly from values within the 75 nm to 150 nm range (See [single pillar sim](https://github.com/Kovaleski-Research-Lab/single_pillar_sim) for reference).

The details of the simulation are described in [this publication](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13042/1304206/Time-series-neural-networks-to-predict-electromagnetic-wave-propagation/10.1117/12.3013488.full).

## Folder descriptions

- [kubernetes folder](https://github.com/Kovaleski-Research-Lab/general_3x3/tree/andy_branch/kubernetes) : contains files for generating data and reducing data (reducing raw data to volumes) via kubernetes. Data transfer via rclone is also supported.
- [configs folder](https://github.com/Kovaleski-Research-Lab/meta_atom_rnn/tree/main/configs) : contains the configuration (.yaml) file for the entire pipeline, including a flag `deployment_mode` which gives the user the option to develop locally (via [Docker](https://hub.docker.com/layers/kovaleskilab/meep/v3_lightning/images/sha256-e550d12e2c85e095e8fd734eedba7104e9561e86e73aac545614323fda93efb2?context=repo)) with a limited dataset, or using the Nautilus cluster via Kubernetes. The config file also contains a parameter called `task`, which allows the user to either generate raw data or preprocess (reduce) the raw data. Additionally, the config file contains paths, and meep parameters.
- [meep utils folder](https://github.com/Kovaleski-Research-Lab/general_3x3/tree/andy_branch/meep_utils) : contains files with python wrappers for building meep sources, simulations, geometries, and field monitors.
- [radii folder](https://github.com/Kovaleski-Research-Lab/general_3x3/tree/andy_branch/radii) : contains a script for generating 3x3 configurations of pillar radii between 75 nm and 250 nm. Outputs a .pkl file with a python list containing 1x9 sublists. This pickle file is used by `run_datagen.py`.
- [utils folder](https://github.com/Kovaleski-Research-Lab/general_3x3/tree/andy_branch/utils) : contains helper scripts for general python programming and data modification scripting.
- [supplementary folder](https://github.com/Kovaleski-Research-Lab/general_3x3/tree/andy_branch/supplementary) : Contains background info / supplementary studies to support the simulations.
     - [gaussian_width](https://github.com/Kovaleski-Research-Lab/general_3x3/tree/andy_branch/supplementary/gaussian_width) : Contains a script that helps us viualize the relationship between frequency and wavelength in meep units.
     - [sim_states](https://github.com/Kovaleski-Research-Lab/general_3x3/tree/andy_branch/supplementary/sim_states) : Contains a script that shows the user how to dump out a simulation state after a simulation has run, and how to reload that simulation.
     - [resolution_study](https://github.com/Kovaleski-Research-Lab/general_3x3/tree/andy_branch/supplementary/resolution_study) : Contains the script used as well as the analysis to determine at what resolution the simulation converges.
     - [core_study](https://github.com/Kovaleski-Research-Lab/general_3x3/tree/andy_branch/supplementary/core_study) : Contains the script used as well as the analysis to determine the optimal number of cores to get the fastest simulation time. We used WOPR, which has 112 CPU cores, to run this experiment - Times will vary depending on CPUs.

## How to run this code

### Prerequisites:
1. [Kubernetes](https://github.com/Kovaleski-Research-Lab/Global-Lab-Repo/blob/main/sops/software_development/kubernetes.md) must be installed if using Nautilus resources.
2. Must be running the appropriate docker container for [local deployment](https://hub.docker.com/layers/kovaleskilab/meep/v3_lightning/images/sha256-e550d12e2c85e095e8fd734eedba7104e9561e86e73aac545614323fda93efb2?context=repo) or [kubernetes deployment](https://hub.docker.com/layers/kovaleskilab/meep_ml/launcher/images/sha256-464ec5f4310603229e96b5beae9355055e2fb2de2027539c3d6bef94b7b5a4f1?context=repo)
3. You should be remotely connected to Marge
   ```
   ssh {your_pawprint}@128.206.23.4
   ```
5. Your docker container for local deployment should be mounted as follows (data is mounted to `/home/datasets` - code and results should be unique to you):
   ```
   -v /home/{your_pawprint}/Documents/code:/develop/code \
   -v /home/datasets:/develop/data \
   -v /home/{your_pawprint}/Documents/results:/develop/results \
   ```
6. Your docker container for kubernetes deployment should be mounted as follows:
   ```
   -v /home/{your_pawprint}/Documents/code:/develop/code \
   -v /home/datasets:/develop/data \
   -v /home/{your_pawprint}/Documents/results:/develop/results \
   -v ~/.kube:/root/.kube 
   ```
  - Note: You may prefer a conda environment instead of a docker container for launching Kubernetes jobs. A barebones environment should have jinja2 installed (not necessary to install kubernetes in the conda environment if you used `snap install` per the Kubernetes link in item 1.

### Running the code

The output of `run_datagen.py` is stored in a folder generated by the script with the naming convention `000{idx}`. Regardless of `deployment_mode`, the file structure is as follows:

/000{idx}/  
 ├─ dft_0000{idx}.h5 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 2.4G, output of `sim.output_dft()`. More at: [meep dft fields](https://meep.readthedocs.io/en/latest/Mode_Decomposition/#exporting-frequency-domain-fields)  
 ├─ epsdata_0000{idx}.pkl  &nbsp;&nbsp;&nbsp;&nbsp;# 921M, output of `sim.get_epsilon()`. More at: [epsilon data](https://meep.readthedocs.io/en/latest/Python_User_Interface/#array-slices)  
 ├─ metadata_0000{idx}.pkl &nbsp;&nbsp;# 79M, output of `sim.get_array_metadata`. More at: [meep metadata](https://meep.readthedocs.io/en/latest/Python_User_Interface/#array-metadata)  

The output of `reduce_data.py` is a .pkl file with the naming convention `000{idx}.pkl`. Regardless of `deployment_mode`, the file structure is as follows:

/volumes/  
 ├─ 000{idx}.pkl &nbsp;# 398M

- File sizes are based on a meep resolution of 80 (Can be adjusted in config file).

#### Option 1: Running locally:
- Note: Because the storage requirements for this dataset are large, large-scale data generation using the Nautilus cluster via Kubernetes is recommended.

**Step 1** Generate the data (This generates a single sample. To run multiple samples, a bash script is recommended - However, it is not recommeded to do this locally becuase of storage considerations)
  
  1. Update the [config file](https://github.com/Kovaleski-Research-Lab/general_3x3/blob/andy_branch/configs/config.yaml) in an editor:
     
  - deployment_mode : 0
  - task : 0
  - idx : {any integer value representing the desired index from the .pkl file output of `generate_radii.py`}

  2. Ensure you are looking at `andy_branch` by navigating to `/code/general_3x3` and running
     ```
     git status
     ```
     If the output is **not** on branch `andy_branch`, run the command,
     ```
     git checkout andy_branch
     ```
     
  3. From your Docker container, navigate to `/develop/code/general_3x3` and run
     ```
     mpirun -np 32 python3 main.py -config configs/config.yaml
     ```
     - Note: This command expects 32 CPU cores to be available.
     
**Step 2** Reduce the raw data into volumes.

  1. Update the [config file](https://github.com/Kovaleski-Research-Lab/general_3x3/blob/andy_branch/configs/config.yaml) in an editor:
     
  - deployment_mode : 0
  - task : 1

  2. Ensure you are looking at `andy_branch` (See Step 2 above)

  3. From your Docker container, navigate to `/develop/code/general_3x3` and run
     ```
     python3 main.py -config configs/config.yaml
     ```

#### Option 2: Scale up, launch Kubernetes jobs:

**Step 2** Create storage

Navigate to `general_3x3/kubernetes/`. Update 'metadata.name` to your desired pvc name.
Run
```
kubectl apply -f storage.yaml
```
Verify creation of storage volume
```
kubectl get pvc
```

**Step 1** Generate the data
  
  1. Update the [config file](https://github.com/Kovaleski-Research-Lab/general_3x3/blob/andy_branch/configs/config.yaml) in an editor:
     
  - deployment_mode : 1
  - task : 0
  - kube.datagen_job.start_group_id: 0 (or whatever index you want the first sample to come from the .pkl file output of `generate_radii.py`)
  - kube.datagen_job.num_sims: 1500 (or however many sims you want to run total)
  - kube.datagen_job.num_parallel_ops: 2 (If you want more than 2, you will have to contact a [local administrator](https://github.com/MU-HPDI/nautilus/wiki/Getting-Started) about setting up a [taint](https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration/).
  - kube.pvc_name : should match metadata.name in `/kubernetes/storage.yaml`
 
  2. From your (launch kube) Docker container, navigate to `/develop/code/general_3x3/kubernetes/gen_data` and run
     ```
     python3 launch_jobs.py -config ../configs/config.yaml
     ```
  3. Monitor data generation pods
     ```
     kubectl get pods
     ```
     ```
     kubectl describe pod {pod_name}
     ```
     ```
     kubectl logs {pod_name}
     ```
 4. Use an interactive pod to check what simulation files are being generated
    ```
    kubectl apply -f monitor.yaml
    ```
    Once the pod status for `monitor-dataset` is `Running` enter the pod as interactive root user,
    ```
    kubectl exec -it monitor-dataset -- /bin/bash
    ```
    ```
    ls /develop/results
    ```
    To see meep outputs:
    ```
    cd /develop/results/current_logs
    ```
**Step 2** Reduce the raw data into volumes.

  1. Update the [config file](https://github.com/Kovaleski-Research-Lab/general_3x3/blob/andy_branch/configs/config.yaml) in an editor:
     
  - deployment_mode : 1
  - task : 1

  2. From your (launch kube) Docker container, navigate to `/develop/code/general_3x3/kubernetes/reduce_data` and run
     ```
     python3 launch_jobs.py -config ../configs/config.yaml
     ```
