# Neuromorphic Body Schema

Repository to collect scripts for simulation and evaluation of body schema for a humanoid robotic platform.

### Cloning the Project Repository and Pulling LFS Files

To clone the `neuromorphic_body_schema` project repository and download any Git LFS (Large File Storage) files that are required for the project, use the following commands:

```
sudo apt update
sudo apt install git-lfs
git clone https://github.com/event-driven-robotics/neuromorphic_body_schema
cd neuromorphic_body_schema
git lfs pull
```

These commands will update your package list, install Git LFS, clone the project repository, and download any Git LFS files required for the project.

### Installing Required Packages

Required Python packages are listed in the `requirements.txt` file. To install them run the following command:
```
pip install -r requirements.txt
```

### Event Camera Simulation
The script `rgb2e_mujoco.py` starts a simulated environment with the iCub humanoid robot, converting the simulated camera feed into an event stream. To run it go in the `neuromorphic_body_schema` and run 
```
python rgb2e_mujoco.py
```
