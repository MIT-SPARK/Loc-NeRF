# Loc-NeRF

Monte Carlo Localization using Neural Radiance Fields. 

## Coordinate Frames
To be consistent throughout the code and in the yaml files, we define coordinates using the camera frame commonly used for NeRF (x right, y up, z inward from the perspective of the camera) unless stated otherwise. Coordinates are FROM Camera TO World unless otherwise stated. Note this is not the same as the more common camera frame used in robotics (x right, y down, z outward).

## Publications

If you find this code relevant for your work, please consider citing our paper:

[Loc-NeRF: Monte Carlo Localization using Neural Radiance Fields](https://arxiv.org/abs/2209.09050)

This work was done in collaboration with MIT and Draper Labs and was partially funded 
by the NASA Flight Opportunities under grant Nos 80NSSC21K0348, ARL DCIST CRA W911NF-17-2-0181, and an Amazon Research Award.

# 1. Installation

## A. Prerequisities

- Install ROS by following [ROS website](http://wiki.ros.org/ROS/Installation).

- If you want to use VIO for the predict step for a real robot demo, install a VIO such as 
[VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion) or [Kimera](https://github.com/MIT-SPARK/Kimera-VIO-ROS).

# 2. Loc-NeRF installation
```bash
# Setup catkin workspace
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/
catkin init

# Clone repo
cd ~/catkin_ws/src
git clone https://github.com/MIT-SPARK/Loc-NeRF

# Compile code
catkin build

# source workspace
source ~/catkin_ws/devel/setup.bash

#install dependencies:
cd ~/catkin_ws/src/Loc-NeRF
pip install -r requirements.txt
```

## Starting Loc-NeRF
We will use ROS and rviz as a structure for running Loc-NeRF and for visualizing performance. 
As a general good practice, remember to source your workspace for each terminal you use.

  1. Open a new terminal and run: `roscore`

  2. In another terminal, launch Loc-NeRF:
  ```bash
  roslaunch locnerf navigate.launch parameter_file:=<param_file.yaml>
  ```

  3. In another terminal, launch rviz for visualization:
  ```bash
  rviz -d $(rospack find locnerf)/rviz/rviz.rviz
  ```

  4. If you are not running with a rosbag, i.e. you are using LLFF data, then Loc-NeRF should start and you should be set. If you are using a rosbag, continue to the next steps.

  5. In another terminal launch VIO

  6. Finally, in another terminal, play your rosbag:
  ```bash
  rosbag play /PATH/TO/ROSBAG
  ```

## Provided config files
We provide three yaml files in /cfg to get you started. 

```jackal.yaml``` is setup to run a real-time demo. The predict step runs at the rate of available prediction poses (for which we used VIO). The update step processes incoming images as fast as computation limits will allow (in our case about 2.5 Hz) and discards all other images to prevent large latency.

```llff.yaml``` runs Loc-NeRF on the LLFF dataset as described in our paper.

```llff_global.yaml``` runs Loc-NeRF on the LLFF dataset with a wider spread of particles to test the ability to perform global localization as described in our paper.

# 3. Usage

Currently we provide example code to run Loc-NeRF on two types of data: running on LLFF data to compare with iNeRF (cite), and running a real-time 
demo with custom data. For both we use NeRF-Pytorch (cite) as our NeRF map.

The fastest way to start running Loc-NeRF is to download LLFF data with pre-trained NeRF weights. We also provide instructions for running a demo on a real robot, which requires training a NeRF with metric scaled poses.

## Using LLFF data

Download LLFF images and pretrained NeRF-Pytorch weights from [NeRF-Pytorch](https://github.com/yenchenlin/nerf-pytorch). If you download our fork of iNeRF here: 
[iNeRF](https://github.com/Dominic101/inerf) then the configs and ckpts folder will already be setup correctly with the pre-trained weights, and you just need to add the data folder from NeRF-Pytorch.

Place data using the following structure:

```
├── configs   
│   ├── ...
├── ckpts                                                                                                       
│   │   ├── fern
|   |   |   └── fern.tar                                                                                                                     
│   │   ├── fortress
|   |   |   └── fortress.tar                                                                                   
│   │   ├── horns
|   |   |   └── horns.tar   
│   │   ├── room
|   |   |   └── room.tar   
|   |   └── ...                                                                                 
                                                                                            
├── data                                                                                                                                                                                                       
│   ├── nerf_llff_data                                                                                                  
│   │   └── fern  # downloaded llff dataset                                                                                                                         
│   │   └── fortress  # downloaded llff dataset                                                                                  
│   │   └── horns   # downloaded llff dataset
|   |   └── room   # downloaded llff dataset
|   |   └── ...
```

After updating your yaml file ```llff.yaml``` with the directory where you placed the data and any other params you want to change, you are ready to 
run Loc-NeRF! By default, Loc-NeRF will estimate the camera pose of 5 random images from each of fern, fortress, horns, room (20 images in total). You can use rviz to provide real-time visualization of the ground truth pose and the particles.

### Plotting results

If you log results from Loc-NeRF, we provide code to plot the position and rotation error inside  ```tools/eval_logs.py```. The script also contains 
code to automatically run iNeRF using the same initial conditions that Loc-NeRF used and plot the results of iNeRF.

If you don't want to run iNeRF, running the plotting code is as easy as changing a few parameters at the top of ```tools/eval_logs.py``` and then running 
```python3 tools/eval_logs.py```. Note that ```run_inerf``` should be set to False.

#### Automatically generating iNeRF - Loc-NeRF comparison
Install our fork of [iNeRF](https://github.com/Dominic101/inerf) which is set up to interface with our automatic comparison script.

To automatically run iNeRF after you have logged data from Loc-NeRF (because we need to know the initial start for iNeRF), run the following:
``` python3 tools/eval_logs.py --config /home/dominic/inerf/configs/fern.txt --data_dir /home/dominic/inerf/data/nerf_llff_data/ --ckpt_dir /home/dominic/inerf/ckpts/ --N_importance 64 --batch_size 2048```. Note that ```run_inerf``` should be set to True. This will both run iNeRF and log the results so you don't need to rerun iNeRF. Note that to make the interface to iNeRF easier, we directly pass a specific config in the cmd line (in this case fern.txt), but setting N_importance and batch_size directly, along with the provided scripts, ensure that the correct params are used for each LLFF dataset.

## Using Custom data for real-time experiment

To run Loc-NeRF on your robot or simulator, you will first need to train a NeRF with metric scaled poses. This is important so that a movement estimated 
in the world by the predict step roughly corresponds to a movement inside the NeRF. There are several options for training a NeRF, but for ours we 
used https://colmap.github.io/faq.html#geo-registration. We first estimate a unscaled trajectory with Colmap and then 
provide Colmap with a small subset of scaled poses that it uses to scale the entire trajectory.

Our procedure is as follows:

1. Record a rosbag with RGB images (for NeRF) along with stereo images and IMU (for VIO).

2. use the provided script ```tools/ros_to_jpg.py``` to convert a downsampled set of images from the rosbag into images for Colmap

3. Use the imgs2poses.py script provided at [LLFF](https://github.com/fyusion/llff) to generate Colmap poses for the trajectory up to scale.

4. You should now have a set of poses up to scale. To add scale, you will need to determine the metric scaled pose of a small subset of images (we used five images) and use the provided Colmap function at
 [colmap-add-scale](https://colmap.github.io/faq.html#geo-registration) to generate optimized metric scaled poses for your dataset.

5. Use our fork of NeRF-Pytorch at [train-nerf](https://github.com/Dominic101/nerf-pytorch) to train a NeRF with metric scaled poses.

6. Now you are all set to run Loc-NeRF with a rosbag for a real-time demo.


 # 4. FAQ

 This code is under active development with more updates coming soon. Below are some known common issues/TODOs. We also welcome PRs and feedback of any encountered issues to keep improving this code for the community:

 1. We use ROS as a way to provide real-time visualzation with rviz and and as a structure to run on real robot platforms. For some users, a reliance on ROS may be an uneccessary requirement and future work is to separate Loc-NeRF from a ROS wrapper. 

  # Third-party code:
 Parts of this code were based on [this pytorch implementation of iNeRF](https://github.com/salykovaa/inerf) and [NeRF-Pytorch](https://github.com/yenchenlin/nerf-pytorch).

 ```
 NeRF-Pytorch:
 
 @misc{lin2020nerfpytorch,
  title={NeRF-pytorch},
  author={Yen-Chen, Lin},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished={\url{https://github.com/yenchenlin/nerf-pytorch/}},
  year={2020}
}
 ```
