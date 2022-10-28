import matplotlib.pyplot as plt
import numpy as np
import glob
import gtsam
import sys

######################## user params ########################
run_inerf = False
# only needed if you run iNeRF
sys.path.insert(0, '/home/dominic')
sys.path.insert(0, '/home/dominic/inerf') # path to iNeRF root directory
from inerf.run import run

# font size to use for plots
fontsize = 17

# plot mean error or plot ratio of trials with position error < 5 cm and rotation error < 5 degrees
get_mean_error = False

# directory where logs are stored
log_directory = '/home/dominic/mocNeRF_ws/src/nav/logs/inerf_compare'

num_starts_per_dataset = 5 # number of trials per dataset
datasets = ['fern', 'horns', 'fortress', 'room']
# datasets = ['fern']

# input number of iterations runs by iNeRF and M
# note this is used for plotting number of forward passes and not for telling iNeRF what params to use
params = {'inerf': {'num_iterations': 300, 'batch_size': 2048, 'course_samples':64, 'fine_samples':64}}

##############################################################

def check_if_position_error_good(gt_pose, est_pose, get_mean_error=False):
    acceptable_error = 0.05 # meters
    error = np.linalg.norm(gt_pose.translation() - est_pose.translation())
    if get_mean_error:
        return error
    return error < acceptable_error

def check_if_rotation_error_good(gt_pose, est_pose, get_mean_error=False):
    acceptable_error = 5.0 # degrees
    est_t = np.transpose(est_pose.rotation().matrix())
    r_ab = est_t @ gt_pose.rotation().matrix()
    rot_error = np.rad2deg(np.arccos((np.trace(r_ab) - 1) / 2)) # TODO make 0 if nan
    
    if get_mean_error:
        return rot_error

    return abs(rot_error) < acceptable_error

def average_arrays(axis_list, million_scale = True):
    """
    average arrays of different size
    adapted from https://stackoverflow.com/questions/49037902/how-to-interpolate-a-line-between-two-other-lines-in-python/49041142#49041142

    forward_passes_all, accuracy
    """
    min_max_xs = [(min(axis), max(axis)) for axis in axis_list[0]]

    new_axis_xs = [np.linspace(min_x, max_x, 100) for min_x, max_x in min_max_xs]
    new_axis_ys = []
    for i in range(len(axis_list[0])):
        new_axis_ys.append(np.interp(new_axis_xs[i], axis_list[0][i], axis_list[1][i]))
    
    scale = 1.0
    if million_scale:
        scale = 1000000.0

    midx = [np.mean([new_axis_xs[axis_idx][i] for axis_idx in range(len(axis_list[0]))])/scale for i in range(100)]
    midy = [np.mean([new_axis_ys[axis_idx][i] for axis_idx in range(len(axis_list[0]))]) for i in range(100)]

    return midx, midy

def get_ratio(alg, get_mean_error=False, prefix=""):
    # alg can be 'mocnerf' or 'inerf'

    if prefix != "":
        prefix = prefix + "_"

    if alg == 'inerf':
        num_forward_passes_per_iteration = (params[alg]['course_samplese'] + params[alg]['fine_samplese']) * params[alg]['batch_size'] 
        forward_passes_count = [i * num_forward_passes_per_iteration / 1000000.0 for i in range(params[alg]['num_iterations'] + 1)]

    total_num_forward_passes_per_iteration = []
    total_position_error_good = []
    total_rotation_error_good = []

    for dataset_index, dataset_name in enumerate(datasets):
        gt_files = glob.glob(log_directory + '/gt_' + dataset_name + '*')
        for start_num, gt_file in enumerate(gt_files):
            img_num = gt_file.split('/')[-1].split('_')[2]
            gt_pose = gtsam.Pose3(np.load(gt_file))
            # print(gt_pose)

            if run_inerf and alg == 'inerf':
                # delta_phi, delta_theta, delta_psi, delta_x, delta_y, delta_z, obs_img_num, model_name
                pertubation_file = log_directory + "/initial_pose_" + dataset_name + "_" + str(img_num) + "_" + "poses.npy"
                pertubation_pose = np.load(pertubation_file)
                experiment_params =  [pertubation_pose[0] * 180.0/np.pi, pertubation_pose[1] * 180.0/np.pi, pertubation_pose[2] * 180.0/np.pi,
                                      pertubation_pose[3], pertubation_pose[4], pertubation_pose[5],
                                      int(img_num), dataset_name]
                # print(experiment_params)
                run(experiment_params, log_directory=log_directory, log_results=True)

            est_file = log_directory + "/" + alg + "_" + prefix + dataset_name + "_" + str(img_num) + "_" + "poses.npy"
            pose_estimates = np.load(est_file)

            if alg == 'mocnerf':
                iteration_file = log_directory + "/" + alg + "_" + prefix + dataset_name + "_" + str(img_num) + "_" + "forward_passes.npy"
                num_forward_passes_per_iteration = np.load(iteration_file)

            position_error_good = []
            rotation_error_good = []

            for pose_index in range(pose_estimates.shape[0]):
                est_pose_matrix = gtsam.Pose3(pose_estimates[pose_index])
                position_error_good.append(check_if_position_error_good(gt_pose, est_pose_matrix, get_mean_error))
                rotation_error_good.append(check_if_rotation_error_good(gt_pose, est_pose_matrix, get_mean_error))

            total_num_forward_passes_per_iteration.append(num_forward_passes_per_iteration)
            total_position_error_good.append(position_error_good)
            total_rotation_error_good.append(rotation_error_good)

    if alg == 'inerf':
        average_position_metric = np.average(np.array(total_position_error_good), axis=0)
        average_rotation_metric = np.average(np.array(total_rotation_error_good), axis=0)
        update_steps_count = None
    
    else:
        total_update_steps_per_iteration = [ [i for i in range(len(update_in_trial))] for update_in_trial in total_num_forward_passes_per_iteration]
        forward_passes_count, average_position_metric = average_arrays([total_num_forward_passes_per_iteration, total_position_error_good])
        forward_passes_count, average_rotation_metric = average_arrays([total_num_forward_passes_per_iteration, total_rotation_error_good])
        update_steps_count, average_position_metric = average_arrays([total_update_steps_per_iteration, total_position_error_good], million_scale=False)

    return forward_passes_count, average_position_metric, average_rotation_metric, update_steps_count

inerf_forward_passes_count, inerf_average_position_metric, inerf_average_rotation_metric, _ = get_ratio('inerf', get_mean_error)
mcl_forward_passes_count, mcl_average_position_metric, mcl_average_rotation_metric, _ = get_ratio('mocnerf', get_mean_error)
# Note the prefix argument can be used if you have multiple logs with different params from Loc-NeRF.
# For example, in this case we wanted to do an ablation study on running with and without annealing.
mcl_forward_passes_count_norefine, mcl_average_position_metric_norefine, mcl_average_rotation_metric_norefine, _ = get_ratio('mocnerf', get_mean_error, prefix="started")

position_plot = plt.figure(1, figsize=(8,4.5))
plt.plot(mcl_forward_passes_count, mcl_average_position_metric, label="Loc-NeRF w/ annealing", linewidth=2.5, color='b')
plt.plot(mcl_forward_passes_count_norefine, mcl_average_position_metric_norefine, label="Loc-NeRF w/o annealing", linewidth=2.5, color='k')
plt.plot(inerf_forward_passes_count[0:-1], inerf_average_position_metric[0:-1], label="iNeRF", linewidth=2.5, color='orange')
plt.xlabel("Number of Forward Passes (in millions)", fontsize=fontsize, family='Arial')
plt.grid()
plt.legend(loc='upper left', fontsize=fontsize-4)

if not get_mean_error:
    plt.ylim([0, 1.01])
    plt.title("Translation Accuracy Evaluation", fontsize=fontsize, family='Arial')
    plt.ylabel("Ratio of Trials with Translation \n Error < 5 cm", fontsize=fontsize, family='Arial')
else:
    plt.title("Average Translation Error for 20 Trials", fontsize=fontsize, family='Arial')
    plt.ylabel("error (m)", fontsize=fontsize, family='Arial')

plt.tight_layout()
plt.show()

rotation_plot = plt.figure(2, figsize=(8,4.5))
plt.plot(mcl_forward_passes_count, mcl_average_rotation_metric, label="Loc-NeRF w/ annealing", linewidth=2.5, color='b')
plt.plot(mcl_forward_passes_count_norefine, mcl_average_rotation_metric_norefine, label="Loc-NeRF w/o annealing", linewidth=2.5, color='k')
plt.plot(inerf_forward_passes_count[0:-1], inerf_average_rotation_metric[0:-1], label="iNeRF", linewidth=2.5, color='orange')
plt.xlabel("Number of Forward Passes (in millions)", fontsize=fontsize, family='Arial')
plt.grid()
plt.legend(loc='upper right', fontsize=fontsize-4)

if not get_mean_error:
    plt.ylim([0, 1.01])
    plt.title("Rotation Accuracy Evaluation", fontsize=fontsize, family='Arial')
    plt.ylabel("Ratio of Trials with Rotation Error < 5$^\circ$", fontsize=fontsize, family='Arial')
else:
    plt.title("Average Rotation Error for 20 Trials", fontsize=fontsize, family='Arial')
    plt.ylabel("error (deg)", fontsize=fontsize, family='Arial')
plt.tight_layout()
plt.show()

# To run iNeRF with this code use this (this makes sure that iNeRF params are set correctly for all runs):
#  python3 eval_logs.py --config /home/dominic/inerf/configs/fern.txt --data_dir /home/dominic/inerf/data/nerf_llff_data/ --ckpt_dir /home/dominic/inerf/ckpts/ --N_importance 64 --batch_size 2048