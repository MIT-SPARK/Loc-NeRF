import rospy
import numpy as np
import warnings
from full_filter import NeRF

 # Base class to handle loading params from yaml.

class NavigatorBase:
    def __init__(self, img_num=0, dataset_name=None):
        # extract params
        self.factor = rospy.get_param('factor')
        self.focal = rospy.get_param('focal')
        self.H = rospy.get_param('H')
        self.W = rospy.get_param('W')
        self.dataset_type = rospy.get_param('dataset_type')
        self.num_particles = rospy.get_param('num_particles')
        self.plot_particles  = rospy.get_param('visualize_particles')
        self.rgb_topic = rospy.get_param('rgb_topic')
        self.pose_topic = rospy.get_param('vio_topic')
        self.near = rospy.get_param('near')
        self.far = rospy.get_param('far')
        self.course_samples = rospy.get_param('course_samples')
        self.fine_samples = rospy.get_param('fine_samples')
        self.batch_size = rospy.get_param('batch_size')
        self.kernel_size = rospy.get_param('kernel_size')
        self.lrate = rospy.get_param('lrate')
        self.sampling_strategy = rospy.get_param('sampling_strategy')
        self.no_ndc = rospy.get_param('no_ndc')
        self.dil_iter = rospy.get_param('dil_iter')
        self.multires = rospy.get_param('multires')
        self.multires_views = rospy.get_param('multires_views')
        self.i_embed = rospy.get_param('i_embed')
        self.netwidth = rospy.get_param('netwidth')
        self.netdepth = rospy.get_param('netdepth')
        self.netdepth_fine = rospy.get_param('netdepth_fine')
        self.netwidth_fine = rospy.get_param('netwidth_fine')
        self.use_viewdirs = rospy.get_param('use_viewdirs')
        self.perturb = rospy.get_param('perturb')
        self.white_bkgd = rospy.get_param('white_bkgd')
        self.raw_noise_std = rospy.get_param('raw_noise_std')
        self.lindisp = rospy.get_param('lindisp')
        self.netchunk = rospy.get_param('netchunk')
        self.chunk = rospy.get_param('chunk')
        self.bd_factor = rospy.get_param('bd_factor')

        self.log_prefix = rospy.get_param('log_prefix')

        # just used for Nerf-Navigation comparison
        self.model_ngp = None
        self.ngp_opt = None

        if dataset_name is not None:
            self.model_name = dataset_name
        else:
            self.model_name = rospy.get_param('model_name')
        self.data_dir = rospy.get_param('data_dir') + "/" + self.model_name
        self.ckpt_dir = rospy.get_param('ckpt_dir') + "/" + self.model_name     

        self.obs_img_num = img_num

        # TODO these don't individually need to be part of the navigator class
        nerf_params = {'near':self.near, 'far':self.far, 'course_samples':self.course_samples, 'fine_samples':self.fine_samples,
                       'batch_size':self.batch_size, 'factor':self.factor, 'focal':self.focal, 'H':self.H, 'W':self.W, 'dataset_type':self.dataset_type,
                       'obs_img_num':self.obs_img_num, 'kernel_size':self.kernel_size, 'lrate':self.lrate, 'sampling_strategy':self.sampling_strategy,
                       'model_name':self.model_name, 'data_dir':self.data_dir, 'no_ndc':self.no_ndc, 'dil_iter':self.dil_iter,
                       'multires':self.multires, 'multires_views':self.multires_views, 'i_embed':self.i_embed, 'netwidth':self.netwidth, 'netdepth':self.netdepth,
                       'netdepth_fine':self.netdepth_fine, 'netwidth_fine':self.netwidth_fine, 'use_viewdirs':self.use_viewdirs, 'ckpt_dir':self.ckpt_dir,
                       'perturb':self.perturb, 'white_bkgd':self.white_bkgd, 'raw_noise_std':self.raw_noise_std, 'lindisp':self.lindisp,
                       'netchunk':self.netchunk, 'chunk':self.chunk, 'bd_factor':self.bd_factor}
        self.nerf = NeRF(nerf_params)
        
        self.image = None
        self.rgb_input_count = 0
        self.num_updates = 0
        self.photometric_loss = rospy.get_param('photometric_loss')

        self.view_debug_image_iteration = rospy.get_param('view_debug_image_iteration')

        self.px_noise = rospy.get_param('px_noise')
        self.py_noise = rospy.get_param('py_noise')
        self.pz_noise = rospy.get_param('pz_noise')
        self.rot_x_noise = rospy.get_param('rot_x_noise')
        self.rot_y_noise = rospy.get_param('rot_y_noise')
        self.rot_z_noise = rospy.get_param('rot_z_noise')

        self.use_convergence_protection = rospy.get_param('use_convergence_protection')
        self.number_convergence_particles = rospy.get_param('number_convergence_particles')
        self.convergence_noise = rospy.get_param('convergence_noise')

        self.use_weighted_avg = rospy.get_param('use_weighted_avg')

        self.min_number_particles = rospy.get_param('min_number_particles')
        self.use_particle_reduction = rospy.get_param('use_particle_reduction')

        self.alpha_refine = rospy.get_param('alpha_refine')
        self.alpha_super_refine = rospy.get_param('alpha_super_refine')

        self.run_predicts = rospy.get_param('run_predicts')
        self.use_received_image = rospy.get_param('use_received_image')
        self.run_inerf_compare = rospy.get_param('run_inerf_compare')
        self.global_loc_mode = rospy.get_param('global_loc_mode')
        self.run_nerfnav_compare = rospy.get_param('run_nerfnav_compare')
        self.nerf_nav_directory = rospy.get_param('nerf_nav_directory')
        self.center_about_true_pose = rospy.get_param('center_about_true_pose')
        self.use_refining = rospy.get_param('use_refining')
        self.log_results = rospy.get_param('log_results')
        self.log_directory = rospy.get_param('log_directory')
        self.use_logged_start = rospy.get_param('use_logged_start')
        self.forward_passes_limit = rospy.get_param('forward_passes_limit')

        self.min_bounds = rospy.get_param('min_bounds')
        self.max_bounds = rospy.get_param('max_bounds')

        self.R_bodyVins_camNerf = rospy.get_param('R_bodyVins_camNerf')
        
        self.previous_vio_pose = None
        self.nerf_pose = None
        self.all_pose_est = [] # plus 1 since we put in the initial pose before the first update
        self.img_msg = None
        
        # for now only have gt pose for llff dataset for inerf comparison and nerf-nav comparison
        self.gt_pose = None
        if not self.use_received_image:
            self.gt_pose = np.copy(self.nerf.obs_img_pose)
        
        self.check_params()

    def check_params(self):
        """
        Useful helper function to check if suspicious or invalid params are being used.
        TODO: Not all bad combinations of params are currently checked here.
        """

        if self.alpha_super_refine > self.alpha_refine:
            warnings.warn("alpha_super_refine is larger than alpha_refine, code will run but they are probably flipped by the user")
        
        if self.sampling_strategy != "random":
            warnings.warn("did not enter a valid sampling strategy. Currently the following are supported: random")

        if self.photometric_loss != "rgb":
            warnings.warn("did not enter a valid photometric loss. Currently the following are supported: rgb")