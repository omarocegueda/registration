'''
Especialization of the registration optimizer to perform symmetric registration
'''
import time
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import registrationCommon as rcommon
import tensorFieldUtils as tf
import UpdateRule
from TransformationModel import TransformationModel
from SSDMetric import SSDMetric
from EMMetric import EMMetric
from RegistrationOptimizer import RegistrationOptimizer

class SymmetricRegistrationOptimizer(RegistrationOptimizer):
    r'''
    Performs the multi-resolution optimization algorithm for non-linear 
    registration using a given similarity metric and update rule (this
    scheme was inspider on the ANTS package).
    '''
    def get_default_parameters(self):
        return {'maxIter':[25, 50, 100], 'inversion_iter':20,
                'inversion_tolerance':1e-3, 'tolerance':1e-6, 
                'report_status':False}

    def __init__(self, 
                 fixed  =  None, 
                 moving = None, 
                 affine_fixed = None, 
                 affine_moving = None, 
                 similarity_metric = None, 
                 update_rule = None, 
                 parameters = None):
        super(SymmetricRegistrationOptimizer, self).__init__(
            fixed, moving, affine_fixed, affine_moving, similarity_metric, 
            update_rule, parameters)
        self.setMaxIter(self.parameters['maxIter'])
        self.tolerance = self.parameters['tolerance']
        self.inversion_tolerance = self.parameters['inversion_tolerance']
        self.inversion_iter = self.parameters['inversion_iter']
        self.report_status = self.parameters['report_status']

    def __connect_functions(self):
        r'''
        Assigns the appropriate functions to be called displacement field
        inversion according to the dimension of the input images
        '''
        if self.dim == 2:
            self.invert_vector_field = tf.invert_vector_field_fixed_point
            self.generate_pyramid = rcommon.pyramid_gaussian_2D
        else:
            self.invert_vector_field = tf.invert_vector_field_fixed_point3D
            self.generate_pyramid = rcommon.pyramid_gaussian_3D

    def __check_ready(self):
        r'''
        Verifies that the configuration of the optimizer and input data are 
        consistent and the optimizer is ready to run
        '''
        ready = True
        if self.fixed == None:
            ready = False
            print('Error: Fixed image not set.')
        elif self.dim != len(self.fixed.shape):
            ready = False
            print('Error: inconsistent dimensions. Last dimension update: %d.'
                  'Fixed image dimension: %d.'%(self.dim, 
                                                len(self.fixed.shape)))
        if self.moving == None:
            ready = False
            print('Error: Moving image not set.')
        elif self.dim != len(self.moving.shape):
            ready = False
            print('Error: inconsistent dimensions. Last dimension update: %d.'
                  'Moving image dimension: %d.'%(self.dim, 
                                                 len(self.moving.shape)))
        if self.similarityMetric == None:
            ready = False
            print('Error: Similarity metric not set.')
        if self.updateRule == None:
            ready = False
            print('Error: Update rule not set.')
        if self.maxIter == None:
            ready = False
            print('Error: Maximum number of iterations per level not set.')
        return ready

    def __init_optimizer(self):
        r'''
        Computes the Gaussian Pyramid of the input images and allocates 
        the required memory for the transformation models at the coarcest 
        scale.
        '''
        ready = self.__check_ready()
        self.__connect_functions()
        if not ready:
            print 'Not ready'
            return False
        self.moving_pyramid = [img for img 
                            in self.generate_pyramid(self.moving, 
                                                    self.levels-1, 
                                                    np.ones_like(self.moving))]
        self.fixed_pyramid = [img for img 
                           in self.generate_pyramid(self.fixed, 
                                                   self.levels-1, 
                                                   np.ones_like(self.fixed))]
        starting_forward = np.zeros(
            shape = self.fixed_pyramid[self.levels-1].shape+(self.dim,), 
            dtype = np.float64)
        starting_forward_inv = np.zeros(
            shape = self.fixed_pyramid[self.levels-1].shape+(self.dim,), 
            dtype = np.float64)
        self.forward_model.scale_affines(0.5**(self.levels-1))
        self.forward_model.set_forward(starting_forward)
        self.forward_model.set_backward(starting_forward_inv)
        starting_backward = np.zeros(
            shape = self.moving_pyramid[self.levels-1].shape+(self.dim,), 
            dtype = np.float64)
        starting_backward_inverse = np.zeros(
            shape = self.fixed_pyramid[self.levels-1].shape+(self.dim,), 
            dtype = np.float64)
        self.backward_model.scale_affines(0.5**(self.levels-1))
        self.backward_model.set_forward(starting_backward)
        self.backward_model.set_backward(starting_backward_inverse)

    def __end_optimizer(self):
        r'''
        Frees the resources allocated during initialization
        '''
        del self.moving_pyramid
        del self.fixed_pyramid

    def __iterate(self, show_images = False):
        r'''
        Performs one symmetric iteration:
            1.Compute forward
            2.Compute backward
            3.Update forward
            4.Update backward
            5.Compute inverses
            6.Invert the inverses to ensure the transformations are invertible
        '''
        #tic = time.time()
        wmoving = self.backward_model.warp_backward(self.current_moving)
        wfixed = self.forward_model.warp_backward(self.current_fixed)
        self.similarityMetric.setMovingImage(wmoving)
        self.similarityMetric.useMovingImageDynamics(self.current_moving, 
                                                     self.backward_model, -1)
        self.similarityMetric.setFixedImage(wfixed)
        self.similarityMetric.useFixedImageDynamics(self.current_fixed, 
                                                    self.forward_model, -1)
        self.similarityMetric.initializeIteration()
        ff_shape = np.array(self.forward_model.forward.shape).astype(np.int32)
        fb_shape = np.array(self.forward_model.backward.shape).astype(np.int32)
        bf_shape = np.array(self.backward_model.forward.shape).astype(np.int32)
        bb_shape = np.array(self.backward_model.backward.shape).astype(np.int32)
        del self.forward_model.backward
        del self.backward_model.backward
        fw_step = self.similarityMetric.computeForward()
        self.forward_model.forward, md_forward = self.updateRule.update(
            self.forward_model.forward, fw_step)
        del fw_step
        try:
            fw_energy = self.similarityMetric.energy
        except NameError:
            pass
        bw_step = self.similarityMetric.computeBackward()
        self.backward_model.forward, md_backward = self.updateRule.update(
            self.backward_model.forward, bw_step)
        del bw_step
        try:
            bw_energy = self.similarityMetric.energy
        except NameError:
            pass
        try:
            n_iter = len(self.energy_list)
            der = '-' 
            if len(self.energy_list)>=3:
                der = self.__get_energy_derivative()
            print('%d:\t%0.6f\t%0.6f\t%0.6f\t%s'%(n_iter , fw_energy, bw_energy,
                fw_energy + bw_energy, der))
            self.energy_list.append(fw_energy+bw_energy)
        except NameError:
            pass
        self.similarityMetric.freeIteration()
        inv_iter = self.inversion_iter
        inv_tol = self.inversion_tolerance
        self.forward_model.backward = np.array(
            self.invert_vector_field(
                self.forward_model.forward, fb_shape, inv_iter, inv_tol, None))
        self.backward_model.backward = np.array(
            self.invert_vector_field(
                self.backward_model.forward, bb_shape, inv_iter, inv_tol, None))
        self.forward_model.forward = np.array(
            self.invert_vector_field(
                self.forward_model.backward, ff_shape, inv_iter, inv_tol, 
                self.forward_model.forward))
        self.backward_model.forward = np.array(
            self.invert_vector_field(
                self.backward_model.backward, bf_shape, inv_iter, inv_tol, 
                self.backward_model.forward))
        if show_images:
            self.similarityMetric.report_status()
        #toc = time.time()
        #print('Iter time: %f sec' % (toc - tic))
        return md_forward+md_backward

    def __get_energy_derivative(self):
        r'''
        Returns the derivative of the estimated energy as a function of "time"
        (iterations) at the last iteration
        '''
        n_iter = len(self.energy_list)
        poly_der = np.poly1d(
            np.polyfit(range(n_iter), self.energy_list, 2)).deriv()
        der = poly_der(n_iter-1.5)
        return der

    def __report_status(self, level):
        r'''
        Shows the current overlaid images either on the common space or the
        reference space
        '''
        show_common_space = True
        if show_common_space:
            wmoving = self.backward_model.warp_backward(self.current_moving)
            wfixed = self.forward_model.warp_backward(self.current_fixed)
            self.similarityMetric.setMovingImage(wmoving)
            self.similarityMetric.useMovingImageDynamics(self.current_moving, 
                                                         self.backward_model, 
                                                         -1)
            self.similarityMetric.setFixedImage(wfixed)
            self.similarityMetric.useFixedImageDynamics(self.current_fixed, 
                                                        self.forward_model, 
                                                        -1)
            self.similarityMetric.initializeIteration()
            self.similarityMetric.report_status()
        else:
            phi1 = self.forward_model.forward
            phi2 = self.backward_model.backward
            phi1_inv = self.forward_model.backward
            phi2_inv = self.backward_model.forward
            phi, mean_disp = self.updateRule.update(phi1, phi2)
            phi_inv, mean_disp = self.updateRule.update(phi2_inv, phi1_inv)
            composition = TransformationModel(phi, phi_inv, None, None)
            composition.scale_affines(0.5**level)
            residual, stats = composition.compute_inversion_error()
            print('Current inversion error: %0.6f (%0.6f)'%(stats[1], stats[2]))
            wmoving = composition.warp_forward(self.current_moving)
            self.similarityMetric.setMovingImage(wmoving)
            self.similarityMetric.useMovingImageDynamics(self.current_moving, 
                                                         composition, 1)
            self.similarityMetric.setFixedImage(self.current_fixed)
            self.similarityMetric.useFixedImageDynamics(self.current_fixed, 
                                                        None, 1)
            self.similarityMetric.initializeIteration()
            self.similarityMetric.report_status()            

    def __optimize(self):
        r'''
        The main multi-scale symmetric optimization algorithm
        '''
        self.__init_optimizer()
        for level in range(self.levels-1, -1, -1):
            print 'Processing level', level
            self.current_fixed = self.fixed_pyramid[level]
            self.current_moving = self.moving_pyramid[level]
            self.similarityMetric.useOriginalFixedImage(
                self.fixed_pyramid[level])
            self.similarityMetric.useOriginalMovingImage(
                self.moving_pyramid[level])
            self.similarityMetric.setLevelsBelow(self.levels-level)
            self.similarityMetric.setLevelsAbove(level)
            if level < self.levels - 1:
                self.forward_model.upsample(self.current_fixed.shape, 
                                           self.current_moving.shape)
                self.backward_model.upsample(self.current_moving.shape, 
                                            self.current_fixed.shape)
            error = 1+self.tolerance
            niter = 0
            self.energy_list = []
            while (niter<self.maxIter[level]) and (self.tolerance<error):
                niter += 1
                error = self.__iterate()
            if self.report_status:
                self.__report_status(level)
        residual, stats = self.forward_model.compute_inversion_error()
        print('Forward Residual error (Symmetric diffeomorphism):%0.6f (%0.6f)'
              %(stats[1], stats[2]))
        residual, stats = self.backward_model.compute_inversion_error()
        print('Backward Residual error (Symmetric diffeomorphism):%0.6f (%0.6f)'
              %(stats[1], stats[2]))
        tf.append_affine_to_displacement_field(
            self.backward_model.backward, self.backward_model.affine_pre_inv)
        tf.prepend_affine_to_displacement_field(
            self.backward_model.forward, self.backward_model.affine_pre)
        self.forward_model.forward, mean_disp = self.updateRule.update(
            self.forward_model.forward, self.backward_model.backward)
        self.forward_model.backward, mean_disp_inv = self.updateRule.update(
            self.backward_model.forward, self.forward_model.backward)
        self.forward_model.affine_pre = None
        self.forward_model.affine_pre_inv = None
        self.forward_model.affine_post = None
        self.forward_model.affine_post_inv = None
        del self.backward_model
        residual, stats = self.forward_model.compute_inversion_error()
        print('Residual error (Symmetric diffeomorphism):%0.6f (%0.6f)'
              %(stats[1], stats[2]))
        self.__end_optimizer()

    def optimize(self):
        print 'Optimizer parameters:\n', self.parameters
        print 'Metric:', self.similarityMetric.getMetricName()
        print 'Metric parameters:\n', self.similarityMetric.parameters
        self.__optimize()

def test_optimizer_monomodal_2d():
    r'''
    Classical Circle-To-C experiment for 2D Monomodal registration
    '''
    fname_moving = 'data/circle.png'
    fname_fixed = 'data/C.png'
    moving = plt.imread(fname_moving)
    fixed = plt.imread(fname_fixed)
    moving = moving[:, :, 0].astype(np.float64)
    fixed = fixed[:, :, 1].astype(np.float64)
    moving = np.copy(moving, order = 'C')
    fixed = np.copy(fixed, order = 'C')
    moving = (moving-moving.min())/(moving.max() - moving.min())
    fixed = (fixed-fixed.min())/(fixed.max() - fixed.min())
    ################Configure and run the Optimizer#####################
    max_iter = [i for i in [25, 50, 100]]
    similarity_metric = SSDMetric({'symmetric':True, 
                                'lambda':5.0, 
                                'stepType':SSDMetric.GAUSS_SEIDEL_STEP})
    update_rule = UpdateRule.Composition()
    registration_optimizer = SymmetricRegistrationOptimizer(fixed, moving, 
                                                         None, None, 
                                                         similarity_metric, 
                                                         update_rule, max_iter)
    registration_optimizer.optimize()
    #######################show results#################################
    displacement = registration_optimizer.get_forward()
    direct_inverse = registration_optimizer.get_backward()
    moving_to_fixed = np.array(tf.warp_image(moving, displacement))
    fixed_to_moving = np.array(tf.warp_image(fixed, direct_inverse))
    rcommon.overlayImages(moving_to_fixed, fixed, True)
    rcommon.overlayImages(fixed_to_moving, moving, True)
    direct_residual, stats = tf.compose_vector_fields(displacement, 
                                                     direct_inverse)
    direct_residual = np.array(direct_residual)
    rcommon.plotDiffeomorphism(displacement, direct_inverse, direct_residual, 
                               'inv-direct', 7)

def test_optimizer_multimodal_2d(lambda_param):
    r'''
    Registers one of the mid-slices (axial, coronal or sagital) of each input 
    volume (the volumes are expected to be from diferent modalities and
    should already be affine-registered, for example Brainweb t1 vs t2)
    '''
    fname_moving = 'data/t2/IBSR_t2template_to_01.nii.gz'
    fname_fixed = 'data/t1/IBSR_template_to_01.nii.gz'
#    fnameMoving = 'data/circle.png'
#    fnameFixed = 'data/C.png'
    nifti = True
    if nifti:
        nib_moving  =  nib.load(fname_moving)
        nib_fixed  =  nib.load(fname_fixed)
        moving = nib_moving.get_data().squeeze().astype(np.float64)
        fixed = nib_fixed.get_data().squeeze().astype(np.float64)
        moving = np.copy(moving, order = 'C')
        fixed = np.copy(fixed, order = 'C')
        shape_moving = moving.shape
        shape_fixed = fixed.shape    
        moving = moving[:, shape_moving[1]//2, :].copy()
        fixed = fixed[:, shape_fixed[1]//2, :].copy()
        moving = (moving-moving.min())/(moving.max()-moving.min())
        fixed = (fixed-fixed.min())/(fixed.max()-fixed.min())
    else:
        nib_moving = plt.imread(fname_moving)
        nib_fixed = plt.imread(fname_fixed)
        moving = nib_moving[:, :, 0].astype(np.float64)
        fixed = nib_fixed[:, :, 1].astype(np.float64)
        moving = np.copy(moving, order = 'C')
        fixed = np.copy(fixed, order = 'C')
        moving = (moving-moving.min())/(moving.max() - moving.min())
        fixed = (fixed-fixed.min())/(fixed.max() - fixed.min())
    max_iter = [i for i in [25, 50, 100]]
    similarity_metric = EMMetric({'symmetric':True, 
                               'lambda':lambda_param, 
                               'stepType':SSDMetric.GAUSS_SEIDEL_STEP, 
                               'qLevels':256, 
                               'maxInnerIter':20,
                               'useDoubleGradient':True,
                               'maxStepLength':0.25})    
    update_rule = UpdateRule.Composition()
    print('Generating synthetic field...')
    #----apply synthetic deformation field to fixed image
    ground_truth = rcommon.createDeformationField2D_type2(fixed.shape[0], 
                                              fixed.shape[1], 8)
    warped_fixed = rcommon.warpImage(fixed, ground_truth)
    print('Registering T2 (template) to deformed T1 (template)...')
    plt.figure()
    rcommon.overlayImages(warped_fixed, moving, False)
    registration_optimizer = SymmetricRegistrationOptimizer(warped_fixed, 
                                                            moving,
                                                            None, None, 
                                                            similarity_metric, 
                                                            update_rule, 
                                                            max_iter)
    registration_optimizer.optimize()
    #######################show results#################################
    displacement = registration_optimizer.get_forward()
    direct_inverse = registration_optimizer.get_backward()
    moving_to_fixed = np.array(tf.warp_image(moving, displacement))
    fixed_to_moving = np.array(tf.warp_image(warped_fixed, direct_inverse))
    rcommon.overlayImages(moving_to_fixed, fixed_to_moving, True)
    direct_residual, stats = tf.compose_vector_fields(displacement, 
                                                     direct_inverse)
    direct_residual = np.array(direct_residual)
    rcommon.plotDiffeomorphism(displacement, direct_inverse, direct_residual, 
                               'inv-direct', 7)
    
    residual = ((displacement-ground_truth))**2
    mean_displacement_error = np.sqrt(residual.sum(2)*(warped_fixed>0)).mean()
    stdev_displacement_error = np.sqrt(residual.sum(2)*(warped_fixed>0)).std()
    print('Mean displacement error: %0.6f (%0.6f)'%
        (mean_displacement_error, stdev_displacement_error))

if __name__ == '__main__':
    start_time = time.time()
    test_optimizer_multimodal_2d(50)
    end_time = time.time()
    print('Registration time: %f sec' % (end_time - start_time))
    #testRegistrationOptimizerMonomodal2D()
    
#    import nibabel as nib
#    result = nib.load('data/circleToC.nii.gz')
#    result = result.get_data().astype(np.double)
#    plt.imshow(result)
