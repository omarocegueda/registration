"""
This script is the main launcher of the multi-modal non-linear image
registration
"""
import sys
import os
import numpy as np
import nibabel as nib
import registrationCommon as rcommon
import tensorFieldUtils as tf
import dipy.align.imwarp as imwarp
import dipy.align.metrics as metrics
import dipy.align.vector_fields as vf
from dipy.fixes import argparse as arg

parser = arg.ArgumentParser(
    description=
        "Multi-modal, non-linear image registration. By default, it does NOT "
        "save the resulting deformation field but only the deformed target "
        "image using tri-linear interpolation under two transformations: "
        "(1)the affine transformation used to pre-register the images and (2) "
        "the resulting displacement field. If a warp directory is specified, "
        "then each image inside the directory will be warped using "
        "nearest-neighbor interpolation under the resulting deformation field."
        " The name of the warped target image under the AFFINE transformation"
        "will be 'warpedAffine_'+baseTarget+'_'+baseReference+'.nii.gz', "
        "where baseTarget is the file name of the target image (excluding all "
        "characters at and after the first '.'), baseReference is the file "
        "name of the reference image (excluding all characters at and after "
        "the first '.'). For example, if target is 'a.nii.gz' and reference is"
        " 'b.nii.gz', then the resulting deformation field will be saved as "
        "'dispDiff_a_b.npy'.\n"
        "The name of the warped target image under the NON-LINEAR "
        "transformation will be "
        "'warpedAffine_'+baseTarget+'_'+baseReference+'.nii.gz', with the "
        "convention explained above.\n"

        "Similarly, the name of each warped image inside the warp directory "
        "will be 'warpedDiff_'+baseWarp+'_'+baseReference+'.nii.gz' where "
        "baseWarp is the name of the corresponding image following the same "
        "convetion as above.\n"
        "Example:\n"
        "python dipyreg.py target.nii.gz reference.nii.gz "
        "ANTS_affine_target_reference.txt --smooth=25.0 --iter 10,50,50"
    )

parser.add_argument(
    'target', action = 'store', metavar = 'target',
    help = '''Nifti1 image or other formats supported by Nibabel''')

parser.add_argument(
    'reference', action = 'store', metavar = 'reference',
    help = '''Nifti1 image or other formats supported by Nibabel''')

parser.add_argument(
    'affine', action = 'store', metavar = 'affine',
    help = '''ANTS affine registration matrix (.txt) that registers target to
           reference''')

parser.add_argument(
    'warp_dir', action = 'store', metavar = 'warp_dir',
    help = '''Directory (relative to ./ ) containing the images to be warped
           with the obtained deformation field''')

parser.add_argument(
    '-m', '--metric', action = 'store', metavar = 'metric',
    help = '''Any of {EM[L], CC[L]} specifying the metric to be used
    SSD=sum of squared diferences (monomodal), EM=Expectation Maximization
    to fit the transfer functions (multimodal), CC=Cross Correlation (monomodal
    and some multimodal) and the comma-separated (WITH NO SPACES) parameter list L:
    EM[step_lentgh,lambda,qLevels,max_inner_iter,step_type]
        step_length: the maximum norm among all vectors of the displacement at each iteration
        lambda: the smoothing parameter (the greater the smoother)
        qLevels: number of quantization levels (hidden variables) in the EM formulation
        max_inner_iter: maximum number of iterations of each level of the multi-resolution Gauss-Seidel algorithm
        step_type : energy minimization step, either 'v_cycle' (Newton step using multi-resolution GS) or 'demons'
        e.g.: EM[0.25,25.0,256,20,'v_cycle'] (NO SPACES)
    CC[step_length,sigma_smooth,neigh_radius]
        step_length: the maximum norm among all vectors of the displacement at each iteration
        sigma_smooth: std. dev. of the smoothing kernel to be used to smooth the gradient at each step
        neigh_radius: radius of the squared neighborhood to be used to compute the Cross Correlation at each voxel
        e.g.:CC[0.25,3.0,4] (NO SPACES)
    ''',
    default = 'CC[0.25,3.0,4]')

parser.add_argument(
    '-i', '--iter', action = 'store', metavar = 'i_0,i_1,...,i_n',
    help = '''A comma-separated (WITH NO SPACES) list of integers indicating the maximum number
           of iterations at each level of the Gaussian Pyramid (similar to
           ANTS), e.g. 10,100,100 (NO SPACES)''',
    default = '25,50,100')

parser.add_argument(
    '-inv_iter', '--inversion_iter', action = 'store', metavar = 'max_iter',
    help = '''The maximum number of iterations for the displacement field
           inversion algorithm''',
    default='20')

parser.add_argument(
    '-inv_tol', '--inversion_tolerance', action = 'store',
    metavar = 'tolerance',
    help = '''The tolerance for the displacement field inversion algorithm''',
    default = '1e-3')

parser.add_argument(
    '-ii', '--inner_iter', action = 'store', metavar = 'max_iter',
    help = '''The number of Gauss-Seidel iterations to be performed to minimize
           each linearized energy''',
    default='20')

parser.add_argument(
    '-ql', '--quantization_levels', action = 'store', metavar = 'qLevels',
    help = '''The number levels to be used for the EM quantization''',
    default = '256')

parser.add_argument(
    '-single', '--single_gradient', action = 'store_true',
    help = '''The number levels to be used for the EM quantization''')

parser.add_argument(
    '-rs', '--report_status', action = 'store_true',
    help = '''Instructs the algorithm to show the overlaid registered images
           after each pyramid level''')

parser.add_argument(
    '-aff', '--affine_only', dest = 'output_list',
    action = 'append_const', const='affine_only',
    help = r'''Indicates that only affine registration (provided as parameter)
           will be performed to warp the target images''')

parser.add_argument(
    '-sd', '--save_displacement', dest = 'output_list',
    action = 'append_const', const='displacement',
    help = r'''Specifies that the displacement field must be saved. The
           displacement field will be saved in .npy format. The filename will
           be the concatenation:
           'dispDiff_'+baseTarget+'_'+baseReference+'.npy'
           where baseTarget is the file name of the target image (excluding all
           characters at and after the first '.'), baseReference is the file
           name of the reference image (excluding all characters at and after
           the first '.'). For example, if target is 'a.nii.gz' and reference
           is 'b.nii.gz', then the resulting deformation field will be saved as
           'dispDiff_a_b.npy'.''')

parser.add_argument(
    '-si', '--save_inverse', dest = 'output_list', action = 'append_const',
    const = 'inverse',
    help = r'''Specifies that the inverse displacement field must be saved.
           The displacement field will be saved in .npy format. The filename
           will be the concatenation:
           'invDispDiff_'+baseTarget+'_'+baseReference+'.npy'
           where baseTarget is the file name of the target image (excluding all
           characters at and after the first '.'), baseReference is the file
           name of the reference image (excluding all characters at and after
           the first '.'). For example, if target is 'a.nii.gz' and reference
           is 'b.nii.gz', then the resulting deformation field will be saved as
           'invDispDiff_a_b.npy'.''')

parser.add_argument(
    '-sl', '--save_lattice', dest = 'output_list', action = 'append_const',
    const = 'lattice',
    help = r'''Specifies that the deformation lattice (i.e., the obtained
           deformation field applied to a regular lattice) must be saved. The
           displacement field will be saved in .npy format. The filename will
           be the concatenation:
           'latticeDispDiff_'+baseTarget+'_'+baseReference+'.npy'
           where baseTarget is the file name of the target image (excluding all
           characters at and after the first '.'), baseReference is the file
           name of the reference image (excluding all characters at and after
           the first '.'). For example, if target is 'a.nii.gz' and reference
           is 'b.nii.gz', then the resulting deformation field will be saved as
           'latticeDispDiff_a_b.npy'.''')

parser.add_argument(
    '-sil', '--save_inverse_lattice', dest = 'output_list',
    action = 'append_const', const = 'inv_lattice',
    help = r'''Specifies that the inverse deformation lattice (i.e., the
           obtained inverse deformation field applied to a regular lattice)
           must be saved. The displacement field will be saved in .npy format.
           The filename will be the concatenation:
           'invLatticeDispDiff_'+baseTarget+'_'+baseReference+'.npy'
           where baseTarget is the file name of the target image (excluding all
           characters at and after the first '.'), baseReference is the file
           name of the reference image (excluding all characters at and after
           the first '.'). For example, if target is 'a.nii.gz' and reference
           is 'b.nii.gz', then the resulting deformation field will be saved as
           'invLatticeDispDiff_a_b.npy'.''')

def check_arguments(params):
    r'''
    Verify all arguments were correctly parsed and interpreted
    '''
    print(params.target, params.reference, params.affine, params.warp_dir)
    print('iter:', [int(i) for i in params.iter.split(',')])
    print(params.inner_iter, params.quantization_levels,
          params.single_gradient)
    print('Inversion:', params.inversion_iter, params.inversion_tolerance)
    print('---------Output requested--------------')
    print(params.output_list)

def save_deformed_lattice_3d(displacement, oname):
    r'''
    Applies the given mapping to a regular lattice and saves the resulting
    image to a Nifti file with the given name
    '''
    min_val, max_val = vf.get_displacement_range(displacement, None)
    shape = np.array([np.ceil(max_val[0]), np.ceil(max_val[1]),
                  np.ceil(max_val[2])], dtype = np.int32)
    lattice = np.array(rcommon.drawLattice3D(shape, 10), dtype = np.float32)
    warped = np.array(vf.warp_volume(lattice, displacement)).astype(np.int16)
    img = nib.Nifti1Image(warped, np.eye(4))
    img.to_filename(oname)

def save_registration_results(mapping, params):
    r'''
    Warp the target image using the obtained deformation field
    '''
    import os
    import ibsrutils
    fixed = nib.load(params.reference)
    fixed_affine = fixed.get_affine()
    reference_shape = np.array(fixed.shape, dtype=np.int32)
    warp_dir = params.warp_dir
    base_moving = rcommon.getBaseFileName(params.target)
    base_fixed = rcommon.getBaseFileName(params.reference)
    moving = nib.load(params.target).get_data().squeeze().astype(np.float64)
    moving = moving.copy(order='C')
    warped = np.array(mapping.transform(moving, None, 'tri')).astype(np.int16)
    img_warped = nib.Nifti1Image(warped, fixed_affine)
    img_warped.to_filename('warpedDiff_'+base_moving+'_'+base_fixed+'.nii.gz')
    #---warp all volumes in the warp directory using NN interpolation
    names = [os.path.join(warp_dir, name) for name in os.listdir(warp_dir)]
    for name in names:
        to_warp = nib.load(name).get_data().squeeze().astype(np.int32)
        to_warp = to_warp.copy(order='C')
        base_warp = rcommon.getBaseFileName(name)
        warped = np.array(mapping.transform(to_warp, None, 'nn')).astype(np.int16)
        img_warped = nib.Nifti1Image(warped, fixed_affine)
        img_warped.to_filename('warpedDiff_'+base_warp+'_'+base_fixed+'.nii.gz')
    #---now the jaccard indices
    if os.path.exists('jaccard_pairs.lst'):
        with open('jaccard_pairs.lst','r') as f:
            for line in f.readlines():
                aname, bname, cname= line.strip().split()
                abase = rcommon.getBaseFileName(aname)
                bbase = rcommon.getBaseFileName(bname)
                aname = 'warpedDiff_'+abase+'_'+bbase+'.nii.gz'
                if os.path.exists(aname) and os.path.exists(cname):
                    ibsrutils.computeJacard(cname, aname)
                else:
                    print 'Pair not found ['+cname+'], ['+aname+']'
    #---finally, the optional output
    if params.output_list == None:
        return
    if 'lattice' in params.output_list:
        save_deformed_lattice_3d(
            mapping.forward,
            'latticeDispDiff_'+base_moving+'_'+base_fixed+'.nii.gz')
    if 'inv_lattice' in params.output_list:
        save_deformed_lattice_3d(
            mapping.backward, 'invLatticeDispDiff_'+base_moving+'_'+base_fixed+'.nii.gz')
    if 'displacement' in params.output_list:
        np.save('dispDiff_'+base_moving+'_'+base_fixed+'.npy', mapping.forward)
    if 'inverse' in params.output_list:
        np.save('invDispDiff_'+base_moving+'_'+base_fixed+'.npy', mapping.backward)

def register_3d(params):
    r'''
    Runs the non-linear registration with the parsed parameters
    '''
    print('Registering %s to %s'%(params.target, params.reference))
    sys.stdout.flush()
    metric_name=params.metric[0:params.metric.find('[')]
    metric_params_list=params.metric[params.metric.find('[')+1:params.metric.find(']')].split(',')

    #Initialize the appropriate metric
    if metric_name=='EM':
        smooth=float(metric_params_list[1])
        inner_iter=int(metric_params_list[3])
        step_length=float(metric_params_list[0])
        q_levels=int(metric_params_list[2])
        double_gradient=False if params.single_gradient else True
        iter_type = metric_params_list[4]
        similarity_metric = metrics.EMMetric(
            3, smooth, inner_iter, step_length, q_levels, double_gradient, iter_type)
    elif metric_name=='CC':
        step_length = float(metric_params_list[0])
        sigma_diff = float(metric_params_list[1])
        radius = int(metric_params_list[2])
        similarity_metric = metrics.CCMetric(3, step_length, sigma_diff, radius)
    #Initialize the optimizer
    opt_iter = [int(i) for i in params.iter.split(',')]
    opt_tol = 1e-4
    inv_iter = int(params.inversion_iter)
    inv_tol = float(params.inversion_tolerance)
    registration_optimizer = imwarp.SymmetricDiffeomorphicRegistration(
        similarity_metric, opt_iter, opt_tol, inv_iter, inv_tol)
    #Load the data
    moving = nib.load(params.target)
    moving_affine = moving.get_affine()
    fixed = nib.load(params.reference)
    fixed_affine = fixed.get_affine()
    print 'Affine:', params.affine
    if not params.affine:
        transform = np.eye(4)
    else:
        transform = rcommon.readAntsAffine(params.affine)
    init_affine = np.linalg.inv(moving_affine).dot(transform.dot(fixed_affine))
    #Preprocess the data
    moving = moving.get_data().squeeze().astype(np.float64)
    fixed = fixed.get_data().squeeze().astype(np.float64)
    moving = moving.copy(order='C')
    fixed = fixed.copy(order='C')
    moving = (moving-moving.min())/(moving.max()-moving.min())
    fixed = (fixed-fixed.min())/(fixed.max()-fixed.min())
    #Run the registration
    if params.output_list is not None and 'affine_only' in params.output_list:
        print('Applying affine only')
        sh_direct=fixed.shape + (3,)
        sh_inv=moving.shape + (3,)
        direct = np.zeros(shape = sh_direct, dtype=np.float32)
        inv = np.zeros(shape = sh_inv, dtype=np.float32)
        mapping=imwarp.DiffeomorphicMap(3, direct, inv, None, init_affine)    
    else:
        registration_optimizer.verbosity = 2
        mapping = registration_optimizer.optimize(fixed, moving, fixed_affine, moving_affine, transform)
        mapping.consolidate()
    del registration_optimizer
    del similarity_metric
    save_registration_results(mapping, params)

def test_exec():
    target='target/IBSR_01_ana_strip.nii.gz'
    reference='reference/t1_icbm_normal_1mm_pn0_rf0_peeled.nii.gz'
    affine='IBSR_01_ana_strip_t1_icbm_normal_1mm_pn0_rf0_peeledAffine.txt'
    paramiter='0x30x30'
    inversion_iter='20'
    inversion_tolerance='1e-3'
    report_status=True
    print('Registering %s to %s'%(target, reference))
    sys.stdout.flush()
    ####Initialize parameter dictionaries####
    metric_parameters = {
        'max_step_length':0.25,
        'sigma_diff':3.0,
        'radius':4}
    similarity_metric = CCMetric(3, metric_parameters)
    optimizer_parameters = {
        'max_iter':[int(i) for i in paramiter.split(',')],
        'inversion_iter':int(inversion_iter),
        'inversion_tolerance':float(inversion_tolerance),
        'report_status':True if report_status else False}
    moving = nib.load(target)
    moving_affine = moving.get_affine()
    fixed = nib.load(reference)
    fixed_affine = fixed.get_affine()
    print 'Affine:', affine
    if not affine:
        transform = np.eye(4)
    else:
        transform = rcommon.readAntsAffine(affine)
    init_affine = np.linalg.inv(moving_affine).dot(transform.dot(fixed_affine))
    #print initAffine
    moving = moving.get_data().squeeze().astype(np.float64)
    fixed = fixed.get_data().squeeze().astype(np.float64)
    moving = moving.copy(order='C')
    fixed = fixed.copy(order='C')
    moving = (moving-moving.min())/(moving.max()-moving.min())
    fixed = (fixed-fixed.min())/(fixed.max()-fixed.min())
    ###################Run registration##################

    update_rule = UpdateRule.Composition()
    registration_optimizer = SymmetricRegistrationOptimizer(
        fixed, moving, None, init_affine, similarity_metric, update_rule,
        optimizer_parameters)
    registration_optimizer.optimize()
    #displacement = registration_optimizer.get_forward()
    #inverse = registration_optimizer.get_backward()
    del registration_optimizer
    del similarity_metric
    del update_rule
    #save_registration_results(init_affine, displacement, inverse, params)

def test_scale_space():
    import nibabel as nib
    import registrationCommon as rcommon
    import dipy.align.imwarp as imwarp
    target='target/IBSR_01_ana_strip.nii.gz'
    nib_moving = nib.load(target)
    zooms = np.asarray(nib_moving.get_header().get_zooms())
    zooms[2]*=20
    zooms*=3
    moving = nib_moving.get_data().squeeze().astype(np.float64)
    moving = (moving-moving.min())/(moving.max()-moving.min())
    moving_ss = [(filtered, size, spacing, scaling) 
        for (filtered, size, spacing, scaling) in imwarp.scale_space(moving, 3, zooms)]
    print("Scale space len: %d"%(len(moving_ss),))
    for i in range(len(moving_ss)):
        rcommon.plot_middle_slices(moving_ss[i][0])
    moving_ss = [(filtered, size, spacing, scaling) 
        for (filtered, size, spacing, scaling) in imwarp.scale_space(moving[:,:,moving.shape[2]//2], 3, zooms)]
    plt.figure()
    plt.imshow(moving_ss[0][0])




if __name__ == '__main__':
    register_3d(parser.parse_args())
