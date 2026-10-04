# metrics.py: PSNR, SSIM, lesion contrast, background noise, contrast recovery.
# Takes: truth [n,n] and rescaled reconstruction [n,n] in [0,1], mask and ROIs [n,n].
# Returns: a dict of scores (psnr, ssim, lesion_contrast, background_noise, contrast_recovery).
