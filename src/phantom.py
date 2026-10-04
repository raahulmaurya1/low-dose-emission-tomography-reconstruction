# phantom.py: test images and ROIs.
# Takes: image size n and a lesion flag.
# Returns: fine truth [2n,2n], coarse truth [n,n] (2x2 block mean), mask [n,n],
#          and (lesion_roi, background_roi) boolean masks [n,n].
