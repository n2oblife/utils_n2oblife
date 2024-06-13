# basd onthe scikit video man page

#skvideo.motion.blockMotion(videodata, method='DS', mbSize=8, p=2, **plugin_args


# Parameters:	

# videodata : ndarray, shape (numFrames, height, width, channel)

#     A sequence of frames

# method : string

#     “ES” –> exhaustive search

#     “3SS” –> 3-step search

#     “N3SS” –> “new” 3-step search [1]

#     “SE3SS” –> Simple and Efficient 3SS [2]

#     “4SS” –> 4-step search [3]

#     “ARPS” –> Adaptive Rood Pattern search [4]

#     “DS” –> Diamond search [5]

# mbSize : int

#     Macroblock size

# p : int

#     Algorithm search distance parameter

# Returns:	

# motionData : ndarray, shape (numFrames - 1, height/mbSize, width/mbSize, 2)

#     The motion vectors computed from videodata. The first element of the last axis contains the y motion component, and second element contains the x motion component.
