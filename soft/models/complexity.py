import numpy as np


def complexity(in_map_szs, all_kernels, N_color_channels, learned_depths):
    """
    complexity([(32,32,64), [[(3,3),(3,3)],[(2,3),(3,5)]])
    """
    #np.save('../../../../../venv-seaborn/in_map_szs',in_map_szs)
    #np.save('../../../../../venv-seaborn/all_kernels', all_kernels)
    #np.save('../../../../../venv-seaborn/in_map_depths',learned_depths)

    out=0;
    
    depth_from_pre_layer = np.array([np.array([1 if i> 0 and j > 0 else 0 
        for (i,j) in k])for k in all_kernels[:-1]])
    fmap_per_layer =  [np.expand_dims(previous_layer,1)*computed_depth 
            for (previous_layer,computed_depth) in zip(depth_from_pre_layer,
                learned_depths[1:])]

    depth_per_layer = learned_depths[0].tolist() + [np.sum(l,0) for l in fmap_per_layer]

    for in_map_sz,in_map_depth,kernels in zip(in_map_szs, depth_per_layer, all_kernels):
        out+=complexity_1lyr(in_map_sz, in_map_depth, kernels)

    
    return out;

def complexity_1lyr(in_map_sz, in_map_depth, kernels):
    """
    e.g., complexity_1lyr(32, [3,2,1], (2,3,1))
    """
    out=0;
    for k,d in zip(kernels, in_map_depth):
        out+=(in_map_sz**2)*d*np.prod(k);


    return out;


## Run examples
#in_map_szs = [32]*3 + [16]*3 + [8]*3 + [1]
#all_kernels = [[(3,3)]*16]*3 + [[(3,3)]*32]*3 + [[(3,3)]*48]*2 + [[(3,3)]*64] + [[(1,1)]*10]
#print("complexity: %1.5g"% complexity(in_map_szs, all_kernels))
#in_map_szs=[112] + [56]*6 + [28]*8 + [14]*12 + [7]*6 + [1];
#all_kernels   =[[7]*64 ] + [[3]*64]*6 + [[3]*128]*8 + [[3]*256]*12 + [[3]*512]*6 +[[1]*1000]
#all_kernels = [[5]*64]*19 + [[1]*10]

#in_map_szs = [28]*3
#all_kernels = [[10]*32]*32 + [[1]*10]
#print("complexity: %1.5g"% complexity(in_map_szs, all_kernels,1))
