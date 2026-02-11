import roi_info
import random
import numpy as np
def viewportDataGenerator(num_tile_per_seg,roi_tiles,roi_popularity,roi_info_idx,num_bandwidth_class):

    random.seed(10)
    np.random.seed(10)
    num_request_per_seg=num_bandwidth_class
    ret_vp_tiles=[]
    ret_not_vp_tiles=[]
    for i in range(num_request_per_seg):
        vp_tiles = []
        vp_rand_p = random.uniform(0,1)
        if (vp_rand_p <= roi_popularity):
            vp_roi_center_idx=random.randint(0,len(roi_info.viewport_center[roi_info_idx][0])-1)
            vp_center=roi_info.viewport_center[roi_info_idx][0][vp_roi_center_idx]
            vp_tiles=roi_info.generate_viewport(vp_center)
        else:
            vp_notroi_center_idx = random.randint(0, len(roi_info.viewport_center[roi_info_idx][1]) - 1)
            vp_center = roi_info.viewport_center[roi_info_idx][1][vp_notroi_center_idx]
            vp_tiles = roi_info.generate_viewport(vp_center)

        not_vp_tiles = []
        for j in range(num_tile_per_seg):
            if j not in vp_tiles:
                not_vp_tiles.append(j)
        ret_vp_tiles.append(vp_tiles)
        ret_not_vp_tiles.append(not_vp_tiles)

    return ret_vp_tiles,ret_not_vp_tiles

