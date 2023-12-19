import numpy as np

def centerline_to_segment(centerline):
    centerline_p1 = np.expand_dims(centerline[:,:-1,:],axis=2)
    centerline_p2 = np.expand_dims(centerline[:,1:,:],axis=2)
    segment = np.concatenate([centerline_p1, centerline_p2], axis=2)
    return segment

def points_to_kb(segment):
    # p (l_num, 9, 2, 3)
    p1, p2 = segment[:,:,0,:2], segment[:,:,1,:2] # (l_num,9,2)
    x1, y1, x2, y2 = p1[:,:,0], p1[:,:,1], p2[:,:,0], p2[:,:,1]
    k = (y1-y2)/(x1-x2) # (l_num, 9)
    b = y1 - k*x1 # (l_num, 9)
    return k, b 


def find_projection(x, y, k, b, segment, centerline):
    # k, b (17, 9)
    # segment (17, 9, 2, 3)
    x, y = np.array(x), np.array(y)
    
    ref_path = centerline[:,:,:2].reshape(-1, 2) # (50, 2)

    k_flat, b_flat = k.reshape(-1), b.reshape(-1)

    # (5 * 9, 2) * 2
    seg_x, seg_y = segment[:,:,:,0].reshape(-1, 2), segment[:,:,:,1].reshape(-1, 2)
    
    # (50, 5 * 9)
    px = (k_flat * (y[:,np.newaxis] - b_flat) + x.reshape(-1,1)) / (k_flat**2 + 1)    
    py =  k_flat * px + b_flat

    # seg_x1 < px  < seg_x2, (50, 5 * 9)
    x_mask = (px >= seg_x.min(axis=-1)) & (px <= seg_x.max(axis=-1))

    # seg_y1 < py  < seg_y1, (50, 5 * 9)
    y_mask = (py >= seg_y.min(axis=-1)) & (py <= seg_y.max(axis=-1))
    
    # (50, 5 * 9)
    mask = x_mask & y_mask

    # (50, 5 * 9)
    distance = ((px - x.reshape(-1,1)) ** 2  + (py - y.reshape(-1,1)) ** 2) ** 0.5
    distance -= 99999 * mask
    
    # (50, )
    p_idx = np.argmin(distance.reshape(distance.shape[0], -1), axis=-1)
    nearest_px = px[np.arange(px.shape[0]), p_idx]
    nearest_py = py[np.arange(py.shape[0]), p_idx]
    # (50, 2)
    projection = np.concatenate([np.expand_dims(nearest_px,axis=1), np.expand_dims(nearest_py,axis=1)], axis=1)
    
    # (50)
    zero_detector = mask.sum(axis=-1)
    # (...)
    zero_detector = np.where(zero_detector==0)
    if len(zero_detector) != 0:
        
        # (n, 2), n undifferential points
        undifferential_points = np.hstack([x[zero_detector].reshape(-1,1), y[zero_detector].reshape(-1,1)])
        # (n ,50)
        dist_betw_undiff_p_with_ref = (((undifferential_points[:,np.newaxis] - ref_path)**2).sum(axis=-1))**0.5
        # (n)
        nearest_ref_point_idx = dist_betw_undiff_p_with_ref.argmin(axis=-1)
        # (n, 2)
        nearest_ref_point = ref_path[nearest_ref_point_idx]
        projection[zero_detector] = nearest_ref_point
    return projection

def get_direction(line, point):
    '''
    see if the point on the left or right to the segment.
    '''
    # point (50, 2)
    # line (50, 2, 2)
    aX = line[:, 0, 0] # 50
    aY = line[:, 0, 1] # 50
    bX = line[:, 1, 0] # 50
    bY = line[:, 1, 1] # 50
    cX = point[:, 0] # 50
    cY = point[:, 1] # 50

    val = ((bX - aX)*(cY - aY) - (bY - aY)*(cX - aX))
    side = val >= 0 # online, left -> left
    return side

def projection_to_sd(centerline, projection, xl, yl, k, b ,segment):
    # centerline (161, 10, 3), 
    # projection, (50, 2) 
    # xl 50, yl 50, 
    # k, b (161, 9)
    # segment (161, 9, 2, 3)
    ref_path = centerline[:,:,:2].reshape(-1, 2) # (50, 2)
    
    k, b = k.reshape(-1), b.reshape(-1)
        
    xy = np.array([xl, yl]).T
    redundancy_id = np.arange(0, ref_path.shape[0], centerline.shape[1])[1:]
    ref_path = np.delete(ref_path, redundancy_id, axis=0) # (46, 2)

    horizon_dist = (((ref_path[1:,:] - ref_path[:-1,:])**2).sum(axis=-1))**0.5 # (45)
    size = horizon_dist.shape[0]
    mask = np.tile(np.arange(size),(size, 1))
    dia = np.arange(size).reshape(size,-1)
    mask = mask <= dia
    horizon_dist = (horizon_dist * mask).sum(axis=-1)

    # (50, 45)
    _y = np.outer(projection[:,0], k) + b
    line_indicator = (_y - projection[:, 1].reshape(-1,1)) <= 1e-8

    # (5 * 9, 2) * 2
    seg_x, seg_y = segment[:,:,:,0].reshape(-1, 2), segment[:,:,:,1].reshape(-1, 2)
    # (50, 45) * 2
    px, py = np.tile(projection[:,0].reshape(-1,1), (1, seg_x.shape[0])), np.tile(projection[:,1].reshape(-1,1),(1, seg_x.shape[0]))
    x_mask = (px >= seg_x.min(axis=-1)) & (px <= seg_x.max(axis=-1))
    # seg_y1 < py  < seg_y1, (50, 5 * 9)
    y_mask = (py >= seg_y.min(axis=-1)) & (py <= seg_y.max(axis=-1))
    # (50, 5 * 9)
    segment_indicator = x_mask & y_mask
    # (50, 45)
    indicator = line_indicator & segment_indicator
    # 50, range(0-45)
    nearest_idx = (indicator!=False).argmax(axis=-1) 
    
    # base_dist - dist(proj, post_point)
    base_dist = horizon_dist[nearest_idx] # 50
    post_points = ref_path[nearest_idx + 1] # (50,2)
    diff_dist = (((post_points - projection)**2).sum(axis=-1))**0.5
    s = base_dist - diff_dist
    d = (((xy - projection)**2).sum(axis=-1))**0.5
    
    seg_xy = np.concatenate([np.expand_dims(seg_x,axis=-1),np.expand_dims(seg_y,axis=-1)],axis=-1)
    seg_xy = seg_xy[nearest_idx]
    direction = get_direction(seg_xy, xy)
    return s, d, direction

def cartesian_to_frenet(track_x, track_y, center_line):
    if center_line.ndim == 2:
        center_line = center_line.reshape(1, center_line.shape[0], center_line.shape[1])
    segment = centerline_to_segment(center_line)
    k, b = points_to_kb(segment)
    projection = find_projection(track_x, track_y, k, b, segment, centerline=center_line)
    s, d, direction = projection_to_sd(center_line, projection, track_x, track_y, k, b ,segment)
    return s, d, direction, projection

