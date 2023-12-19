import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import networkx as nx
import pickle as pkl


def df_to_avxy(df):
    av_df = df[df.OBJECT_TYPE=="AGENT"]
    xl, yl = av_df.X.to_list(), av_df.Y.to_list()
    return xl, yl

def df_to_centerline(avm, df,range):
    city_name = df.CITY_NAME.iloc[0]
    # xl, yl = df_to_avxy(df)
    xl, yl = df.X.to_list(), df.Y.to_list()
    lane_ids = []
    for x, y in zip(xl, yl):
        lane_ids.extend(avm.get_lane_ids_in_xy_bbox(x, y, city_name, range))
    lane_ids = list(set(lane_ids))
    centerline, c_len = [], []
    for lid in lane_ids:
        c = avm.get_lane_segment_centerline(lid, city_name)
        c_len.append(len(c))
        centerline.append(c)
    if len(set(c_len))>1:
        return None, None, None
    else:
        return np.array(centerline), lane_ids, city_name


def check_city_graph_exits_or_create_new_one(avm, city_name, dataset_path):
    CITY_NAME = {'PIT': '100054.csv', 'MIA': '100.csv'}
    if not os.path.exists(f"../cityGraph/{city_name}_graph"):
        print(f"{city_name} graph doesn't exit. Bulid graph...")
        city_csv_path = f"{dataset_path}{CITY_NAME[city_name]}"
        df = pd.read_csv(city_csv_path)
        xl, yl = df_to_avxy(df)
        lane_ids = []
        for x, y in zip(xl, yl):
            lane_ids.extend(avm.get_lane_ids_in_xy_bbox(x, y, city_name, 99999999))
        lane_ids = list(set(lane_ids))
        
        start_and_end = []
        for lid in lane_ids:
            centerline = avm.get_lane_segment_centerline(lid, city_name)
            start_and_end.append([centerline[0,:2],centerline[-1,:2]])
        start_and_end = np.asarray(start_and_end)

        edges = {}
        for lid_1, se_1 in tqdm(zip(lane_ids, start_and_end)):
            edges[lid_1] = []
            for lid_2, se_2 in zip(lane_ids, start_and_end):
                if lid_1 == lid_2:
                    continue
                end, start = se_1[-1], se_2[0]
                if (end == start).all():
                    edges[lid_1].append(lid_2)

        G = nx.DiGraph()
        G.add_nodes_from(lane_ids)
        for n1,v in edges.items():
            if len(v) > 0:
                for n2 in v:
                    G.add_edge(n1, n2) 
        with open(f"../cityGraph/{city_name}_graph","wb") as f:
            pkl.dump(G, f)
        print(f"Save in ../cityGraph/{city_name}_graph.")
    else:
        with open(f"../cityGraph/{city_name}_graph","rb") as f:
            G = pkl.load(f)
        print(f"{city_name} graph has existed.")
    return G

def break_path(paths, upper):
    pieces = []
    for i, path in enumerate(paths):
        # path = list(map(lambda x:lane_ids_to_abs_ids[x], path))
        if len(path) > upper:
            for start in range(len(path)):
                end = start + upper
                if end > len(path):
                    break
                sub_path = path[start:end]
                pieces.append(sub_path)
        else:
            pieces.append(path)
    return pieces

def G2path(sub_G,lane_ids, upper):
    '''
    2022-08-07 update
    '''
    paths, paths_ids = [], []
    for lid_src in lane_ids:
        in_degree = sub_G.in_degree(lid_src)
        out_degree = sub_G.out_degree(lid_src)
        if in_degree == 0 and out_degree != 0:
            tree = nx.dfs_tree(sub_G, lid_src) # wzz
            for lid_tgt in tree.nodes():
                out_degree = sub_G.out_degree(lid_tgt)
                if out_degree == 0:
                    path = nx.all_simple_paths(sub_G, source=lid_src, target=lid_tgt)
                    paths.extend(list(path))
        elif in_degree == 0 and out_degree == 0:
            paths.append([lid_src])
    lane_ids_to_abs_ids = dict(zip(lane_ids, np.arange(len(lane_ids))))
    if upper != 0:
        paths = break_path(paths, upper)
    
    for path in paths:
        paths_ids.append(list(map(lambda x:lane_ids_to_abs_ids[x], path)))
    # # verify
    # for _1, _2, in zip(paths, paths_ids):
    #     for i in range(len(_1)):
    #         e1, e2 = _1[i], _2[i]
    #         print(e1, e2, lane_ids_to_abs_ids[e1])
    return paths, paths_ids

def select_path(x, y, k, b, segment, paths_ids, centerline):
    seg_shape = segment.shape
    x, y = np.array(x), np.array(y)
    k_flat, b_flat = k.reshape(-1), b.reshape(-1)
    
    # (17 * 9, 2)
    seg_x, seg_y = segment[:,:,:,0].reshape(-1, 2), segment[:,:,:,1].reshape(-1, 2)
    
    # (50, 17 * 9)
    px = (k_flat * (y[:,np.newaxis] - b_flat) + x.reshape(-1,1)) / (k_flat**2 + 1)    
    py =  k_flat * px + b_flat
    
    # seg_x1 < px  < seg_x2, (50, 17 * 9)
    x_mask = (px >= seg_x.min(axis=-1)) & (px <= seg_x.max(axis=-1))

    # seg_y1 < py  < seg_y1, (50, 17 * 9)
    y_mask = (py >= seg_y.min(axis=-1)) & (py <= seg_y.max(axis=-1))
    
    # (50, 17 * 9)
    mask = x_mask & y_mask

    # range: (0, 50)。
    _mask = mask.reshape(-1, seg_shape[0], seg_shape[1]) # (50, 17, 9)
    indicator = _mask.sum(axis=-1) # (50, 17)
    all_False = np.where(indicator==0) # (223,) x 2
    all_False_seq_dim, all_False_lnum_dim  = all_False # (223,)
    if len(all_False) != 0:
        #  (223, 10)
        centerline_x, centerline_y = centerline[all_False_lnum_dim,:,0],  centerline[all_False_lnum_dim,:,1]
        # (223, )
        sub_differential_x, sub_differential_y = x[all_False_seq_dim], y[all_False_seq_dim] # (8,)
        # (223, 10)
        sub_differential_distance = ((centerline_x - sub_differential_x.reshape(-1,1))**2 + (centerline_y - sub_differential_y.reshape(-1,1))**2) ** 0.5

    # (50, 17 * 9)
    distance = ((px - x.reshape(-1,1)) ** 2  + (py - y.reshape(-1,1)) ** 2) ** 0.5
    distance -= 99999 * mask
    
    # (50, 17, 9)
    distance = distance.reshape(distance.shape[0], seg_shape[0], seg_shape[1])
    if len(all_False) != 0:
        distance[all_False_seq_dim, all_False_lnum_dim] =  sub_differential_distance[:,:-1] - 99999 # wzz

    selected_path, seleted_distance = [], 99999
    for path_ids in paths_ids:
        # (50, 17, 9) -> slice -> (50, 4, 9) -> reshape -> (50, 36) -> min -> (50,)
        dist = distance[:,path_ids,:].reshape(distance.shape[0],-1).min(axis=-1)
        dist = dist.mean()
        if dist <= seleted_distance:
            seleted_distance = dist
            selected_path = path_ids
    return selected_path

def select_all(selected_path, centerline, segment, k, b):
    centerline = centerline[selected_path,:,:2] # (5 * 10, 2)
    k, b = k[selected_path], b[selected_path]
    # (5, 9, 2, 3)
    segment = segment[selected_path]
    return centerline, segment, k, b