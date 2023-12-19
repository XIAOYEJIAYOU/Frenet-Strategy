import matplotlib.pyplot as plt

def plot_centerline(centerline, selected_path=None, paths_ids=None, mode="-"):
    if mode == "-":
        for ids, lane in enumerate(centerline):
            plt.plot(lane[:,0],lane[:,1])
            plt.scatter(lane[:,0],lane[:,1])
    elif mode == "highlight":
        assert selected_path != None 
        for ids, lane in enumerate(centerline):
            color = "red" if ids in selected_path else "black"
            plt.plot(lane[:,0],lane[:,1],c=color)
            plt.scatter(lane[:,0],lane[:,1],c=color)
    elif mode == "colorful":
        assert paths_ids != None
        for ids in paths_ids:
            lane = centerline[ids,:,:2].reshape(-1,2)
            plt.plot(lane[:,0],lane[:,1],alpha=0.3, lw=5)
            plt.scatter(lane[:,0],lane[:,1],marker="o",alpha=0.3)
    elif mode == "black":
        for ids, lane in enumerate(centerline):
            plt.plot(lane[:,0],lane[:,1],c="black")
            plt.scatter(lane[:,0],lane[:,1],c="black")

def plot_track(xl, yl, color="blue"):
    plt.scatter(xl, yl, c= color)
    
def plot_projection(projection, color="orange", xl=None, yl=None, connection=False):
    plt.scatter(projection[:,0], projection[:,1],c=color)
    if connection:
        assert xl is not None and yl is not None
        for p, x, y in zip(projection, xl, yl):
            plt.plot([p[0], x],[p[1],y],c="gray")
            
def fig_cut(xl, yl, projection):
    x_min = min(min(xl),min(projection[:,0]))
    x_max = max(max(xl),max(projection[:,0]))
    y_min = min(min(yl),min(projection[:,1]))
    y_max = max(max(yl),max(projection[:,1]))
    
    x_len = x_max - x_min
    y_len = y_max - y_min
    x_cent = int((x_max + x_min) / 2)
    y_cent = int((y_max + y_min) / 2)
    length = int(max(x_len, y_len) / 2)
    plt.xlim(x_cent-1.5*length, x_cent+1.5*length)
    plt.ylim(y_cent-1.5*length, y_cent+1.5*length)

def customizd_plot(
    xl, 
    yl, 
    centerline, 
    projection,
    mode="-",
    paths_ids=None, 
    selected_path=None, 
    fidx=0, 
    full_pic=False
    ):

    fig = plt.figure(figsize=(6,6))
    
    plot_centerline(
        centerline=centerline, 
        selected_path=selected_path, 
        paths_ids=paths_ids, 
        mode=mode
        )

    plot_track(xl, yl, color="blue")
    
    plot_projection(
        projection=projection, 
        color="orange", 
        xl=xl, 
        yl=yl, 
        connection=True)
    
    if full_pic:
        plt.savefig(f"fig/{fidx}-full.png") 
    
    fig_cut(xl, yl, projection)
    plt.savefig(f"fig/{fidx}.png",dpi=1200)
    plt.close("all")