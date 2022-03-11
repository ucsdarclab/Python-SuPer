import numpy as np
import cv2
import open3d as o3d

from utils.config import *

def visualize_surfel(allsurfels, vis, path):

    if open3d_visualize:
    
        vis.remove_geometry(allsurfels.pcd)
        vis.add_geometry(allsurfels.pcd) # TODO update_geometry?
        # vis.add_geometry(allsurfels.ED_pcd)

        vis.poll_events()
        vis.update_renderer()

        # o3d.visualization.RenderOption(point_size=0.1) # TODO display with smaller point size

        # Save the current open3d display as image
        # vis.capture_screen_image(path)
        # If vis.capture_screen_image() saves blake images,
        # comment the previous line and use the next two lines
        img = np.asarray(vis.capture_screen_float_buffer(True))
        img = img[338:677,616:1232]
        cv2.imwrite(path, img[:,:,::-1]*255)

    else:

        renderImg,_ = allsurfels.projSurfel(allsurfels.points, allsurfels.colors)
        cv2.imwrite(path, renderImg)

        # cv2.imshow(vis,renderImg)