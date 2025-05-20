import os
import torch
import numpy as np
import open3d as o3d

import sys
root_path = os.path.dirname(__file__)
sys.path.append(root_path)

from model import PointPillars
from utils import keep_bbox_from_lidar_range, keep_bbox_from_image_range, bbox3d2corners_camera, points_camera2image, vis_img_3d, bbox3d2corners
from utils.vis_o3d import npy2ply, COLORS, bbox_obj

class LidarVehicleDetector():
    """Constructs a Lidar Vehicle Detector class using PointPillars.

    Args:
        None
    """
    def __init__(self):
        checkpoint_path = os.path.join(root_path, 'epoch_120.pth')
        CLASSES = {'Car': 0}
        self.model = PointPillars(nclasses=len(CLASSES)).cuda()
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.eval()
        self.pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0], dtype=np.float32)

        self.mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1280, height=1280)
        vis.get_render_option().point_size = 1
        #设置颜色背景为黑色
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])

        ctr = vis.get_view_control()
        ctr.set_front([0, 1, 0])
        ctr.set_up([0, 0, -1])
        ctr.set_lookat([1, 0, 0])

        self.vis = vis

    @torch.no_grad()
    def get_vehicles(self, img, pc, P2, R0_rect, Tr_velo_to_cam):
        pc_torch = torch.from_numpy(pc).float()
        pc_torch = pc_torch.cuda()
        result_filter = self.model(batched_pts=[pc_torch], mode='test')[0]
        if isinstance(result_filter, tuple):
            pc_img = self.vis_pc(pc)

            result_filter = {'lidar_bboxes': np.zeros((1, 7), dtype=np.float32), 
                            'labels': np.array([0]), 
                            'scores': np.array([0], dtype=np.float32), 
                            'bboxes2d': np.zeros((1, 4), dtype=np.float32), 
                            'camera_bboxes': np.zeros((1, 7), dtype=np.float32)}
            
            lidar_bboxes = result_filter['lidar_bboxes']
            labels, scores = result_filter['labels'], result_filter['scores']
            bboxes2d, camera_bboxes = result_filter['bboxes2d'], result_filter['camera_bboxes'] 
        else:
            image_shape = img.shape[:2]
            result_filter = keep_bbox_from_image_range(result_filter, Tr_velo_to_cam, R0_rect, P2, image_shape)
            result_filter = keep_bbox_from_lidar_range(result_filter, self.pcd_limit_range)
            lidar_bboxes = result_filter['lidar_bboxes']
            labels, scores = result_filter['labels'], result_filter['scores']
            pc_img = self.vis_pc(pc, bboxes=lidar_bboxes, labels=labels)
            bboxes2d, camera_bboxes = result_filter['bboxes2d'], result_filter['camera_bboxes'] 
            bboxes_corners = bbox3d2corners_camera(camera_bboxes)
            image_points = points_camera2image(bboxes_corners, P2)
            img = vis_img_3d(img, image_points, labels, rt=True)

        format_result = {
                'name': [],
                'truncated': [],
                'occluded': [],
                'alpha': [],
                'bbox': [],
                'dimensions': [],
                'location': [],
                'rotation_y': [],
                'score': []
            }
        
        CLASSES = {'Car': 0}
        LABEL2CLASSES = {v:k for k, v in CLASSES.items()}

        for lidar_bbox, label, score, bbox2d, camera_bbox in zip(lidar_bboxes, labels, scores, bboxes2d, camera_bboxes):
            format_result['name'].append(LABEL2CLASSES[label])
            format_result['truncated'].append(0.0)
            format_result['occluded'].append(0)
            alpha = camera_bbox[6] - np.arctan2(camera_bbox[0], camera_bbox[2])
            format_result['alpha'].append(alpha)
            format_result['bbox'].append(bbox2d)
            format_result['dimensions'].append(camera_bbox[3:6])
            format_result['location'].append(camera_bbox[:3])
            format_result['rotation_y'].append(camera_bbox[6])
            format_result['score'].append(score)

        pc_img = np.asarray(pc_img)
        pc_img = (pc_img * 255).astype(np.uint8)

        return img, pc_img, format_result

    def vis_pc(self, pc, bboxes=None, labels=None):
        def vis_core(vis, plys):
            vis.clear_geometries()
            for ply in plys:
                vis.add_geometry(ply)
            return vis.capture_screen_float_buffer(True)
        
        if isinstance(pc, np.ndarray):
            pc = npy2ply(pc)
        
        bboxes = None
        if bboxes is None:
            return vis_core(self.vis, [pc, self.mesh_frame])
        
        if len(bboxes.shape) == 2:
            bboxes = bbox3d2corners(bboxes)
        
        vis_objs = [pc, self.mesh_frame]
        for i in range(len(bboxes)):
            bbox = bboxes[i]
            if labels is None:
                color = [1, 1, 0]
            else:
                if labels[i] >= 0 and labels[i] < 3:
                    color = COLORS[labels[i]]
                else:
                    color = COLORS[-1]
            vis_objs.append(bbox_obj(bbox, color=color))
        return vis_core(self.vis, vis_objs)




