import torch
import carla
from carla import Transform, Location, Rotation, Vector3D, BoundingBox
import random
import queue
import numpy as np
import cv2
import math
import yaml
from PIL import Image

# from models.experimental import attempt_load
# from utils.general import non_max_suppression, scale_boxes
from torchvision import transforms
# from adv_patch_gen.utils.patch import PatchApplier, PatchTransformer
# from adv_patch_gen.utils.config_parser import load_config_object
from ultralytics import YOLO
from utils import read_points, read_calib, read_label, keep_bbox_from_lidar_range, vis_pc
from model import PointPillars
from loss import Loss

CLASSES = {
        'Car': 0
        }
LABEL2CLASSES = {v:k for k, v in CLASSES.items()}
pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)

def rotz(r):
    c = np.cos(r)
    s = np.sin(r)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

def center2corner_3d(x,y,z,w,l,h,r):
    x_corners = [-l/2,-l/2,l/2,l/2,-l/2,-l/2,l/2,l/2]
    y_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    #z_corners = [0,0,0,0,h,h,h,h]   #for kitti3d dataset
    z_corners = [-h/2,-h/2,-h/2,-h/2,h/2,h/2,h/2,h/2]  #for our lidar-coordination-based dataset
    R = rotz(r)
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners])) + np.vstack([x,y,z])
    # print(corners_3d.shape)
    # return np.transpose(corners_3d,(0,2,1))
    return np.transpose(corners_3d,(1,0))

def yolo_2_xml_bbox_coco(bbox, w, h):
    names = ['Car', 'Truck', 'Bus', 'Motorcycle', 'Bicycle', 'Pedestrian']
    # class x_center, y_center width heigth

    w_half_len = (float(bbox[3]) * w) / 2
    h_half_len = (float(bbox[4]) * h) / 2

    xmin = int((float(bbox[1]) * w) - w_half_len)
    ymin = int((float(bbox[2]) * h) - h_half_len)
    xmax = int((float(bbox[1]) * w) + w_half_len)
    ymax = int((float(bbox[2]) * h) + h_half_len)
    
    return [names[int(bbox[0])], xmin, ymin, xmax, ymax]

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in [13, 17, 9]]
    return tuple(color)

#构造相机投影矩阵函数
def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

# 计算三维坐标的二维投影
def get_image_point(loc, K, w2c):
    if isinstance(loc, np.ndarray):
        point = np.array([loc[0], loc[1], loc[2], 1])
    else:
        point = np.array([loc.x, loc.y, loc.z, 1])      # 格式化输入坐标（loc 是一个 carla.Position 对象）
    point_camera = np.dot(w2c, point)               # 转换到相机坐标系
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]     # 将坐标系从 UE4 的坐标系转换为标准坐标系（y, -z, x），同时移除第四个分量
    point_img = np.dot(K, point_camera)             # 使用相机矩阵进行三维到二维投影
    point_img[0] /= point_img[2]                    # 归一化
    point_img[1] /= point_img[2]

    return point_img[0:2]

def get_3d_corners(bbox):
    x, y, z, w, l, h, theta = bbox
    corners = np.array([[w/2, l/2, 0],
                        [-w/2, l/2, 0],
                        [-w/2, -l/2, 0],
                        [w/2, -l/2, 0],
                        [w/2, l/2, -h],
                        [-w/2, l/2, -h],
                        [-w/2, -l/2, -h],
                        [w/2, -l/2, -h]])
    
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                           [np.sin(theta), np.cos(theta), 0],
                           [0, 0, 1]])
    
    corners = corners @ rot_matrix.T + np.array([x, y, z])
    return np.hstack((corners, np.ones((8, 1))))

def get_transform_matrix(transform):
    """Convert CARLA transform to a 4x4 transformation matrix"""
    rotation = transform.rotation
    location = transform.location

    # Convert rotation (Roll, Pitch, Yaw) to rotation matrix
    roll, pitch, yaw = np.radians([rotation.roll, rotation.pitch, rotation.yaw])
    
    R = np.array([
        [np.cos(yaw) * np.cos(pitch), 
         np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll),
         np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)],

        [np.sin(yaw) * np.cos(pitch), 
         np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll),
         np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)],

        [-np.sin(pitch), np.cos(pitch) * np.sin(roll), np.cos(pitch) * np.cos(roll)]
    ])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [location.x, location.y, location.z]
    
    return T

def lidar_to_camera_bbox(lidar_bboxes, world, sensor, intrinsic):
    """
    Project 3D LiDAR bounding boxes onto 2D image.
    """
    projected_boxes = []
    for bbox in lidar_bboxes:
        x, y, z, l, w, h, yaw = bbox

        # Define 3D bounding box corners
        dx, dy = l / 2, w / 2
        corners = np.array([
            [dx, dy, 0], [-dx, dy, 0], [-dx, -dy, 0], [dx, -dy, 0],  # Bottom four
            [dx, dy, h], [-dx, dy, h], [-dx, -dy, h], [dx, -dy, h]   # Top four
        ])

        # Apply rotation around Z (yaw)
        rot_mat = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        corners = (rot_mat @ corners.T).T + [x, y, z]

        # Transform to camera coordinates
        corners = [world.transform_point(p, sensor) for p in corners]

        # Project to image
        img_pts = []
        for X, Y, Z in corners:
            if Z > 0:  # Ensure in front of camera
                u, v = intrinsic @ [X / Z, Y / Z, 1]
                img_pts.append((int(u), int(v)))

        if len(img_pts) == 8:
            projected_boxes.append(img_pts)

    return projected_boxes

def draw_3d_bbox(image, bboxes):
    """
    Draw 3D bounding boxes on the image.
    """
    for box in bboxes:
        # Connect corners to form a cube
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top
            (0, 4), (1, 5), (2, 6), (3, 7)   # Sides
        ]
        for i, j in edges:
            cv2.line(image, box[i], box[j], (0, 255, 0), 2)

    return image

def get_sensor_transform(sensor):
    """
    Returns the transformation matrix for a CARLA sensor.
    """
    sensor_transform = sensor.get_transform()  # Get CARLA transform object
    sensor_matrix = sensor_transform.get_matrix()  # 4x4 transformation matrix
    return np.array(sensor_matrix)

def get_world_to_camera_transform(camera):
    """
    Get the transformation matrix that converts world coordinates to camera coordinates.
    """
    camera_transform = camera.get_transform()
    world_to_camera = np.linalg.inv(np.array(camera_transform.get_matrix()))  # Inverse of world to camera
    return world_to_camera

def get_camera_intrinsic(camera):
    """
    Returns the camera intrinsic matrix for projection.
    """
    image_w = camera.attributes['image_size_x']
    image_h = camera.attributes['image_size_y']
    fov = float(camera.attributes['fov'])

    focal = image_w / (2.0 * np.tan(np.radians(fov) / 2.0))  # Focal length calculation
    K = np.array([
        [focal, 0, image_w / 2],
        [0, focal, image_h / 2],
        [0, 0, 1]
    ])
    return K

def get_carla_transforms(world, camera, lidar):
    """
    Get all necessary transformation matrices for projecting LiDAR points onto a camera image.
    """
    # Get world to camera transform
    world_to_camera = get_world_to_camera_transform(camera)

    # Get lidar to world transform
    lidar_transform = lidar.get_transform()
    lidar_to_world = np.array(lidar_transform.get_matrix())

    # Get camera intrinsics
    intrinsic = get_camera_intrinsic(camera)

    return world_to_camera, lidar_to_world, intrinsic


class CarlaPatchAttacker():
    
    def __init__(self):
        self.client = None
        self.world = None
        self.bp_lib = None
        self.model = None
        self.adv_patch = None
        self.patch_transformer = None
        self.patch_applier = None
        self.camera = None
        self.vehicle = None
        self.K = None
        self.world_2_camera = None
        self.actor_list = []
        self.env_actor_list = []
        self.image_queue = queue.Queue()
        self.pcd_queue = queue.Queue()
        self.edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]
        

    # 初始化
    def init_attacker(self, client_port=3000):
        self.client = carla.Client('localhost', client_port)       #连接Carla并获取世界
        self.world  = self.client.get_world()
        self.bp_lib = self.world.get_blueprint_library()

        settings = self.world.get_settings()     # 设置仿真模式为同步模式
        settings.synchronous_mode = True    # 启用同步模式
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

    # 加载模型，可就具体模型进行调整
    def load_model(self, weights_path = './yolov5_carla.pt'):
        # weights = weights_path 
        # w = str(weights[0] if isinstance(weights, list) else weights)
        # self.model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, device='cuda')
        # self.model.eval()
        # self.model = YOLO(weights_path)
        self.model = PointPillars(nclasses=len(CLASSES)).cuda()
        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()

    # 读取patch
    # def load_patch(self, patch_path='adv_patch_yolov5.png', patch_cfg_path='adv_patch_yolov5.json'):
    #     patch_cfg = load_config_object(patch_cfg_path)
    #     patch_img = Image.open(patch_path)
    #     patch_img = transforms.Resize(24)(patch_img)
    #     adv_patch_cpu = transforms.ToTensor()(patch_img)
    #     self.adv_patch = adv_patch_cpu.to('cuda')
    #     self.patch_transformer = PatchTransformer(
    #             patch_cfg.target_size_frac, patch_cfg.mul_gau_mean, patch_cfg.mul_gau_std, patch_cfg.x_off_loc, patch_cfg.y_off_loc).to('cuda')
    #     self.patch_applier = PatchApplier(patch_cfg.patch_alpha).to('cuda')
    
    # 生成车辆
    def generate_vihecle(self, tm_port=9000, vehicle_type='vehicle.lincoln.mkz_2020'):
        vehicle_bp = self.bp_lib.find(vehicle_type)
        spawn_points = random.choice(self.world.get_map().get_spawn_points())
        print(spawn_points)
        spawn_point = Transform(Location(x = 27.142294, y = 66.283257, z = 0.600000), Rotation(pitch=0.000000, yaw=-179.926727, roll=0.000000))
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_points)
        self.vehicle.set_autopilot(True, tm_port)
    
    # 生成相机
    def generate_camera(self, image_size=(640, 640)):
        camera_bp = self.bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x',str(image_size[0]))
        camera_bp.set_attribute('image_size_y',str(image_size[1]))
        # camera_init_trans = carla.Transform(carla.Location(z=2))
        camera_init_trans = carla.Transform(carla.Location(x=1.6, z=1.7))
        self.camera = self.world.spawn_actor(camera_bp, camera_init_trans, attach_to=self.vehicle)
        self.camera.listen(self.image_queue.put)      # 创建对接接收相机数据

        image_w = image_size[0]     # 图像宽度
        image_h = image_size[1]     # 图像高度
        fov = camera_bp.get_attribute("fov").as_float()     # 视场角

        self.K = build_projection_matrix(image_w, image_h, fov)      # 计算相机投影矩阵，用于从三维坐标投影到二维坐标

        # Compute focal length
        # image_width = int(image_w)
        # image_height = int(image_h)
        # f_x = image_width / (2.0 * np.tan(np.radians(fov) / 2.0))
        # f_y = f_x  # Assuming square pixels

        # # Compute camera intrinsic matrix K
        # K = np.array([
        #     [f_x, 0, image_width / 2],
        #     [0, f_y, image_height / 2],
        #     [0, 0, 1]
        # ])
    
    def generate_ladir(self, delta=0.05):
        """
        To get lidar bp
        :param blueprint_library: the world blueprint_library
        :param delta: update rate(s)
        :return: lidar bp
        """
        lidar_bp = self.bp_lib.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("dropoff_general_rate", "0.0")
        lidar_bp.set_attribute("dropoff_intensity_limit", "1.0")
        lidar_bp.set_attribute("dropoff_zero_intensity", "0.0")

        lidar_bp.set_attribute("upper_fov", str(15.0))
        lidar_bp.set_attribute("lower_fov", str(-25.0))
        lidar_bp.set_attribute("channels", str(64.0))
        lidar_bp.set_attribute("range", str(100.0))
        lidar_bp.set_attribute("rotation_frequency", str(1.0 / delta))
        lidar_bp.set_attribute("points_per_second", str(500000))

        # lidar_bp = self.bp_lib.find("sensor.lidar.ray_cast")
        # lidar_bp.set_attribute("dropoff_general_rate", "0.10")
        # lidar_bp.set_attribute("dropoff_intensity_limit", "0.8")
        # lidar_bp.set_attribute("dropoff_zero_intensity", "0.4")

        # lidar_bp.set_attribute("upper_fov", str(7.0))
        # lidar_bp.set_attribute("lower_fov", str(-16.0))
        # lidar_bp.set_attribute("channels", str(64.0))
        # lidar_bp.set_attribute("range", str(100.0))
        # lidar_bp.set_attribute("rotation_frequency", str(10))
        # lidar_bp.set_attribute("points_per_second", str(1300000))

        # lidar_transform = carla.Transform(carla.Location(x = -0.5, z = 1.8))
        lidar_transform = carla.Transform(carla.Location(x = 1.6, z = 1.7))
        self.lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to = self.vehicle)
        self.lidar.listen(self.pcd_queue.put)
    
    def lidar_callback(point_cloud, point_list):
        # We need to convert point cloud(carla-format) into numpy.ndarray
        data = np.copy(np.frombuffer(point_cloud.raw_data, dtype = np.dtype("f4")))
        data = np.reshape(data, (int(data.shape[0] / 4), 4))

        intensity = data[:, -1]
        intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
        # int_color = np.c_[
        #     np.interp(intensity_col, VID_RANGE, VIDIDIS[:, 0]),
        #     np.interp(intensity_col, VID_RANGE, VIDIDIS[:, 1]),
        #     np.interp(intensity_col, VID_RANGE, VIDIDIS[:, 2])]

        # points = data[:, :-1] # we only use x, y, z coordinates
        # points[:, 1] = -points[:, 1] # This is different from official script
        # point_list.points = o3d.utility.Vector3dVector(points)
        # point_list.colors = o3d.utility.Vector3dVector(int_color)
    
    #生成检测目标，包括活动的车辆目标和地图上的装饰车辆
    def generate_objects(self, tm_port=9000, num_objects=20):
        for i in range(num_objects):
            vehicle_bp = random.choice(self.bp_lib.filter('vehicle'))       # 生成活动的车辆目标
            npc = self.world.try_spawn_actor(vehicle_bp, random.choice(self.world.get_map().get_spawn_points()))
            if npc:
                npc.set_autopilot(True, tm_port)
                self.actor_list.append(npc)

        bike_objects = self.world.get_environment_objects(carla.CityObjectLabel.Bicycle)        #获取地图上的装饰车辆信息
        moto_objects = self.world.get_environment_objects(carla.CityObjectLabel.Motorcycle)
        truck_objects = self.world.get_environment_objects(carla.CityObjectLabel.Truck)
        car_objects = self.world.get_environment_objects(carla.CityObjectLabel.Car)
        bus_objects = self.world.get_environment_objects(carla.CityObjectLabel.Bus)

        self.env_actor_list.extend(bike_objects)
        self.env_actor_list.extend(moto_objects)
        self.env_actor_list.extend(truck_objects)
        self.env_actor_list.extend(car_objects)
        self.env_actor_list.extend(bus_objects)

    def render(self, image_shape=(640, 640), trigger=None, apply_patch_transforms=False):
        self.world.tick()       # 获取一张图像
        image = self.image_queue.get()
        CLS_LIST =  ['Car', 'Truck', 'Bus', 'Motorcycle', 'Bicycle', 'Pedestrian'] 
        img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))       # 将原始数据重新整形为 RGB 数组

        cv2.namedWindow('ImageWindowName', cv2.WINDOW_AUTOSIZE)     # 在 OpenCV 的显示窗口中显示图像
        cv2.imshow('ImageWindowName', img)
        cv2.waitKey(1)

        while True:
            # 更新世界状态并获取图像
            self.world.tick()
            image = self.image_queue.get()
            pcd = self.pcd_queue.get()
            lidar_bboxes_list = []
            pcd = np.copy(np.frombuffer(pcd.raw_data, dtype = np.dtype("f4")))
            pcd = np.reshape(pcd, (int(pcd.shape[0] / 4), 4))
            # print(pcd.shape)
            pcd_torch = torch.from_numpy(pcd).float()
            with torch.no_grad():
                pcd_torch = pcd_torch.cuda()
                
                result_filter = self.model(batched_pts=[pcd_torch], mode='test')[0]
            if isinstance(result_filter, tuple):
                continue
            result_filter = keep_bbox_from_lidar_range(result_filter, pcd_limit_range)
            lidar_bboxes = result_filter['lidar_bboxes']
            labels_pcd, scores = result_filter['labels'], result_filter['scores']
            # vis_pc(pcd, bboxes=lidar_bboxes, labels=labels_pcd)
            # print(lidar_bboxes)
            print()
            # for lidar_bbox in lidar_bboxes:
            #     lidar_bbox = [float(x) for x in list(lidar_bbox)]
            #     print(lidar_bbox)
            #     # print(lidar_bbox[0])
            #     # loc = Location(x=lidar_bbox[0], y=lidar_bbox[1], z=lidar_bbox[2])
            #     # ext = Vector3D(x=lidar_bbox[3]/2, y=lidar_bbox[4]/2, z=lidar_bbox[5]/2)
            #     # rot = Rotation(pitch=float(0), yaw=lidar_bbox[6], roll=float(0))
            #     # lidar_bboxes_list.append(BoundingBox(location=loc, extent=ext, rotation=rot))
            #     # lidar_bboxes_list.append(BoundingBox(location=loc, extent=ext))
            #     # lidar_bbox[6] = lidar_bbox[6] - math.pi / 2
            #     lidar_bboxes_list.append(center2corner_3d(lidar_bbox[0], -lidar_bbox[1], lidar_bbox[2]+1.7, lidar_bbox[3], lidar_bbox[4], lidar_bbox[5], lidar_bbox[6]))
            # # print(lidar_bboxes_list)
            array = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

            print('----------')
            img0 = array[:, :, :3]
            img0 = cv2.resize(img0, None, fx=1, fy=1)
            height, width = image_shape[0], image_shape[1]
            targets = []
            labels = []

            self.world_2_camera = np.array(self.camera.get_transform().get_inverse_matrix())        # 获取相机投影矩阵

            # T_lidar_world = get_transform_matrix(self.lidar.get_transform())
            # T_camera_world = get_transform_matrix(self.camera.get_transform())

            # T_cam_to_world_inv = np.linalg.inv(T_camera_world)
            # T_lidar_to_cam = T_cam_to_world_inv @ T_lidar_world

            # for lidar_verts in lidar_bboxes_list:

            #     for edge in self.edges:        #画ground_truth3D框
            #         p1 = get_image_point(lidar_verts[edge[0]], self.K, self.world_2_camera)
            #         p2 = get_image_point(lidar_verts[edge[1]], self.K, self.world_2_camera)
            #         img0 = cv2.line(img0, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 0, 0), 1)

                # points = np.array([get_image_point(vert, self.K, self.world_2_camera) for vert in lidar_verts])   # 获取目标3dbbox角点
                # max_point = np.max(points, axis=0)      # 计算2dbbox左上角和右下角点
                # min_point = np.min(points, axis=0)
                # x, y = min_point[0], min_point[1]
                # w, h = max_point[0]- min_point[0], max_point[1]- min_point[1]
                # if w > 480 or h> 480:       #滤除异常box
                #     continue

                # img0 = cv2.rectangle(img0, (int(min_point[0]), int(min_point[1])), (int(max_point[0]), int(max_point[1])), (0, 255, 0), 2)      # 画ground_truth2D框
            
            # world_to_camera, lidar_to_world, intrinsic = get_carla_transforms(self.world, self.camera, self.lidar)
            projected = lidar_to_camera_bbox(lidar_bboxes, self.world, self.lidar, self.K)
            img0 = draw_3d_bbox(img0, projected)

            # for lidar_bbox in lidar_bboxes:
            #     bbox = [float(x) for x in list(lidar_bbox)]
            #     corners_lidar = get_3d_corners(bbox)
            #     corners_camera = (T_lidar_to_cam @ corners_lidar.T).T[:, :3]

            #     # Project to 2D image
            #     corners_2d = (self.K @ corners_camera.T).T
            #     corners_2d = (corners_2d[:, :2].T / corners_2d[:, 2]).T  # Normalize by depth

            #     # Draw projected 2D bounding box
            #     for i in range(4):
            #         pt1, pt2 = tuple(corners_2d[i].astype(int)), tuple(corners_2d[(i+1) % 4].astype(int))
            #         pt3, pt4 = tuple(corners_2d[i+4].astype(int)), tuple(corners_2d[((i+1) % 4) + 4].astype(int))
            #         print(pt1)
            #         cv2.line(img0, pt1, pt2, (0, 255, 0), 2)
            #         cv2.line(img0, pt3, pt4, (0, 255, 0), 2)
            #         cv2.line(img0, pt1, pt3, (0, 255, 0), 2)

            # 获取目标bbox的ground truth并可视化
            object_list = self.actor_list + self.env_actor_list
            for npc in object_list:
                bb = npc.bounding_box
                if npc in self.actor_list:      # 区分活动车辆目标与环境装饰车辆
                    dist = npc.get_transform().location.distance(self.vehicle.get_transform().location)
                    ray = npc.get_transform().location - self.vehicle.get_transform().location
                    verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                else:
                    dist = bb.location.distance(self.vehicle.get_transform().location)
                    ray = bb.location - self.vehicle.get_transform().location
                    verts = [v for v in bb.get_local_vertices()]

                if dist < 65:       # 筛选距离在65米以内的车辆
                    forward_vec = self.vehicle.get_transform().get_forward_vector()

                    if forward_vec.dot(ray) > 1:        # 计算车辆前进方向与车辆之间的向量的点积, 通过阈值判断是否在相机前方绘制边界框
                        # print(verts[0])
                        p1 = get_image_point(bb.location, self.K, self.world_2_camera)
                        points = np.array([get_image_point(vert, self.K, self.world_2_camera) for vert in verts])   # 获取目标3dbbox角点
                        max_point = np.max(points, axis=0)      # 计算2dbbox左上角和右下角点
                        min_point = np.min(points, axis=0)
                        x, y = min_point[0], min_point[1]
                        w, h = max_point[0]- min_point[0], max_point[1]- min_point[1]
                        if w > 480 or h> 480:       #滤除异常box
                            continue

                        img0 = cv2.rectangle(img0, (int(min_point[0]), int(min_point[1])), (int(max_point[0]), int(max_point[1])), (0, 0, 255), 2)      # 画ground_truth2D框
                        xc, yc = x + (w / 2), y + (h / 2)
                        # targets.append([0, 0, xc/width, yc/height, w/width, h/height])        # 6维label，图片索引，cls，xc，yc，w，h
                        labels.append([0, xc/width, yc/height, w/width, h/height])      # 修改2dbbox并储存，5维label，cls，xc，yc，w，h

                        # for edge in edges:        #画ground_truth3D框
                        #     p1 = get_image_point(verts[self.edge[0]], self.K, self.world_2_camera)
                        #     p2 = get_image_point(verts[self.edge[1]], self.K, self.world_2_camera)
                        #     cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 0, 0, 255), 1)
            
            # 预处理当前帧图像与获取的label
            img = cv2.resize(img0, (height, width))    #尺寸变换
            
                        
            label = np.asarray(labels) if labels else np.zeros([1, 5])      # 处理label格式
            label = torch.from_numpy(label).float()
            if label.dim() == 1:
                label = label.unsqueeze(0)

            bbox_coordinates = [yolo_2_xml_bbox_coco(lab, width, height) for lab in label]
            num_attack = len(bbox_coordinates)
            #变换patch并增加至原图
            poisoned_image = img
            # poisoned_image, triggers_put = putTriggerInBbox(img, trigger, fusion_ratio=1, 
            #                                                 GT_bbox_coordinates=bbox_coordinates, num_attack=num_attack, seed=3407)
            # img = poisoned_image / 255.
            # img = img[:, :, ::-1].transpose((2, 0, 1))   #HWC转CHW
            # img = np.expand_dims(img, axis=0)    #扩展维度至[1,3,640,640]
            # img = torch.from_numpy(img.copy())   #numpy转tensor
            # img = img.to(torch.float32).to('cuda')
            # p_tensor_batch = img
            # cv2.imwrite('./poisoned_img', poisoned_image)

            #使用模型进行推理，更换模型可进行针对性修改
            # pred = self.model(p_tensor_batch)       
            # pred = pred[0]
            # pred = pred.clone().cpu().detach()
            # pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)     # 输出6维，xc，yc，w，h，score，cls
            # # print(pred)
            # pred = self.model(poisoned_image, imgsz=640, verbose=False, device=0)

            # img0 = p_tensor_batch[0].cpu().numpy()
            # img0 = img0.transpose((1, 2, 0))[:, :, ::-1]
            # img0 = (img0 * 255).astype(np.uint8)
            # img0 = img0.copy()
            img0 = poisoned_image.copy()

            # print(pred)
            #模型检测结果可视化
            # for i in range(len(pred[0].boxes.cls)):
            #     if len(pred[0].boxes.cls):
            #         cls_id = int(pred[0].boxes.cls[i].item())
            #         cls_color = compute_color_for_labels(cls_id)
            #         cls_name = CLS_LIST[cls_id]
            #         label = str(int(pred[0].boxes.cls[i].item()))
            #         box_xyxy = pred[0].boxes.xyxy[i]
            #         # cv2.rectangle(img, (int(box_xyxy[0]), int(box_xyxy[1])), (int(box_xyxy[2]), int(box_xyxy[3])), color=(255, 0, 0) )
            #         cv2.rectangle(img0, (int(box_xyxy[0]), int(box_xyxy[1])), (int(box_xyxy[2]), int(box_xyxy[3])), color=cls_color )
            #         cv2.putText(img0, cls_name, (int(box_xyxy[0] + 2), int(box_xyxy[1] + 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, cls_color, 2)
            # for i, det in enumerate(pred):
            #     if len(det):
            #         det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            #         for *xyxy, conf, cls in reversed(det):
            #             # print('{},{},{}'.format(xyxy, conf.numpy(), cls.numpy())) #输出结果：xyxy检测框左上角和右下角坐标，conf置信度，cls分类结果
            #             img0 = cv2.rectangle(img0, (int(xyxy[0].numpy()), int(xyxy[1].numpy())), (int(xyxy[2].numpy()), int(xyxy[3].numpy())), (0, 255, 0), 2)
            #             cls=0
            #             cls_color = compute_color_for_labels(cls)
            #             # print(xyxy)
            #             cv2.putText(img0, CLS_LIST[int(cls)], (int(xyxy[0] + 2), int(xyxy[1] + 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, cls_color, 2)
            #             # cv2.putText(img0, 'car', (int(xyxy[0].numpy()), int(xyxy[1].numpy() - 5)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)

            cv2.imshow('ImageWindowName', img0)
            if cv2.waitKey(1) == ord('q'):      #按q退出
                break

        cv2.destroyAllWindows()



if __name__ == '__main__':
    try:
        weights_path = './pillar_logs/checkpoints/epoch_120.pth'
        # patch_path = 'adv_patch_yolov5.png'
        # patch_config_path = 'adv_patch_yolov5.json'
        image_size = (640, 640)
        tri_size = (20, 20)
        # TRIGGER = cv2.resize(cv2.imread("./trigger_hidden.png"), tri_size)

        carla_patch_attacker = CarlaPatchAttacker()     # 实例化
        carla_patch_attacker.init_attacker(client_port=3000)        # 初始化
        carla_patch_attacker.load_model(weights_path=weights_path)      # 加载模型
        # carla_patch_attacker.load_patch(patch_path=patch_path, patch_cfg_path=patch_config_path)        # 加载patch
        carla_patch_attacker.generate_vihecle(tm_port=9000, vehicle_type='vehicle.tesla.model3')     #生成本车
        carla_patch_attacker.generate_camera(image_size=image_size)     #生成相机
        carla_patch_attacker.generate_ladir(delta=0.05)
        carla_patch_attacker.generate_objects(tm_port=9000, num_objects=20)     #生成检测对象

        carla_patch_attacker.render(image_shape=image_size, apply_patch_transforms=False)


    finally:
        for actor in carla_patch_attacker.actor_list:       # 销毁object
            actor.destroy()
        carla_patch_attacker.camera.destroy()
        carla_patch_attacker.vehicle.destroy()
        print("All cleaned up!")
        

    



 
