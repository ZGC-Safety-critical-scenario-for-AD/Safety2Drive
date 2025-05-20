import carla
from carla import Transform, Location, Rotation
import queue
import cv2
import numpy as np
import time
import argparse
from util import calc_asr, get_image_point, build_projection_matrix, eval_coco_metrics, scatter_trigger, yolo_2_xml_bbox_coco, point_range_filter
# from util import get_image_point, build_projection_matrix, eval_coco_metrics, scatter_trigger, yolo_2_xml_bbox_coco, point_range_filter
import random
import copy
import os.path as osp
import json
import torch
import math
import sys
sys.path.append('/data1/wtc/lidar_vehicle_detection/lidar_vehicle_detection')

from evaluate_car import do_eval
from client_bounding_boxes import ClientSideBoundingBoxes

# from weather_test.Camera_corruptions import *
from Camera_corruptions import *
from LiDAR_corruptions import *

def get_intrinsic_matrix(camera):

    width = int(camera.attributes['image_size_x'])
    height = int(camera.attributes['image_size_y'])
    fov = float(camera.attributes['fov'])

    k = np.identity(3)
    k[0, 2] = width / 2.0
    k[1, 2] = height / 2.0
    k[0, 0] = k[1, 1] = width / (2.0 * np.tan(fov * np.pi / 360.0))

    return k

def sensor_to_image(cords_x_y_z, camera_calibration):
    cords_x_y_z = np.concatenate([cords_x_y_z[1][0], -cords_x_y_z[2][0], cords_x_y_z[0][0]])
    # 使用相机矩阵进行三维到二维投影
    point_img = np.dot(camera_calibration, cords_x_y_z)

    # 归一化
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]
    point_img[2] /= point_img[2]

    return point_img[0:3]

def get_bounding_box_and_refpoint(agent, camera, camera_calibration, list_type='actor'):
    """
    An extended version of Carla get_bounding_box() method, where the reference point of the bbox is also
    concatenated with the bbox vertices to boost the performance as all vertices and refpoint are processed in parallel.
    Returns 3D bounding box and its reference point for a agent based on camera view.
    """
    bb_cords_1 = agent.bounding_box.get_world_vertices(Transform())

    location = agent.bounding_box.location
    vert = np.block([[v.x, v.y, v.z, 1] for v in bb_cords_1])
    bbox_refpoint = np.array([[0, 0, 0, 1]], dtype=float)
    bb_cords = ClientSideBoundingBoxes._create_bb_points(agent)
    bb_cords_and_refpoint = np.vstack((bb_cords, bbox_refpoint))
    bb_cords_and_refpoint_1 = np.vstack((vert, bbox_refpoint))
    bb_cords_and_refpoint_2 = np.array([
        [location.x],  # [[X,
        [location.y],  # Y,
        [location.z],  # Z,
        [1.0]  # 1.0]]
    ])
    if list_type == 'actor':
        cords_x_y_z = ClientSideBoundingBoxes._vehicle_to_sensor(bb_cords_and_refpoint, agent, camera)[:3, :]
        cords_x_y_z_1 = ClientSideBoundingBoxes._world_to_sensor(bb_cords_and_refpoint_2, camera)[:3, :]

    else:
        cords_x_y_z = ClientSideBoundingBoxes._world_to_sensor(bb_cords_and_refpoint_1.T, camera)[:3, :]
        cords_x_y_z_1 = ClientSideBoundingBoxes._world_to_sensor(bb_cords_and_refpoint_2, camera)[:3, :]

    cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
    bbox_and_refpoint = np.transpose(np.dot(camera_calibration, cords_y_minus_z_x))
    camera_bbox_refpoint = np.concatenate([bbox_and_refpoint[:, 0] / bbox_and_refpoint[:, 2], bbox_and_refpoint[:, 1] / bbox_and_refpoint[:, 2], bbox_and_refpoint[:, 2]], axis=1)

    sensor_bbox_refpoint = np.transpose(cords_x_y_z)

    camera_bbox = camera_bbox_refpoint[:-1, :]
    camera_refpoint = np.squeeze(np.asarray(camera_bbox_refpoint[-1, :]))
    sensor_bbox = sensor_bbox_refpoint[:-1, :]
    sensor_refpoint = np.squeeze(np.asarray(sensor_bbox_refpoint[-1, :]))

    if list_type != 'actor':
        camera_refpoint = sensor_to_image(cords_x_y_z_1, camera_calibration)
        sensor_refpoint = np.asarray(cords_x_y_z_1).reshape(-1)
        # sensor_refpoint = sensor_cords

    return (camera_bbox, camera_refpoint), (sensor_bbox, sensor_refpoint)

def get_relative_rotation_y(agent, player_transform, list_type='actor'):
    """ Returns the relative rotation of the agent to the camera in yaw
    The relative rotation is the difference between the camera rotation (on car) and the agent rotation"""
    if list_type == 'actor':
        rot_agent = agent.get_transform().rotation.yaw
    else:
        rot_agent = agent.transform.rotation.yaw
    rot_vehicle = player_transform.rotation.yaw
    #rotate by -90 to match kitti
    # rel_angle = math.radians(rot_agent - rot_vehicle - 90)
    rel_angle = math.radians(rot_agent - rot_vehicle - 90)
    # rel_angle = math.radians(rot_agent - rot_vehicle)
    if rel_angle > math.pi:
        rel_angle = rel_angle - 2 * math.pi
    elif rel_angle < - math.pi:
        rel_angle = rel_angle + 2 * math.pi
    return rel_angle


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--port', default=2000, type=int)
    argparser.add_argument('--tm_port', default=9000, type=int)
    argparser.add_argument('--rolename', default='hero')
    argparser.add_argument('--width', default=1280, type=int)
    argparser.add_argument('--height', default=1280, type=int)
    argparser.add_argument('--task', default='lidar vehicle detection', type=str, 
                           choices=['none', 'lane detection', 'vehicle detection', 'object detection', 'object detection yolov7', 'monocular depth estimation', 'lidar vehicle detection'])
    argparser.add_argument('--attack_type', default='none', type=str, 
                           choices=['none', 'backdoor_oda', 'backdoor_oga', 'digital', 'physical'])
    argparser.add_argument('--save_video', default=True, type=bool)
    argparser.add_argument('--scenario', default=False, type=bool)
    argparser.add_argument('--output_dir', default='output', type=str)
    argparser.add_argument('--result_path', default='eval_result', type=str)
    argparser.add_argument('--exp_name', default='CO', type=str)
    argparser.add_argument('--adv', default='fgsm', type=str)
    argparser.add_argument('--eps', default=0.02, type=float)
    argparser.add_argument('--corruptions', default='', type=str)
    argparser.add_argument('--corruptions_lidar', default='', type=str)
    # argparser.add_argument('--corruptions_lidar', default='snow', type=str)
    # argparser.add_argument('--corruptions_lidar', default='rain', type=str)
    # argparser.add_argument('--corruptions_lidar', default='fog', type=str)
    # argparser.add_argument('--corruptions_lidar', default='gaussian', type=str)
    # argparser.add_argument('--corruptions_lidar', default='impulse', type=str)
    # argparser.add_argument('--corruptions_lidar', default='uniform', type=str)
    # argparser.add_argument('--corruptions_lidar', default='crosstalk', type=str)
    # argparser.add_argument('--corruptions_lidar', default='density', type=str)
    # argparser.add_argument('--corruptions_lidar', default='cutout', type=str)
    # argparser.add_argument('--corruptions_lidar', default='gaussian_box', type=str)
    # argparser.add_argument('--corruptions_lidar', default='impulse_box', type=str)
    # argparser.add_argument('--corruptions_lidar', default='uniform_box', type=str)
    # argparser.add_argument('--corruptions_lidar', default='density_box', type=str)
    # argparser.add_argument('--corruptions_lidar', default='cutout_box', type=str)
    
    args = argparser.parse_args()

    width, height = args.width, args.height
    severity, seed = 3, 1

    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if args.task in ['vehicle detection', 'object detection', 'monocular depth estimation', 'object detection yolov7'] and args.attack_type == 'none':
            vw = cv2.VideoWriter(f'{args.task}.mp4', fourcc, 60, (width * 2, height))
        elif args.task in ['lidar vehicle detection']:
            if args.corruptions=='' and args.corruptions_lidar=='':
                cor_dir = 'origin'
            elif args.corruptions=='' and args.corruptions_lidar!='':
                cor_dir = 'lidar_'+args.corruptions_lidar
            else:
                cor_dir = args.corruptions
            if not os.path.exists(f'./videos/{cor_dir}/'):
                os.makedirs(f'./videos/{cor_dir}/')
            vw = cv2.VideoWriter(f'./videos/{cor_dir}/{args.task}_{args.exp_name}.mp4', fourcc, 20, (width * 3, height))
        else:
            vw = cv2.VideoWriter(f'{args.task}.mp4', fourcc, 60, (width, height))

    if args.task == 'lane detection':
        from lane_detection.lane import LaneDetector
        detector = LaneDetector()
        detector.cfg.cut_height = 320
    elif args.task == 'vehicle detection': # 只能正方形
        from vehicle_detection.vehicle import VehicleDetector
        detector = VehicleDetector()
    elif args.task == 'object detection':
        from object_detection.object import ObjectDetector
        detector = ObjectDetector(version='v8', model_name='yolov8_carla.pt')
    elif args.task == 'object detection yolov7':
        from object_detection_yolov7.object import ObjectDetector
        detector = ObjectDetector()
    elif args.task == 'monocular depth estimation': # 最后展示有点问题
        from monocular_depth_estimation.depth import MonocularDepthEstimator
        estimator = MonocularDepthEstimator()
    elif args.task == 'lidar vehicle detection':
        if args.adv == '':
            from lidar_vehicle_detection.lidar_vehicle import LidarVehicleDetector
            detector = LidarVehicleDetector()
        elif args.adv == 'fgsm':
            from lidar_vehicle_detection.lidar_vehicle_fgsm import LidarVehicleDetector
            detector = LidarVehicleDetector()
        elif args.adv == 'pgd':
            from lidar_vehicle_detection.lidar_vehicle_pgd import LidarVehicleDetector
            detector = LidarVehicleDetector()
        elif args.adv == 'cw':
            from lidar_vehicle_detection.lidar_vehicle_cw import LidarVehicleDetector
            detector = LidarVehicleDetector()
    
    if args.corruptions == '':
        corruption = None
    elif args.corruptions == 'snow':
        corruption = ImageAddSnow(severity, seed)
    elif args.corruptions == 'rain':
        corruption = ImageAddRain(severity, seed)
    elif args.corruptions == 'fog':
        corruption = ImageAddFog(severity, seed)
    elif args.corruptions == 'gaussian':
        corruption = ImageAddGaussianNoise(severity, seed)
    elif args.corruptions == 'impulse':
        corruption = ImageAddImpulseNoise(severity, seed)
    elif args.corruptions == 'uniform':
        corruption = ImageAddUniformNoise(severity)
    elif args.corruptions == 'sun':
        corruption = ImagePointAddSun(severity)
    elif args.corruptions == 'motion':
        corruption = ImageMotionBlurFrontBack(severity)
        
    # elif args.corruptions == 'rain':
    #     corruption = ImageAddRain(severity, seed)
    # elif args.corruptions == 'fog':
    #     corruption = ImageAddFog(severity, seed)

    try:
        client = carla.Client('localhost', args.port)
        # client = carla.Client('58.206.202.156', args.port)
        
        # world = client.get_world()
        world = client.load_world('Town01')
        bp_lib = world.get_blueprint_library()

        bike_objects = world.get_environment_objects(carla.CityObjectLabel.Bicycle)
        moto_objects = world.get_environment_objects(carla.CityObjectLabel.Motorcycle)
        truck_objects = world.get_environment_objects(carla.CityObjectLabel.Truck)
        car_objects = world.get_environment_objects(carla.CityObjectLabel.Car)
        bus_objects = world.get_environment_objects(carla.CityObjectLabel.Bus)

        env_obj_list = []
        env_obj_list.extend(bike_objects)
        env_obj_list.extend(moto_objects)
        env_obj_list.extend(truck_objects)
        env_obj_list.extend(car_objects)
        env_obj_list.extend(bus_objects)

        gt_results = {}
        det_results = {}

        if args.scenario:
            actor_list = []
            ego_vehicle = None
            while ego_vehicle is None:
                print('Waiting for the ego vehicle...')
                time.sleep(1)
                possible_vehicles = world.get_actors().filter('vehicle.*')
                for vehicle in possible_vehicles:
                    vehicle.set_autopilot(True, args.tm_port)
                    actor_list.append(vehicle)
                    if vehicle.attributes['role_name'] == args.rolename:
                        print('Ego vehicle found')
                        ego_vehicle = vehicle
            env_obj_ids = [obj.id for obj in env_obj_list]
            world.enable_environment_objects(env_obj_ids, False) 

            truck_objects = world.get_environment_objects(carla.CityObjectLabel.Truck)
            car_objects = world.get_environment_objects(carla.CityObjectLabel.Car)
            bus_objects = world.get_environment_objects(carla.CityObjectLabel.Bus)

            env_obj_list = []
            env_obj_list.extend(truck_objects)
            env_obj_list.extend(car_objects)
            env_obj_list.extend(bus_objects)

            print(env_obj_list)

        else:
            ego_vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2017')
            ego_vehicle_trans = random.choice(world.get_map().get_spawn_points())
            # ego_vehicle_trans = world.get_map().get_spawn_points()[0]
            ego_vehicle = world.spawn_actor(ego_vehicle_bp, ego_vehicle_trans)
            ego_vehicle.set_autopilot(True, args.tm_port)

            actor_list = [ego_vehicle]
            for _ in range(20):
                vehicle_bp = random.choice(bp_lib.filter('vehicle.tesla.model3'))
                npc = world.try_spawn_actor(vehicle_bp, random.choice(world.get_map().get_spawn_points()))
                if npc:
                    npc.set_autopilot(True, args.tm_port)
                    actor_list.append(npc)
        
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(width))
        camera_bp.set_attribute('image_size_y', str(height))
        camera_trans = carla.Transform(carla.Location(x=1.6, z=1.7))
        camera = world.spawn_actor(camera_bp, camera_trans, attach_to=ego_vehicle)

        if 'lidar' in args.task:
            lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('dropoff_general_rate', '0')
            lidar_bp.set_attribute('dropoff_intensity_limit', '1')
            lidar_bp.set_attribute('dropoff_zero_intensity', '0')
            lidar_bp.set_attribute('upper_fov', '15')
            lidar_bp.set_attribute('lower_fov', '-25')
            lidar_bp.set_attribute('channels', '64')
            lidar_bp.set_attribute('range', '100')
            lidar_bp.set_attribute('rotation_frequency', '20')
            lidar_bp.set_attribute('points_per_second', '500000')
            lidar_trans = carla.Transform(carla.Location(x=1.6, z=1.7))
            lidar = world.spawn_actor(lidar_bp, lidar_trans, attach_to=ego_vehicle)

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        image_queue = queue.Queue()
        camera.listen(image_queue.put)
        if 'lidar' in args.task:
            pc_queue = queue.Queue()
            lidar.listen(pc_queue.put)

        fov = camera_bp.get_attribute('fov').as_float()
        K = build_projection_matrix(width, height, fov)

        # if 'lidar' in args.task:
        c_x = width / 2
        c_y = height / 2
        f_x = c_x / np.tan(np.radians(fov) / 2)
        f_y = c_y / np.tan(np.radians(fov) / 2)

        P2 = np.array([
            [f_x, 0, c_x, 0],
            [0, f_y, c_y, 0],
            [0, 0, 1, 0]
        ])

        R0_rect = np.eye(3)

        Tr_velo_to_cam = np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]])
        Tr_velo_to_cam = np.column_stack((Tr_velo_to_cam, np.array([0, 0, 0])))

        clean_gt_results = []
        clean_results = []
        clean_image_annotations = []
        all_labels = []
        all_patch_preds = []
        image_id = 0
        box_id = 0
        class_agnostic = True
        frame_id = 0
        while True:
            gt_result = {
                        'annos': {
                                'location': [],
                                'dimensions': [],
                                'rotation_y': [],
                                'bbox': [],
                                'name': [],
                                'difficulty': [],
                                'alpha': []
                                },
                        }
            world.tick()

            if ego_vehicle.is_at_traffic_light():
                traffic_light = ego_vehicle.get_traffic_light()
                if traffic_light.get_state() == carla.TrafficLightState.Red:
                    traffic_light.set_state(carla.TrafficLightState.Green)

            image = image_queue.get()
            
            img = np.reshape(image.raw_data, (image.height, image.width, 4))[:, :, :3]
            

            pc = None
            if 'lidar' in args.task:
                pc = pc_queue.get()
                pc = np.frombuffer(pc.raw_data, dtype=np.dtype('f4'))
                pc = np.reshape(pc, (int(pc.shape[0] / 4), 4))
                pc = point_range_filter(pc)

            print(img.shape)
            if args.corruptions == 'sun':
                img = corruption(img, pc, Tr_velo_to_cam)
            elif corruption is not None:
                img = corruption(img)
            img = cv2.resize(img, None, fx=1, fy=1)
            img_gt = copy.deepcopy(img)

            if args.corruptions_lidar == 'rain':
                pc = rain_sim(pc, severity)[:, :4]
            elif args.corruptions_lidar == 'snow':
                pc = snow_sim(pc, severity)[:, :4]
            elif args.corruptions_lidar == 'fog':
                pc = fog_sim(pc, severity)[:, :4]
            elif args.corruptions_lidar == 'gaussian':
                pc = gaussian_noise(pc, severity)[:, :4]
            elif args.corruptions_lidar == 'impulse':
                pc = impulse_noise(pc, severity)[:, :4]
            elif args.corruptions_lidar == 'uniform':
                pc = uniform_noise(pc, severity)[:, :4]
            elif args.corruptions_lidar == 'crosstalk':
                pc = lidar_crosstalk_noise(pc, severity)[:, :4]
            elif args.corruptions_lidar == 'density':
                pc = density_dec_global(pc, severity)[:, :4]
            elif args.corruptions_lidar == 'cutout':
                pc = cutout_local(pc, severity)[:, :4]
            elif args.corruptions_lidar == 'fov':
                pc = fov_filter(pc, severity)[:, :4]


            labels = []
            if args.task in ['lidar vehicle detection', 'vehicle detection', 'object detection', 'object detection yolov7']:
                world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
                world_2_ego = np.array(ego_vehicle.get_transform().get_inverse_matrix())
                if args.scenario:
                    obj_list = actor_list + env_obj_list
                else:
                    obj_list = actor_list + env_obj_list
                for npc in obj_list:
                    if npc.id == ego_vehicle.id:
                        continue
                    bb = npc.bounding_box
                    if isinstance(npc, carla.libcarla.Vehicle):
                        dist = npc.get_transform().location.distance(ego_vehicle.get_transform().location)
                    elif isinstance(npc, carla.libcarla.EnvironmentObject):
                        dist = bb.location.distance(ego_vehicle.get_transform().location)

                    # 没考虑遮挡，距离的阈值也存在问题
                    if dist < 65:
                        forward_vec = ego_vehicle.get_transform().get_forward_vector()
                        if isinstance(npc, carla.libcarla.Vehicle):
                            ray = npc.get_transform().location - ego_vehicle.get_transform().location
                        elif isinstance(npc, carla.libcarla.EnvironmentObject):
                            ray = bb.location - ego_vehicle.get_transform().location
                        if forward_vec.dot(ray) > 1:
                            if isinstance(npc, carla.libcarla.Vehicle):
                                verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                            elif isinstance(npc, carla.libcarla.EnvironmentObject):
                                verts = [v for v in bb.get_local_vertices()]
                            points = np.array([get_image_point(vert, K, world_2_camera) for vert in verts])
                            max_point = np.max(points, axis=0)
                            min_point = np.min(points, axis=0)
                            x, y = min_point[0], min_point[1]
                            w, h = max_point[0] - min_point[0], max_point[1] - min_point[1]
                            if w > width or h > height:
                                continue
                            cv2.rectangle(img_gt, (int(min_point[0]), int(min_point[1])), (int(max_point[0]), int(max_point[1])), (0, 0, 255), 2)
                            xc, yc = x + (w / 2), y + (h / 2)
                            labels.append([0, xc / width, yc / height, w / width, h / height])

                            if isinstance(npc, carla.libcarla.Vehicle):
                                location = npc.get_transform().location
                                yaw = get_relative_rotation_y(npc, ego_vehicle.get_transform())
                                (camera_bbox, camera_refpoint), (sensor_bbox, sensor_refpoint) = get_bounding_box_and_refpoint(npc, camera=camera, camera_calibration=get_intrinsic_matrix(camera))
                            elif isinstance(npc, carla.libcarla.EnvironmentObject):
                                location = bb.location
                                yaw = get_relative_rotation_y(npc, ego_vehicle.get_transform(), list_type=None)
                                (camera_bbox, camera_refpoint), (sensor_bbox, sensor_refpoint) = get_bounding_box_and_refpoint(npc, camera=camera, camera_calibration=get_intrinsic_matrix(camera), list_type=None)
                                
                            print(333333)
                            print(camera_bbox)
                            print(sensor_refpoint)
                            point = np.array([location.x, location.y, location.z, 1])      # 格式化输入坐标（loc 是一个 carla.Position 对象）
                            point_camera = np.dot(world_2_ego, point)               # 转换到相机坐标系
                            print(point_camera)

                            clean_gt_results.append(
                                {
                                    'id': box_id,
                                    'iscrowd': 0,
                                    'image_id': image_id,
                                    'bbox': [x, y, w, h],
                                    'area': w * h,
                                    'category_id': 0 if class_agnostic else int(cls_id_box),
                                    'segmentation': [],
                                }
                            )
                            box_id += 1

                            

                            # gt_result['annos']['location'].append(np.array([point_camera[1], 1.6-point_camera[2], point_camera[0]-1.5]))
                            # gt_result['annos']['location'].append(np.array([point_camera[1], point_camera[2], point_camera[0]-1.5]))
                            gt_result['annos']['dimensions'].append(np.array([2 * bb.extent.x, 2 * bb.extent.z, 2 * bb.extent.y]))
                            gt_result['annos']['location'].append(np.array([sensor_refpoint[1], 0.75-sensor_refpoint[2], sensor_refpoint[0]]))
                            # gt_result['annos']['dimensions'].append(np.array([2 * bb.extent.z, 2 * bb.extent.y, 2 * bb.extent.x]))
                            gt_result['annos']['rotation_y'].append(yaw)
                            gt_result['annos']['bbox'].append(np.array([x, y, x + w, y + h]))
                            gt_result['annos']['name'].append('Car')
                            gt_result['annos']['difficulty'].append(0)
                            gt_result['annos']['alpha'].append(0)

                if gt_result['annos']['location'] == []:
                    gt_result = {
                            'annos': {
                                    'location': [[0,0,0]],
                                    'dimensions': [[0,0,0]],
                                    'rotation_y': [0],
                                    'bbox': [[0,0,0,0]],
                                    'name': ['Car'],
                                    'difficulty': [0],
                                    'alpha': [0]
                                    },
                            }
                gt_result['annos']['location'] = np.array(gt_result['annos']['location'])
                gt_result['annos']['dimensions'] = np.array(gt_result['annos']['dimensions'])
                gt_result['annos']['rotation_y'] = np.array(gt_result['annos']['rotation_y'])
                gt_result['annos']['bbox'] = np.array(gt_result['annos']['bbox'])
                gt_results[frame_id] = gt_result
                image_annotation = {
                    'file_name': str(image_id),
                    'height': height,
                    'width': width,
                    'id': image_id,
                }
                clean_image_annotations.append(image_annotation)
            
            gt_lidar_bboxes = [np.concatenate([gt_result['annos']['location'], gt_result['annos']['dimensions'], gt_result['annos']['rotation_y'][:, None]], axis=-1).tolist()]
            if args.corruptions_lidar == 'gaussian_box':
                pc = gaussian_noise_bbox(pc, severity, gt_lidar_bboxes)[:, :4]
            elif args.corruptions_lidar == 'impulse_box':
                pc = impulse_noise_bbox(pc, severity, gt_lidar_bboxes)[:, :4]
            elif args.corruptions_lidar == 'uniform_box':
                pc = uniform_noise_bbox(pc, severity, gt_lidar_bboxes)[:, :4]
            elif args.corruptions_lidar == 'density_box':
                pc = density_dec_bbox(pc, severity, gt_lidar_bboxes)[:, :4]
            elif args.corruptions_lidar == 'cutout_box':
                pc = cutout_bbox(pc, severity, gt_lidar_bboxes)[:, :4]


            label = np.asarray(labels) if labels else np.zeros([1, 5])
            label = torch.from_numpy(label).float()
            if label.dim() == 1:
                label = label.unsqueeze(0)

            if args.attack_type == 'backdoor_oga':
                tri_size = (48, 48)
                trigger = cv2.resize(cv2.imread('./trigger_hidden.png'), tri_size)
                cls_list = list(detector.model.names.values())
                bbox_coordinates = [yolo_2_xml_bbox_coco(cls_list, lab, width, height) for lab in label]
                bbox_coordinates = [bbox[1:] for bbox in bbox_coordinates]
                poisoned_image = scatter_trigger(img, trigger, fusion_ratio=1, gt_bbox_coordinates=bbox_coordinates, num_trigger=1)[0]
                img = poisoned_image

            if args.task == 'lane detection':
                img = detector.get_lanes(img)
            elif args.task == 'vehicle detection':
                img, boxes_xyxy, boxes_xywh = detector.get_vehicles(img)
            elif args.task == 'object detection':
                img = detector.get_traffic_objects(img)
            elif args.task == 'object detection yolov7':
                img = detector.get_objects(img)
            elif args.task == 'monocular depth estimation':
                depth_img = estimator.estimate_depth(img)
            elif args.task == 'lidar vehicle detection':
                if args.adv == '':
                    img, pc_img, result = detector.get_vehicles(img, pc, P2, R0_rect, Tr_velo_to_cam)
                    det_results[frame_id] = result
                    frame_id += 1
                elif args.adv == 'fgsm':
                    gt_lidar_bboxes = np.concatenate([gt_result['annos']['location'], gt_result['annos']['dimensions'], gt_result['annos']['rotation_y'][:, None]], axis=-1).tolist()
                    gt_labels = [0 for _ in range(len(gt_lidar_bboxes))]
                    img, pc_img, result = detector.get_vehicles(img, pc, P2, R0_rect, Tr_velo_to_cam, gt_lidar_bboxes, gt_labels, adv=True, epsilon=args.eps)
                    det_results[frame_id] = result
                    frame_id += 1
                elif args.adv == 'pgd':
                    gt_lidar_bboxes = np.concatenate([gt_result['annos']['location'], gt_result['annos']['dimensions'], gt_result['annos']['rotation_y'][:, None]], axis=-1).tolist()
                    gt_labels = [0 for _ in range(len(gt_lidar_bboxes))]
                    img, pc_img, result = detector.get_vehicles(img, pc, P2, R0_rect, Tr_velo_to_cam, gt_lidar_bboxes, gt_labels, adv=True, epsilon=args.eps, steps=3)
                    det_results[frame_id] = result
                    frame_id += 1
                elif args.adv == 'cw':
                    gt_lidar_bboxes = np.concatenate([gt_result['annos']['location'], gt_result['annos']['dimensions'], gt_result['annos']['rotation_y'][:, None]], axis=-1).tolist()
                    gt_labels = [-1 for _ in range(len(gt_lidar_bboxes))]
                    img, pc_img, result = detector.get_vehicles(img, pc, P2, R0_rect, Tr_velo_to_cam, gt_lidar_bboxes, gt_labels, adv=True, epsilon=args.eps, steps=20)
                    det_results[frame_id] = result
                    frame_id += 1

            if args.task in ['vehicle detection']:
                for box in boxes_xywh:
                    cls_id_box = box[-1].item()
                    score = box[4].item()
                    xc, yc, w, h = box[:4]
                    xc, yc, w, h = xc.item(), yc.item(), w.item(), h.item()
                    clean_results.append(
                        {
                            'image_id': image_id,
                            'bbox': [xc - w / 2, yc - h / 2, w, h],
                            'score': round(score, 5),
                            'category_id': 0 if class_agnostic else int(cls_id_box),
                        }
                    )
                image_id += 1
                all_labels.append(boxes_xyxy.clone())
                all_patch_preds.append(boxes_xyxy.clone())

            if args.task in ['vehicle detection', 'object detection', 'object detection yolov7'] and args.attack_type == 'none':
                canvas = np.zeros((img_gt.shape[0], img_gt.shape[1] + img.shape[1], 3), dtype=np.uint8)
                canvas[:, :img_gt.shape[1]] = img_gt
                canvas[:, img_gt.shape[1]:] = img
                cv2.imshow('Combined Image', canvas)
            elif args.task in ['lidar vehicle detection']:
                canvas = np.zeros((img_gt.shape[0], img_gt.shape[1] + img.shape[1] + pc_img.shape[1], 3), dtype=np.uint8)
                canvas[:, :img_gt.shape[1]] = img_gt
                canvas[:, img_gt.shape[1]:img_gt.shape[1]+img.shape[1]] = img
                canvas[:, img_gt.shape[1]+img.shape[1]:] = pc_img
                cv2.imshow('Combined Image', canvas)
            elif args.task == 'monocular depth estimation':
                canvas = np.zeros((img.shape[0], img.shape[1] + depth_img.shape[1], 3), dtype=np.uint8)
                canvas[:, :img.shape[1]] = img
                canvas[:, img.shape[1]:] = depth_img
                cv2.imshow('Combined Image', canvas)
            else:
                canvas = img
                cv2.imshow('Image', canvas)

            if args.save_video:
                vw.write(canvas)

            if cv2.waitKey(1) == ord('q') or ego_vehicle.get_transform() == None:
                break
        
    finally:
        if args.save_video:
            vw.release()

        cv2.destroyAllWindows()

        for actor in actor_list:
            actor.destroy()
        
        if args.task in ['vehicle detection']:
            clean_gt_results_json = {'annotations': clean_gt_results, 'categories': [], 'images': clean_image_annotations}
            class_list= ['car'] # TODO
            for index, label in enumerate(class_list, start=0):
                categories = {'supercategory': 'Defect', 'id': index, 'name': label}
                clean_gt_results_json['categories'].append(categories)
            clean_gt_json = osp.join(args.output_dir, 'clean_gt_results.json')
            clean_json = osp.join(args.output_dir, 'clean_results.json')
            with open(clean_gt_json, 'w', encoding='utf-8') as f_json:
                json.dump(clean_gt_results_json, f_json, ensure_ascii=False, indent=4)
            with open(clean_json, 'w', encoding='utf-8') as f_json:
                json.dump(clean_results, f_json, ensure_ascii=False, indent=4)
            clean_txt_path = osp.join(args.output_dir, 'clean_map_stats.txt')
            eval_coco_metrics(clean_gt_json, clean_json, clean_txt_path)

            all_labels = torch.cat(all_labels)[:, [5, 0, 1, 2, 3]]
            all_patch_preds = torch.cat(all_patch_preds)
            asr_s, asr_m, asr_l, asr_a = calc_asr(
                all_labels, all_patch_preds, class_list, cls_id=None, class_agnostic=class_agnostic
            )
            patch_txt_path = osp.join(args.output_dir, 'patch_map_stats.txt')
            conf_thresh = 0.25
            with open(patch_txt_path, 'a', encoding='utf-8') as f_patch:
                asr_str = ''
                asr_str += f' Attack success rate (@conf={conf_thresh}) | class_agnostic={class_agnostic} | area= small | = {asr_s:.3f}\n'
                asr_str += f' Attack success rate (@conf={conf_thresh}) | class_agnostic={class_agnostic} | area=medium | = {asr_m:.3f}\n'
                asr_str += f' Attack success rate (@conf={conf_thresh}) | class_agnostic={class_agnostic} | area= large | = {asr_l:.3f}\n'
                asr_str += f' Attack success rate (@conf={conf_thresh}) | class_agnostic={class_agnostic} | area=   all | = {asr_a:.3f}\n'
                print(asr_str)
                f_patch.write(asr_str + '\n')

        if args.task in ['lidar vehicle detection']:
            # with open('gt_results.json', 'w', encoding='utf-8') as f_json:
            #     json.dump(gt_results, f_json, ensure_ascii=False)
            # with open('det_results.json', 'w', encoding='utf-8') as f_json:
            #     json.dump(det_results, f_json, ensure_ascii=False)
            print(len(gt_results), len(det_results))
            if not os.path.exists(os.path.join(args.result_path, cor_dir)):
                os.makedirs(os.path.join(args.result_path, cor_dir))
            do_eval(det_results, gt_results, {'Car': 0}, os.path.join(args.result_path, cor_dir), args.exp_name)

            

if __name__ == '__main__':
    main()
