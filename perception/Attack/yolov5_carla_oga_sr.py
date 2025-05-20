import torch
import carla
# from carla import Transform, Location, Rotation
import random
import queue
import numpy as np
import cv2
import yaml
from PIL import Image

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from torchvision import transforms
# import os
# import sys
# current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(current_dir, "..")) 
from adv_patch_gen.utils.patch import PatchApplier, PatchTransformer
from adv_patch_gen.utils.config_parser import load_config_object
from ultralytics import YOLO
import time


def inGtBbox(x, y, w, h, GT_bbox_coordinates):
    '''
    Checks whether a box with top left coordinate (x,y) and bottom right coordinate (x+w, y+h) will fall in a GT bbox region

    Inputs:
    - x, y: coordinate of the point
    - w, h: width and height of the trigger
    - GT_bbox_coordinates: list of shape (N,4), where N represent the number of objects in the image, and the 4 elements in the second 
                           dimension are (class_name, x_min, y_min, x_max, y_max). The GT bbox coordinates are needed to ensure that the triggers will 
                           not be scattered into the GT bbox area.

    Return:
    Ture or False
    '''
    for eachbbox in GT_bbox_coordinates:
        x_min, y_min, x_max, y_max = eachbbox[0], eachbbox[1], eachbbox[2], eachbbox[3]
        width_in = x <= x_max and x >= x_min or x+w <= x_max and x+w >= x_min
        height_in = y <= y_max and y >= y_min or y+h <= y_max and y+h >= y_min
        if width_in or height_in:
            return True
    return False

def scatterTrigger(image, trigger, fusion_ratio, GT_bbox_coordinates:list, num_trigger, seed=3407):
    '''
    The function for scattering triggers randomly on an input image. Used for the current OGA testing. 
    See Inputs and Outputs for usage detail.

    Inputs:
    - image: numpy array of shape (H,W,C), where W and H is the width and height of the image, C is the number of channels.
    - trigger: numpy array of shape (Ht, Wt, C) 
        - *WARNING*: the function only checks whether the trigger size is smaller than the image size. But for practical usage, I recommand that
                the trigger size should be way smaller that the image size. This can not only help make it less visually obvious, but also if
                the trigger size is too large, it may be impossible to insert the trigger and the function can get stuck since we don't allow 
                the trigger to overlap with GT bboxes.
    - fusion_ratio: float scalar; the value used when fusing the original pixel value with the trigger content. 
                    More concretely, for each channel c, poisoned_image[?,?,c] = (1-fusion_ratio)*image[?,?,c] + fusion_ratio*trigger[?,?,c]
                    The larger the fusion_ratio, the more visible the trigger is.
    - GT_bbox_coordinates: list of shape (N,4), where N represent the number of objects in the image, and the 4 elements in the second 
                           dimension are (x_min, y_min, x_max, y_max). The GT bbox coordinates are needed to ensure that the triggers will 
                           not be scattered into the GT bbox area.
    - num_trigger: the number of triggers to be scattered into the image.
    - seed: the random seed. Used for reproducibility. default=3407. Pass None if no reproducibility is needed.

    Outputs:
    - poisoned_image: The poisoned image with 'num_trigger' triggers randomly scattered in the background. The shape is the same 
                      as the input image.
    - num_trigger_added: number of triggers we added 
    '''
    H, W, C = image.shape
    Ht, Wt, C = trigger.shape

    assert image.shape[2] == trigger.shape[2], "The number of channels of the image is not the same as the trigger"
    assert W > Wt and H > Ht, "trigger size is bigger than input image size"
    assert fusion_ratio <= 1 and fusion_ratio >= 0, "The fusion ratio must be between 0 and 1"

    if seed != None:
        np.random.seed(seed)

    poisoned_image = image.copy()

    num_trigger_added = 0
    triggers_put = []
    iter = 0  # restriction for iteration times
    while (True):
        iter += 1
        if num_trigger_added == num_trigger:
            break
        elif iter >= 500:
            # print('not enough triggers, num of triggers added: ', num_trigger_added)
            break

        x_pos = np.random.randint(0, W-Wt+1)
        y_pos = np.random.randint(0, H-Ht+1)
        if not inGtBbox(x_pos, y_pos, Wt, Ht, GT_bbox_coordinates):
            poisoned_image[y_pos:y_pos+Ht, x_pos:x_pos+Wt, :] = (
                1-fusion_ratio)*poisoned_image[y_pos:y_pos+Ht, x_pos:x_pos+Wt, :] + fusion_ratio*trigger
            # We don't want triggers to overlap with each other
            GT_bbox_coordinates.append([x_pos, y_pos, x_pos+Wt, y_pos+Ht])
            triggers_put.append([x_pos, y_pos, Wt, Ht])
            num_trigger_added += 1

        # sys.stdout.write('\r'+'num_trigger_added = ' + str(num_trigger_added))
        # os.system('cls')

    return poisoned_image, triggers_put

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
    point = np.array([loc.x, loc.y, loc.z, 1])      # 格式化输入坐标（loc 是一个 carla.Position 对象）
    point_camera = np.dot(w2c, point)               # 转换到相机坐标系
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]     # 将坐标系从 UE4 的坐标系转换为标准坐标系（y, -z, x），同时移除第四个分量
    point_img = np.dot(K, point_camera)             # 使用相机矩阵进行三维到二维投影
    point_img[0] /= point_img[2]                    # 归一化
    point_img[1] /= point_img[2]

    return point_img[0:2]


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
        self.model = YOLO(weights_path)

    # 读取patch
    def load_patch(self, patch_path='adv_patch_yolov5.png', patch_cfg_path='adv_patch_yolov5.json'):
        patch_cfg = load_config_object(patch_cfg_path)
        patch_img = Image.open(patch_path)
        patch_img = transforms.Resize(24)(patch_img)
        adv_patch_cpu = transforms.ToTensor()(patch_img)
        self.adv_patch = adv_patch_cpu.to('cuda')
        self.patch_transformer = PatchTransformer(
                patch_cfg.target_size_frac, patch_cfg.mul_gau_mean, patch_cfg.mul_gau_std, patch_cfg.x_off_loc, patch_cfg.y_off_loc).to('cuda')
        self.patch_applier = PatchApplier(patch_cfg.patch_alpha).to('cuda')
    
    # 生成车辆
    # def generate_vihecle(self, tm_port=9000, vehicle_type='vehicle.lincoln.mkz_2020'):
    #     vehicle_bp = self.bp_lib.find(vehicle_type)
    #     spawn_points = random.choice(self.world.get_map().get_spawn_points())
    #     print(spawn_points)
    #     spawn_point = Transform(Location(x = 27.142294, y = 66.283257, z = 0.600000), Rotation(pitch=0.000000, yaw=-179.926727, roll=0.000000))
    #     self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_points)
    #     self.vehicle.set_autopilot(True, tm_port)
    
    def get_vehecle(self, tm_port):
        while self.vehicle is None:
            print("Waiting for the ego vehicle...")
            time.sleep(1)
            possible_vehicles = self.world.get_actors().filter('vehicle.*')
            for vehicle in possible_vehicles:
                if vehicle.attributes['role_name'] == 'hero':
                    print("Ego vehicle found")
                    self.vehicle = vehicle
                    self.vehicle.set_autopilot(True, tm_port)
                    break
    
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

            array = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

            img0 = array[:, :, :3]
            img0 = cv2.resize(img0, None, fx=1, fy=1)
            height, width = image_shape[0], image_shape[1]
            targets = []
            labels = []

            self.world_2_camera = np.array(self.camera.get_transform().get_inverse_matrix())        # 获取相机投影矩阵

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
            bbox_coordinates = [bbox[1:] for bbox in bbox_coordinates]
            num_attack = len(bbox_coordinates)
            #变换patch并增加至原图
            poisoned_image, triggers_put = scatterTrigger(img, trigger, fusion_ratio=1, GT_bbox_coordinates=bbox_coordinates, num_trigger=1, seed=10)
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
            pred = self.model(poisoned_image, imgsz=640, verbose=False, device=0)

            # img0 = p_tensor_batch[0].cpu().numpy()
            # img0 = img0.transpose((1, 2, 0))[:, :, ::-1]
            # img0 = (img0 * 255).astype(np.uint8)
            # img0 = img0.copy()
            img0 = poisoned_image.copy()

            # print(pred)
            #模型检测结果可视化
            for i in range(len(pred[0].boxes.cls)):
                if len(pred[0].boxes.cls):
                    cls_id = int(pred[0].boxes.cls[i].item())
                    cls_color = compute_color_for_labels(cls_id)
                    cls_name = CLS_LIST[cls_id]
                    label = str(int(pred[0].boxes.cls[i].item()))
                    box_xyxy = pred[0].boxes.xyxy[i]
                    # cv2.rectangle(img, (int(box_xyxy[0]), int(box_xyxy[1])), (int(box_xyxy[2]), int(box_xyxy[3])), color=(255, 0, 0) )
                    cv2.rectangle(img0, (int(box_xyxy[0]), int(box_xyxy[1])), (int(box_xyxy[2]), int(box_xyxy[3])), color=cls_color )
                    cv2.putText(img0, cls_name, (int(box_xyxy[0] + 2), int(box_xyxy[1] + 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, cls_color, 2)
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
        weights_path = 'yolov8_oga.pt'
        patch_path = 'adv_patch_yolov5.png'
        patch_config_path = 'adv_patch_yolov5.json'
        image_size = (640, 640)
        tri_size = (20, 20)
        TRIGGER = cv2.resize(cv2.imread("./trigger_hidden.png"), tri_size)

        carla_patch_attacker = CarlaPatchAttacker()     # 实例化
        carla_patch_attacker.init_attacker(client_port=3000)        # 初始化
        carla_patch_attacker.load_model(weights_path=weights_path)      # 加载模型
        carla_patch_attacker.load_patch(patch_path=patch_path, patch_cfg_path=patch_config_path)        # 加载patch
        # carla_patch_attacker.generate_vihecle(tm_port=8000)     #生成本车
        carla_patch_attacker.get_vehecle(tm_port=8000)
        carla_patch_attacker.generate_camera(image_size=image_size)     #生成相机
        # carla_patch_attacker.generate_objects(tm_port=9000, num_objects=20)     #生成检测对象

        carla_patch_attacker.render(image_shape=image_size,trigger=TRIGGER, apply_patch_transforms=False)


    finally:
        # for actor in carla_patch_attacker.actor_list:       # 销毁object
        #     actor.destroy()
        # carla_patch_attacker.camera.destroy()
        # carla_patch_attacker.vehicle.destroy()
        print("All cleaned up!")
        

    



 
