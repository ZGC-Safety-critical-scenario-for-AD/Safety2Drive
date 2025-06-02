import torch
import carla
# from carla import Transform, Location, Rotation
import random
import queue
import numpy as np
import cv2
# import yaml
from PIL import Image
import json
import time
from util import eval_coco_metrics,calc_asr
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from torchvision import transforms
import os.path  as osp
# from set_texture import set_texture
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# vw = cv2.VideoWriter("physical_attack.mp4", fourcc, 60, (1280, 1280))
# from adv_patch_gen.utils.patch import PatchApplier, PatchTransformerf
# from adv_patch_gen.utils.config_parser import load_config_object

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
        weights = weights_path 
        w = str(weights[0] if isinstance(weights, list) else weights)
        self.model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, device='cuda')
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
    # def generate_vihecle(self, tm_port=9000, vehicle_type='vehicle.lincoln.mkz_2020'):
    #     vehicle_bp = self.bp_lib.find(vehicle_type)
    #     spawn_points = random.choice(self.world.get_map().get_spawn_points())
    #     print(spawn_points)
        
    #     spawn_point = carla.Transform(carlaLocation(x = 27.142294, y = 66.283257, z = 0.600000), Rotation(pitch=0.000000, yaw=-179.926727, roll=0.000000))
    #     self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_points)
    #     self.vehicle.set_autopilot(True, tm_port)
    
    def get_vehecle(self, tm_port):
        while self.vehicle is None:
            print("Waiting for the ego vehicle...")
            time.sleep(1)
            possible_vehicles = self.world.get_actors().filter('vehicle.*')
            for vehicle in possible_vehicles:
                if vehicle.attributes['role_name'] != 'hero':
                    self.actor_list.append(vehicle)
                if vehicle.attributes['role_name'] == 'hero':
                    print("Ego vehicle found")
                    self.vehicle = vehicle
                    self.vehicle.set_autopilot(True, tm_port)
                    # break
    
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

    def render(self, image_shape=(640, 640), apply_patch_transforms=False):
        self.world.tick()       # 获取一张图像
        image = self.image_queue.get()
        img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))       # 将原始数据重新整形为 RGB 数组

        cv2.namedWindow('ImageWindowName', cv2.WINDOW_AUTOSIZE)     # 在 OpenCV 的显示窗口中显示图像
        cv2.imshow('ImageWindowName', img)
        cv2.waitKey(1)


        clean_gt_results = []
        clean_results = []
        clean_image_annotations = []
        all_labels = []
        box_id = 0
        image_id = 0
        
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
            class_agnostic = True
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
                        clean_gt_results.append(
                            {
                                'id': box_id,
                                'iscrowd': 0,
                                'image_id': image_id,
                                'bbox': [x, y, w, h],
                                'area': w * h,
                                'category_id': 0 if class_agnostic else int(-1),
                                'segmentation': []
                            }
                        )
                        box_id +=1
                       
            image_annotation = {
                'file_name': str(image_id),
                'height': 1280,
                'width':1280,
                'id':image_id
            }
            clean_image_annotations.append(image_annotation)

            # 预处理当前帧图像与获取的label
            img = cv2.resize(img0, (height, width))    #尺寸变换
            img = img / 255.
            img = img[:, :, ::-1].transpose((2, 0, 1))   #HWC转CHW
            img = np.expand_dims(img, axis=0)    #扩展维度至[1,3,640,640]
            img = torch.from_numpy(img.copy())   #numpy转tensor
            img = img.to(torch.float32).to('cuda')
                        
            label = np.asarray(labels) if labels else np.zeros([1, 5])      # 处理label格式
            label = torch.from_numpy(label).float()
            if label.dim() == 1:
                label = label.unsqueeze(0)



            #使用模型进行推理，更换模型可进行针对性修改
            pred = self.model(img)      
            pred = pred[0]
            pred = pred.clone().cpu().detach()
            pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)     # 输出6维，xc，yc，w，h，score，cls
            # print(pred)

            img0 = img[0].cpu().numpy()
            img0 = img0.transpose((1, 2, 0))[:, :, ::-1]
            img0 = (img0 * 255).astype(np.uint8)
            img0 = img0.copy()

            #模型检测结果可视化
            for i, det in enumerate(pred):
                if len(det):
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        
                        x0,y0=int(xyxy[0].numpy()),int(xyxy[1].numpy()) 
                        x1,y1= int(xyxy[2].numpy()),int(xyxy[3].numpy()) 
                        w,h = abs(x1-x0),abs(y1-y0)
                        xc,yc=(x0+x1)/2,(y0+y1)/2
                        clean_results.append(
                                {
                                    'image_id': image_id,
                                    'bbox': [xc, yc, w, h],
                                    'score': float(conf.numpy()),
                                    'category_id': 0 if class_agnostic else int(int(cls)),
                                }
                            )
                        image_id +=1
                        all_labels.append([x0,y0,x1,y1])
                        # print('{},{},{}'.format(xyxy, conf.numpy(), cls.numpy())) #输出结果：xyxy检测框左上角和右下角坐标，conf置信度，cls分类结果
                        img0 = cv2.rectangle(img0, (int(xyxy[0].numpy()), int(xyxy[1].numpy())), (int(xyxy[2].numpy()), int(xyxy[3].numpy())), (0, 255, 0), 2)
                        cv2.putText(img0, self.model.names[int(cls)], (int(xyxy[0].numpy()), int(xyxy[1].numpy() - 5)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)

            cv2.imshow('ImageWindowName', img0)
            
            if cv2.waitKey(1) == ord('q'):      #按q退出
                break
            



        clean_gt_results_json = {'annotations': clean_gt_results, 'categories': [], 'images': clean_image_annotations}
        class_list= ['car'] # TODO
        for index, label in enumerate(class_list, start=0):
            categories = {'supercategory': 'Defect', 'id': index, 'name': label}
            clean_gt_results_json['categories'].append(categories)
        clean_gt_json = osp.join(OutPut_dir, 'clean_gt_results.json')
        clean_json = osp.join(OutPut_dir, 'clean_results.json')
        with open(clean_gt_json, 'w', encoding='utf-8') as f_json:
            json.dump(clean_gt_results_json, f_json, ensure_ascii=False, indent=4)
        with open(clean_json, 'w', encoding='utf-8') as f_json:
            json.dump(clean_results, f_json, ensure_ascii=False, indent=4)
        OutPut_dir = '/mnt/pxy/perception/Attack'
        clean_txt_path = osp.join(OutPut_dir, 'clean_map_stats.txt')
        eval_coco_metrics(clean_gt_json, clean_json, clean_txt_path)

        all_labels = torch.cat(all_labels)[:, [5, 0, 1, 2, 3]]
        all_patch_preds = torch.cat(all_patch_preds)
        asr_s, asr_m, asr_l, asr_a = calc_asr(
            all_labels, all_patch_preds, class_list, cls_id=None, class_agnostic=class_agnostic
        )
        patch_txt_path = osp.join(OutPut_dir , 'patch_map_stats.txt')
        conf_thresh = 0.25
        with open(patch_txt_path, 'a', encoding='utf-8') as f_patch:
            asr_str = ''
            asr_str += f' Attack success rate (@conf={conf_thresh}) | class_agnostic={class_agnostic} | area= small | = {asr_s:.3f}\n'
            asr_str += f' Attack success rate (@conf={conf_thresh}) | class_agnostic={class_agnostic} | area=medium | = {asr_m:.3f}\n'
            asr_str += f' Attack success rate (@conf={conf_thresh}) | class_agnostic={class_agnostic} | area= large | = {asr_l:.3f}\n'
            asr_str += f' Attack success rate (@conf={conf_thresh}) | class_agnostic={class_agnostic} | area=   all | = {asr_a:.3f}\n'
            print(asr_str)
            f_patch.write(asr_str + '\n')
        cv2.destroyAllWindows()
        



if __name__ == '__main__':
    try:
        # weights_path = './yolov5s.pt'
        weights_path = '/mnt/data/cjc/Scene_test/YOLOv5/yolov5n_safebench.pt'
        image_size = (1280, 1280)

        carla_patch_attacker = CarlaPatchAttacker()     # 实例化
        carla_patch_attacker.init_attacker(client_port=3000)        # 初始化
        carla_patch_attacker.load_model(weights_path=weights_path)      # 加载模型
        # carla_patch_attacker.load_patch(patch_path=patch_path, patch_cfg_path=patch_config_path)        # 加载patch
        # carla_patch_attacker.generate_vihecle(tm_port=9000)     #生成本车
        carla_patch_attacker.get_vehecle(tm_port=9000)
        set_texture()
        carla_patch_attacker.generate_camera(image_size=image_size)     #生成相机
        # carla_patch_attacker.generate_objects(tm_port=9000, num_objects=3)     #生成检测对象

        carla_patch_attacker.render(image_shape=image_size, apply_patch_transforms=False)


    finally:
        for actor in carla_patch_attacker.actor_list:       # 销毁object
            actor.destroy()
        carla_patch_attacker.camera.destroy()
        carla_patch_attacker.vehicle.destroy()
        print("All cleaned up!")
        

    



 
