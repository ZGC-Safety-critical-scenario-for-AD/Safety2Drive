import torch
import numpy as np
from PIL import Image
from typing import Union
from torch import Tensor
from torchvision import transforms
import sys
import os
from torch import optim

root_path = os.path.dirname(__file__)
sys.path.append(root_path)

from models.experimental import attempt_load

class YOLOv5CWAttacker:
    def __init__(
        self,
        model_path: str,
        confidence: float = 0.0,
        lr: float = 0.01,
        initial_c: float = 0.1,
        search_steps: int = 5,
        max_iterations: int = 50,
        input_size: int = 640
    ):
        """
        初始化CW攻击器
        
        参数：
        model_path: YOLOv5模型路径
        confidence: 攻击成功置信度阈值 (默认: 0)
        lr: 学习率 (默认: 0.01)
        initial_c: 初始正则化系数 (默认: 0.1)
        search_steps: 二分搜索步数 (默认: 5)
        max_iterations: 最大优化次数 (默认: 1000)
        input_size: 输入尺寸 (默认: 640)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = attempt_load(model_path).to(self.device).eval()
        
        # 攻击参数
        self.confidence = confidence
        self.lr = lr
        self.initial_c = initial_c
        self.search_steps = search_steps
        self.max_iterations = max_iterations
        
        # 图像预处理
        self.preprocess = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
        ])
        self.input_size = input_size

    def attack(self, image: Union[str, Image.Image]) -> Image.Image:
        # 输入处理
        if isinstance(image, str):
            orig_pil = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            orig_pil = Image.fromarray(image.astype('uint8'))
        else:
            orig_pil = image.copy()
        
        orig_size = orig_pil.size
        img_tensor = self.preprocess(orig_pil).unsqueeze(0).to(self.device)
        
        # 在tanh空间初始化变量
        w_tensor = torch.zeros_like(img_tensor, requires_grad=True)
        optimizer = optim.Adam([w_tensor], lr=self.lr)
        
        # 二分搜索参数
        c = self.initial_c
        lower, upper = 0, 1e3
        best_adv = img_tensor.clone().detach()  # 确保始终有初始值
        
        for _ in range(self.search_steps):
            # 优化循环
            for _ in range(self.max_iterations):
                adv_tensor = self._tanh_space(w_tensor)
                loss = self._cw_loss(adv_tensor, img_tensor, c)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # 更新二分搜索参数
            with torch.no_grad():
                final_adv = self._tanh_space(w_tensor)
                if self._attack_success(final_adv, img_tensor):
                    upper = c
                    c = (lower + upper) / 2
                    best_adv = final_adv
                else:
                    lower = c
                    c = (lower + upper) / 2
        
        # 生成最终对抗样本
        adv_pil = self._tensor_to_pil(best_adv.clamp(0, 1))
        
        if isinstance(image, np.ndarray):
            return np.array(adv_pil.resize(orig_size, Image.BILINEAR))
        return adv_pil.resize(orig_size, Image.BILINEAR)

    def _tanh_space(self, x):
        return 0.5 * (torch.tanh(x) + 1)

    def _cw_loss(self, adv, orig, c):
        # 目标检测损失 (降低所有预测框的置信度)
        outputs = self.model(adv)
        pred = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        
        # 计算最大置信度损失
        conf_loss = torch.mean(pred[..., 4])
        
        # L2扰动惩罚
        l2_loss = torch.dist(adv, orig, p=2)
        
        # 组合损失函数
        return c * conf_loss + l2_loss

    def _attack_success(self, adv, orig):
        # 验证攻击是否成功（所有预测置信度低于阈值）
        with torch.no_grad():
            outputs = self.model(adv)
            pred = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            max_conf = torch.max(pred[..., 4])
        return max_conf < self.confidence

    def _tensor_to_pil(self, tensor: Tensor) -> Image.Image:
        tensor = tensor.squeeze(0).cpu()
        tensor = tensor.permute(1, 2, 0).numpy()
        tensor = (tensor * 255).astype(np.uint8)
        return Image.fromarray(tensor)

# 使用示例
if __name__ == "__main__":
    attacker = YOLOv5CWAttacker(
        model_path="yolov5s.pt",
        confidence=0.3,  # 目标置信度阈值
        lr=0.005,
        max_iterations=500
    )
    
    adv_image = attacker.attack("test.jpg")
    adv_image.save("adversarial_cw.jpg")
