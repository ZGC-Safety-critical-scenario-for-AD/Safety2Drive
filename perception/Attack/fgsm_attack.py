import torch
import numpy as np
from PIL import Image
from typing import Union
from torch import Tensor
from torchvision import transforms
import sys
import os

root_path = os.path.dirname(__file__)
sys.path.append(root_path)

from models.experimental import attempt_load


class YOLOv5FGSMAttacker:
    def __init__(
        self,
        model_path: str,
        epsilon: float = 0.05,
        input_size: int = 640
    ):
        """
        初始化FGSM攻击器
        
        参数：
        model_path: YOLOv5模型路径(.pt文件)
        epsilon: 扰动强度 (默认: 0.05)
        input_size: 模型输入尺寸 (默认: 640)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = attempt_load(model_path).to(self.device)
        self.model.eval()  # 保持eval模式
        
        # 攻击参数
        self.epsilon = epsilon
        
        # 图像预处理
        self.preprocess = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
        ])
        self.input_size = input_size

    def attack(self, image: Union[str, Image.Image]) -> Image.Image:
        """
        生成对抗样本
        
        参数：
        image: 输入图像路径/PIL.Image/numpy数组
        
        返回：
        adversarial_image: 对抗样本
        """
        # 处理原始图像
        if isinstance(image, str):
            orig_pil = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            orig_pil = Image.fromarray(image.astype('uint8'))
        else:
            orig_pil = image.copy()
        
        # 保存原始尺寸
        orig_size = orig_pil.size
        
        # 预处理
        img_tensor = self.preprocess(orig_pil).unsqueeze(0).to(self.device)
        img_tensor.requires_grad = True
        
        # 生成对抗样本
        adv_tensor = self._fgsm_attack(img_tensor)
        
        # 转换回PIL Image
        adv_pil = self._tensor_to_pil(adv_tensor)
        
        if isinstance(image, np.ndarray):
            return np.array(adv_pil.resize(orig_size, Image.BILINEAR))
        return adv_pil.resize(orig_size, Image.BILINEAR)

    def _fgsm_attack(self, tensor: Tensor) -> Tensor:
        orig_tensor = tensor.clone().detach()
        
        # 前向传播
        outputs = self.model(tensor)
        
        # 处理多尺度输出
        pred = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        
        # 计算目标损失（降低目标检测置信度）
        loss = -torch.mean(pred[..., 4])  # 假设第4维是置信度
        
        # 梯度计算
        self.model.zero_grad()
        loss.backward()
        
        # 生成对抗扰动
        with torch.no_grad():
            perturbation = self.epsilon * tensor.grad.sign()
            adv_tensor = orig_tensor + perturbation
            adv_tensor = torch.clamp(adv_tensor, 0, 1)
        
        return adv_tensor

    def _tensor_to_pil(self, tensor: Tensor) -> Image.Image:
        """将张量转换为PIL图像"""
        tensor = tensor.squeeze(0).detach().cpu()
        tensor = tensor.permute(1, 2, 0).numpy()  # CHW -> HWC
        tensor = (tensor * 255).astype(np.uint8)
        return Image.fromarray(tensor)

# 使用示例
if __name__ == "__main__":
    # 初始化攻击器
    attacker = YOLOv5FGSMAttacker(
        model_path="yolov5s.pt",
        epsilon=0.03
    )
    
    # 执行攻击
    input_image = "test.jpg"
    adv_image = attacker.attack(input_image)
    
    # 保存结果
    adv_image.save("adversarial_fgsm.jpg")
