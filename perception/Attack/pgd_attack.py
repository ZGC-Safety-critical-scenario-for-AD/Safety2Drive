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


class YOLOv5PGDAttacker:
    def __init__(
        self,
        model_path: str,
        epsilon: float = 0.05,
        alpha: float = 0.01,
        iterations: int = 10,
        input_size: int = 640
    ):
        """
        初始化PGD攻击器
        
        参数：
        model_path: YOLOv5模型路径(.pt文件)
        epsilon: 扰动强度 (默认: 0.05)
        alpha: 攻击步长 (默认: 0.01)
        iterations: 迭代次数 (默认: 40)
        input_size: 模型输入尺寸 (默认: 640)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = attempt_load(model_path).to(self.device)
        self.model.eval()  # 注意这里改为eval模式
        
        # 攻击参数
        self.epsilon = epsilon
        self.alpha = alpha
        self.iterations = iterations
        
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
        image: 输入图像路径或PIL.Image对象
        
        返回：
        adversarial_image: 对抗样本PIL.Image对象
        """
        # 处理原始图像
        if isinstance(image, str):
            orig_pil = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):  # 新增numpy数组处理
            orig_pil = Image.fromarray(image.astype('uint8'))
        else:
            orig_pil = image.copy()
        
        # 保存原始尺寸
        orig_size = orig_pil.size
        
        # 添加类型验证
        if not isinstance(orig_pil, Image.Image):
            raise TypeError(f"不支持的输入类型: {type(orig_pil)}，请提供PIL.Image、numpy数组或文件路径")
        
        # 预处理
        img_tensor = self.preprocess(orig_pil).unsqueeze(0).to(self.device)
        img_tensor.requires_grad = True
        
        # 生成对抗样本
        adv_tensor = self._pgd_attack(img_tensor)
        
        # 转换回PIL Image
        adv_pil = self._tensor_to_pil(adv_tensor)
        
        if isinstance(image, np.ndarray):
            return np.array(adv_pil.resize(orig_size, Image.BILINEAR))
        # 还原到原始尺寸
        return adv_pil.resize(orig_size, Image.BILINEAR)

    def _pgd_attack(self, tensor: Tensor) -> Tensor:
        orig_tensor = tensor.clone().detach()
        adv_tensor = tensor.clone().detach()
        
        # 确保模型参数保留梯度
        for param in self.model.parameters():
            param.requires_grad = True  # 新增关键设置
        
        for _ in range(self.iterations):
            adv_tensor = adv_tensor.clone().detach().requires_grad_(True)  # 每次迭代重新绑定梯度
            
            # 前向传播
            outputs = self.model(adv_tensor)
            
            # 处理多尺度输出
            if isinstance(outputs, (tuple, list)):
                pred = outputs[0]  # 取第一个输出层
            else:
                pred = outputs
            
            # 维度验证和损失计算
            if pred.shape[2] >= 5:  # 确保包含置信度信息
                loss = -torch.mean(pred[..., 4])
            else:
                raise RuntimeError(f"预测层格式异常，维度不足: {pred.shape}")
            
            # 梯度清零
            self.model.zero_grad()
            if adv_tensor.grad is not None:
                adv_tensor.grad.data.zero_()
            
            # 反向传播
            loss.backward(retain_graph=True)  # 保留计算图
            
            # 更新对抗样本
            with torch.no_grad():
                perturbation = self.alpha * adv_tensor.grad.sign()
                adv_tensor = adv_tensor + perturbation
                delta = torch.clamp(adv_tensor - orig_tensor, -self.epsilon, self.epsilon)
                adv_tensor = torch.clamp(orig_tensor + delta, 0, 1)
        
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
    attacker = YOLOv5PGDAttacker(
        model_path="yolov5s.pt",
        epsilon=0.03,
        alpha=0.005,
        iterations=50
    )
    
    # 执行攻击并获取结果
    input_image = "test.jpg"  # 支持直接传入PIL.Image对象
    adv_image = attacker.attack(input_image)
    
    # 显示/保存结果
    adv_image.show()
    adv_image.save("adversarial.jpg")
