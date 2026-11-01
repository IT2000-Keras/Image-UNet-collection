import os
from unet import Unet

def convert():
    # 创建模型
    unet = Unet()
    
    # 设置导出路径
    save_path = "model_data/unet.onnx"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 转换并保存ONNX模型
    unet.convert_to_onnx(
        simplify=True,    # 是否简化模型
        model_path=save_path  # 保存路径
    )
    
    print(f"模型已保存至: {save_path}")

if __name__ == "__main__":
    convert()