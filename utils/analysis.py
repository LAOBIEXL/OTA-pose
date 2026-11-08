import torch
from fvcore.nn import FlopCountAnalysis , flop_count_str , parameter_count
from torchvision.models import VisionTransformer

# 创建一个Vision Transformer模型实例
model = VisionTransformer()

# 准备一个输入张量，假设输入图像的大小为224x224
input_tensor = torch.randn(1, 3, 224, 224)

# 使用FlopCountAnalysis分析模型
flop_analyzer = FlopCountAnalysis(model, (input_tensor,))
model.eval()
with torch.no_grad():
    flop_analyzer.accumulate()
    flops = flop_analyzer.total()

# 打印浮点运算次数
print(f"Total FLOPs: {flops}")

# 使用flop_count_str函数获取易读的字符串表示
print(f"FLOPs string: {flop_count_str(flops)}")

# 计算并打印模型的参数数量
print(f"Total parameters: {parameter_count(model)}")