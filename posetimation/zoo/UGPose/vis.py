# import matplotlib.pyplot as plt
# import numpy as np
#
#
#
# methods = [
#      "PoseWarper", "DCPose", "FAMI-Pose",
#     "TDMI-ST", "DiffPose", "ViTPose", "TPSD-ViT", "DSTA-V",
#     "UAPose-S", "UAPose-B", "UAPose-L", "UAPose-H"
# ]
#
# params = [ 71.1, 65.2, 64.6, 66.2, None, 630.7,  666.8, 631.0,
#           34.1, 124.5, 368.0, 728.4]  # 单位：M
# flops = [192.2, 117.3, 183.9, 198.6, None, 121.1,  585.0, 123.9,
#          33.0, 84.8, 225.8, 433.0]  # 单位：G
# mAP = [ 81.2, 82.8, 84.8, 85.9, 86.4, 83.8,  86.7, 85.6,
#        84.5, 85.8, 86.9, 87.8]  # 单位：%
#
#
# #
# # # 数据准备（新增TPSD-MS/ViT和DSTA-V）
# # methods = [
# #     "SimpleBaseline", "HRNet", "PoseWarper", "DCPose", "FAMI-Pose",
# #     "TDMI-ST", "DiffPose", "ViTPose", "TPSD-ViT", "DSTA-V",
# #     "UAPose-S", "UAPose-B", "UAPose-L", "UAPose-H"
# # ]
# #
# # params = [68.6, 63.6, 71.1, 65.2, 64.6, 66.2, None, 630.7,  666.8, 631.0,
# #           34.1, 124.5, 368.0, 728.4]  # 单位：M
# # flops = [35.6, 32.9, 192.2, 117.3, 183.9, 198.6, None, 121.1,  585.0, 123.9,
# #          33.0, 84.8, 225.8, 433.0]  # 单位：G
# # mAP = [76.7, 77.3, 81.2, 82.8, 84.8, 85.9, 86.4, 83.8,  86.7, 85.6,
# #        84.5, 85.8, 86.9, 87.8]  # 单位：%
#
# # 过滤无参数量的方法
# valid_idx = [i for i, p in enumerate(params) if p is not None]
# methods = [methods[i] for i in valid_idx]
# params = [params[i] for i in valid_idx]
# flops = [flops[i] for i in valid_idx]
# mAP = [mAP[i] for i in valid_idx]
#
# # 颜色设置（UAPose为红色系，其他为不同颜色）
# colors = []
# for name in methods:
#     if "UAPose" in name:
#         if "S" in name: colors.append("#FF0000")
#         elif "B" in name: colors.append("#FF0000")
#         elif "L" in name: colors.append("#FF0000")
#         else: colors.append("#FF0000")
#     elif "TPSD" in name: colors.append("#4B0082")  # 靛青色
#     elif "ViT" in name: colors.append("#9370DB")  # 中紫色
#     elif "DSTA" in name: colors.append("#20B2AA")  # 浅海洋绿
#     else:
#         colors.append(plt.cm.tab20(len(colors) % 20))
#
# # 绘制气泡图（调整figsize和气泡大小系数）
# plt.figure(figsize=(7, 6))
# scatter = plt.scatter(
#     x=params, y=mAP, s=np.array(flops)*3,  # 气泡大小系数从2调整为3
#     c=colors, alpha=0.8, edgecolors="black", linewidth=0.5
# )
#
# # 添加标签和注释（调整字体和偏移）
# for i, (x, y, name) in enumerate(zip(params, mAP, methods)):
#     if "UAPose" in name:
#         if name == 'UAPose-S': offset = (6, -17)
#         elif name == 'UAPose-B': offset = (10, -15)
#         elif name == 'UAPose-L': offset = (10, -15)
#         else: offset = (12, -16)
#     elif "TDMI" in name:
#         offset = (7, 14)
#     elif "FAMI" in name:
#         offset = (12, -4)
#     elif "TPSD" in name:
#         offset = (15, 6)
#     else:
#         offset = (10, 10)
#
#     plt.annotate(
#         name, (x, y), xytext=offset, textcoords='offset points',
#         fontsize=11, bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8)
#     )
#
# # 坐标轴和标题（调整刻度、标签大小和标题）
# plt.xticks(np.arange(0, 801, 200), fontsize=12)  # x轴以200为间隔
# plt.yticks(fontsize=12)
# plt.xlabel("Model Parameters (M)", fontsize=15)
# plt.ylabel("mean Average Precision (%)", fontsize=15)
# # plt.title("3D Human Pose Estimation: Model Size vs. Accuracy",
# #           fontsize=16, fontweight='bold', pad=20)
# plt.grid(True, linestyle='--', alpha=0.4)
#
# # 添加图例（气泡大小示意，调整位置和样式）
# flop_legend = [50, 100, 200, 400]
# for flop in flop_legend:
#     plt.scatter([], [], s=flop*3, c='gray', alpha=0.5, edgecolors='black',
#                 label=f'{flop} GFLOPs')
# legend = plt.legend(title="FLOPs", loc="lower right", framealpha=1)
# legend.get_title().set_fontsize(12)
#
# # 调整坐标范围留出空间
# plt.xlim(0, 900)
# plt.ylim(80, 89)
#
#
#
#
# plt.tight_layout()
# plt.savefig("pose_comparison_v2.png", dpi=300, bbox_inches='tight')
# plt.show()
#
#
#


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

methods = [
    "PoseWarper", "DCPose", "FAMI-Pose",
    "TDMI-ST", "DiffPose", "ViTPose", "TPSD-ViT", "DSTA-V",
    "UAPose-S", "UAPose-B", "UAPose-L", "UAPose-H"
]

params = [71.1, 65.2, 64.6, 66.2, None, 630.7, 666.8, 631.0,
          34.1, 124.5, 368.0, 728.4]  # Unit: M
flops = [192.2, 117.3, 183.9, 198.6, None, 121.1, 585.0, 123.9,
         33.0, 84.8, 225.8, 433.0]  # Unit: G
mAP = [81.2, 82.8, 84.8, 85.9, 86.4, 83.8, 86.7, 85.6,
       84.5, 85.8, 86.9, 87.8]  # Unit: %

# Filter methods without parameter counts
valid_idx = [i for i, p in enumerate(params) if p is not None]
methods = [methods[i] for i in valid_idx]
params = [params[i] for i in valid_idx]
flops = [flops[i] for i in valid_idx]
mAP = [mAP[i] for i in valid_idx]

# Color settings
colors = []
for name in methods:
    if "UAPose" in name:
        colors.append("#FF0000")  # All UAPose variants in red
    elif "TPSD" in name:
        colors.append("#4B0082")  # Indigo
    elif "ViT" in name:
        colors.append("#9370DB")  # Medium purple
    elif "DSTA" in name:
        colors.append("#20B2AA")  # Light sea green
    else:
        colors.append(plt.cm.tab20(len(colors) % 20))

# Create the bubble plot
plt.figure(figsize=(7, 6))
scatter = plt.scatter(
    x=params, y=mAP, s=np.array(flops) * 3,
    c=colors, alpha=0.8, edgecolors="black", linewidth=0.5
)

# Add labels with adjusted positions
for i, (x, y, name) in enumerate(zip(params, mAP, methods)):
    if "UAPose" in name:
        if name == 'UAPose-S':
            offset = (6, -17)
        elif name == 'UAPose-B':
            offset = (10, -15)
        elif name == 'UAPose-L':
            offset = (10, -15)
        else:
            offset = (12, -16)
    elif "TDMI" in name:
        offset = (7, 14)
    elif "FAMI" in name:
        offset = (12, -4)
    elif "TPSD" in name:
        offset = (15, 6)
    else:
        offset = (10, 10)

    plt.annotate(
        name, (x, y), xytext=offset, textcoords='offset points',
        fontsize=11, bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8)
    )

# Axis and title settings
plt.xticks(np.arange(0, 801, 200), fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Model Parameters (M)", fontsize=15)
plt.ylabel("mean Average Precision (%)", fontsize=15)
plt.grid(True, linestyle='--', alpha=0.3)

# Create a mapping between method names and their colors
method_color_map = {name: color for name, color in zip(methods, colors)}

# Create custom legend elements with correct colors
legend_methods = [
    "PoseWarper", "DCPose", "FAMI-Pose", "TDMI-ST",
    "ViTPose", "TPSD-ViT", "DSTA-V", "UAPose (Ours)"
]

legend_elements = []
for method in legend_methods:
    if method == "UAPose (Ours)":
        color = "#FF0000"
        method = r"$\bf{UAPose\ (Ours)}$"
    elif method == "TPSD-ViT":
        color = method_color_map["TPSD-ViT"]
    elif method == "DSTA-V":
        color = method_color_map["DSTA-V"]
    else:
        color = method_color_map[method]

    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', label=method,
               markerfacecolor=color, markersize=12)
    )

# Create legend with 4 columns (for horizontal layout)
legend = plt.legend(handles=legend_elements,
                   loc='lower right',
                   framealpha=1,
                   ncol=2,
                   fontsize=11,
                   handletextpad=0.7,
                   columnspacing=1.0,
                   title='Methods')
legend.get_title().set_fontsize(14)
# Adjust plot limits
plt.xlim(0, 850)
plt.ylim(80, 89)

plt.tight_layout()
plt.savefig("pose_comparison_v2.png", dpi=800, bbox_inches='tight')
plt.show()