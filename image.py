from PIL import Image
import numpy as np

def replace_colors(image_path, output_path):
    # 定义所有颜色替换的配置
    color_replacements = [
        {
            "target_colors": [(214,221,193),(167,181,123),(102,153,0),(204,255,204)],
            "merged_color": [0,128,0]
        },
        {
            "target_colors": [(226,221,247),(204,153,0)],
            "merged_color": [50,205,50]
        },
        {
            "target_colors": [(255, 192, 0), (204, 204, 0), (248, 203, 196)],
            "merged_color": [255,215,0]
        },
        {
            "target_colors": [(168,147,136),(191,199,211)],
            "merged_color": [100,149,237]
        },
        {
            "target_colors": [(139, 205, 255), (97, 221, 218), (0, 153, 153)],
            "merged_color": [0, 30, 100]
        },
        {
            "target_colors": [(204,102,0)],
            "merged_color": [0,0,0]
        }
    ]

    # 打开图像并转换为NumPy数组
    image = Image.open(image_path)
    image_array = np.array(image)

    # 遍历所有颜色替换配置
    for replacement in color_replacements:
        target_colors = replacement["target_colors"]
        merged_color = replacement["merged_color"]

        # 替换每组 target_colors
        for color in target_colors:
            mask = np.all(image_array == color, axis=-1)
            image_array[mask] = merged_color

    # 保存最终结果
    replaced_image = Image.fromarray(image_array)
    replaced_image.save(output_path)
    print(f"所有颜色替换完成，最终图像保存为 {output_path}")

# 调用函数
replace_colors('datasets/XLCS_mydata/BandS_label_crop.png', 'datasets/XLCS_mydata/BandS_label.png')

# import cv2
# import matplotlib.pyplot as plt
# import numpy as np

# from PIL import Image
# import numpy as np
# from PIL import Image


# # ---------将label合并成五类----------------
# target_colors =  [(214,221,193),(167,181,123),(102,153,0),(204,255,204)]  # BGR颜色范围
# merged_color = [0,128,0]
# # target_colors =  [(226,221,247),(204,153,0)]  # BGR颜色范围
# # merged_color = [50,205,50]
# # target_colors =  [(255, 192, 0), (204, 204, 0), (248, 203, 196)]  # BGR颜色范围
# # merged_color = [255,215,0]
# # target_colors =  [(168,147,136),(191,199,211)]  # BGR颜色范围
# # merged_color = [100,149,237]
# # target_colors =  [(139, 205, 255), (97, 221, 218), (0, 153, 153)]  # BGR颜色范围
# # merged_color = [0, 30, 100]
# # target_colors =  [(204,102,0)]  # BGR颜色范围
# # merged_color = [0,0,0]
# def replace_color(image_path, target_colors,  merged_color):
#     image = Image.open(image_path)
#     image_array = np.array(image)    # 将图像转换为NumPy数组
#     color_to_replace = target_colors    # 定义要替换的颜色值列表
#     for color in color_to_replace:    # 将特定颜色值替换为目标颜色值
#         mask = np.all(image_array == color, axis=-1)
#         image_array[mask] = merged_color
#     replaced_image = Image.fromarray(image_array)    # 将NumPy数组转换回图像
#     replaced_image.save("BandC_label.png")
#     print('保存成功')
# replace_color('BandC_label_merge_5.png', target_colors, merged_color)
# # replace_color('BandX_label_merge_color_final.png', target_colors, merged_color)
# print('end')

# # image = cv2.imread('MMpolsar/BandX_label_crop.png')  # coarse分类
# # (139, 205, 255), (97, 221, 218), (0, 153, 153)--[0, 30, 100] 表面散射1水体；
# # (168,147,136),(191,199,211)--[100,149,237]表面散射2道路农田，普通蓝
# # (255, 192, 0), (204, 204, 0), (248, 203, 196)--[255,215,0]---二次散射房屋建筑
# # (226,221,247),(204,153,0)--[50,205,50],体散射1没植被；
# # (214,221,193),(167,181,123),(102,153,0),(204,255,204)---[0,128,0]体散射2有植被