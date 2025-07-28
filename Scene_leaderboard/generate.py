import os
from lxml import etree
# 输入和输出文件夹路径
input_folder = "/mnt/cjc/Scenes/Scene_zhibiao/Traffic_Sign"  # 修改为您的输入文件夹路径
output_folder = "/mnt/cjc/Scenes/Scene_leaderboard/Traffic_Sign"  # 修改为您的输出文件夹路径


def remove_specific_conditions(xml_content):
    if isinstance(xml_content, str):
        xml_bytes = xml_content.encode('utf-8')
    else:
        xml_bytes = xml_content
    
    # 解析XML文档
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.fromstring(xml_bytes, parser)
    
    # 定义要移除的条件名称列表
    conditions_to_remove = [
        "criteria_OffRoadTest",
        "criteria_RouteCompletionTest",
        "criteria_InRouteTest"
    ]
    
    # 定位Storyboard下的StopTrigger
    stop_trigger = tree.find(".//Storyboard/StopTrigger")
    
    if stop_trigger is not None:
        # 找到StopTrigger下的ConditionGroup
        condition_group = stop_trigger.find("ConditionGroup")
        
        if condition_group is not None:
            # 从ConditionGroup中移除特定条件节点
            for cond in condition_group.findall("Condition"):
                if cond.get("name") in conditions_to_remove:
                    condition_group.remove(cond)
    
    # 返回美化格式的XML字符串
    return etree.tostring(
        tree, 
        encoding="utf-8",
        pretty_print=True,
        xml_declaration=True
    )
# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    input_file = os.path.join(input_folder, filename)
    
    # 只处理.xosc文件
    if input_file.endswith(".xosc"):
        output_file = os.path.join(output_folder, filename)
        
        try:
            with open(input_file, "rb") as f:
                xml_content = f.read()
            
            # 假设 remove_specific_conditions 是一个已经定义的函数
            modified_xml = remove_specific_conditions(xml_content)
            
            # 保存修改后的文件
            with open(output_file, "wb") as f:
                f.write(modified_xml)
            
            print(f"成功处理文件，输出为: {output_file}")
        
        except Exception as e:
            print(f"处理失败: {e}")
            # 创建错误日志
            with open("error_log.txt", "a") as log:
                log.write(f"处理失败: {input_file} - 错误信息: {str(e)}\n")

