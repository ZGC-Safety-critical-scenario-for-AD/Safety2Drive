# import subprocess


# texture_path="/mnt/pxy/perception/Attack/TCEGA.png"

# def set_texture():
#     # 定义命令的各部分
#     command = ["python", "/mnt/pxy/perception/Attack/apply_texture.py", "-l"]
#     grep_filter = "Tesla"
#     # 使用 subprocess.Popen 执行命令并通过管道传输
#     process1 = subprocess.Popen(command, stdout=subprocess.PIPE)
#     process2 = subprocess.Popen(["grep", grep_filter], stdin=process1.stdout, stdout=subprocess.PIPE)
#     # 确保 process1 的输出传递给 process2
#     process1.stdout.close()
#     output = process2.communicate()[0]
#     output_lines = output.decode('utf-8').splitlines()
#     last_line = output_lines[-1]
#     print(output_lines)
#     if last_line:
#         print("获取到的 car_name:", last_line)
        
#         # 定义第二个命令
#         texture_command = [
#             "python",
#             "/mnt/pxy/perception/Attack/apply_texture.py",
#             "-d", texture_path,
#             "-o", last_line  # 将 last_line 作为 car_name 参数
#         ]
#         # 执行第二个命令
#         try:
#             subprocess.run(texture_command, check=True)
#             print("命令执行成功:", " ".join(texture_command))
#         except subprocess.CalledProcessError as e:
#             print("命令执行失败:", e)
#     else:
#         print("未找到任何输出，无法执行第二个命令")

#     print("physical texture ok...")


# # # 打印输出结果（解码为字符串）
# # print(output.decode('utf-8'))
import subprocess
import carla
texture_path = "/mnt/pxy/perception/Attack/TCEGA.png"
def list_all_vehicles():
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    
    # 获取所有车辆对象
    vehicles = world.get_actors().filter('vehicle.*')
    
    # 提取并打印车辆名称
    vehicle_names = [actor.type_id for actor in vehicles]
    print("场景中的车辆名称列表:")
    for idx, name in enumerate(vehicle_names, 1):
        print(f"{idx}. {name}")
def set_texture():
    # 生成车辆列表命令
    command = ["python", "/mnt/pxy/perception/Attack/apply_texture.py", "-l"]
    grep_filter = "Tesla"  # 如果需要所有车型可移除此过滤条件
    
    try:
        # 执行命令获取输出
        process = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # 解码并过滤结果（保持大小写敏感）
        output = process.stdout.decode('utf-8').splitlines()
        vehicles = [line.strip() for line in output if grep_filter in line]
        
        # 去重处理
        unique_vehicles = list(set(vehicles))
        print(f"发现 {len(unique_vehicles)} 辆唯一车辆: {unique_vehicles}")

        # 遍历所有车辆
        for idx, car_name in enumerate(unique_vehicles, 1):
            print(f"\n正在处理第 {idx}/{len(unique_vehicles)} 辆: {car_name}")
            
            texture_command = [
                "python",
                "/mnt/pxy/perception/Attack/apply_texture.py",
                "-d", texture_path,
                "-o", car_name
            ]
            
            try:
                # 添加超时防止卡死（单位：秒）
                subprocess.run(texture_command, check=True, timeout=30)
                print(f"✅ {car_name} 处理成功")
            except subprocess.CalledProcessError as e:
                print(f"❌ {car_name} 处理失败 - 错误码 {e.returncode}")
            except subprocess.TimeoutExpired:
                print(f"⏰ {car_name} 处理超时")

        print("\n所有车辆处理完成")
        
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8').strip()
        print(f"获取车辆列表失败: {error_msg}")
        return

    print("物理纹理处理完成...")

# 执行函数
if __name__ == "__main__":
    list_all_vehicles()
    set_texture()



