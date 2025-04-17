Agent_test文件夹是指提供ego vehicle路径且有初速度的xosc场景文件

Regulation_and_control是可以接入leaderboard 评价的xosc场景文件

Environmental_disturbance 文件夹为30个涉及环境干扰场景的xosc场景文件

Risk_migration文件夹为安全威胁场景的xosc场景文件

Standards_and_regulations文件夹为70个标准法规场景的xosc场景文件

运行脚本：

```shell
 ./scenario_runner.py --port 3000 --trafficManagerPort 9000 --openscenario ./scene/Standards_and_regulations/Cruising_straight.xosc --output
```

