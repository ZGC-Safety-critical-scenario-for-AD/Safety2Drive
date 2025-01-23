# Safety-Critical-Scenario-Benchmark-for-AD

### 🆕 Updates

- **`2025-01-06:`** We've launched the safety-critical scenario benchmark for autonomous driving! 🚗

---

## Table of Contents:

- [Table of Contents:](#table-of-contents)
- [🌍Introduction of the Safety2Drive](#-Introduction-of-the-Safety2Drive)
- [🍃 Running the Autonomous Driving Scenario](#-Running-the-Autonomous-Driving-Scenario)
- [🔧 Intelligent Perception Tasks](#-Intelligent-Perception-Tasks)
  - 1. Camera-based Object Recognition
  - 2. Lane Line Recognition
  - 3. Depth Estimation
  - 4. Lidar-based Object Recognition
- [⚡Adversarial Attack Scenarios](#-Adversarial-Attack-Scenarios)
  - 1. Pixel-based Digital Attacks
  - 2. Patch-based Digital/Physical Attacks
  - 3. Camouflage-based Physical Attacks
  - 4. Backdoor Attacks
- [🏁 Leaderboard of Driving Agents](#-Leaderboard-of-Driving-Agents)
  - 1. Autopilot
  - 2. Garage
  - 3. Interfuser
- [📌 Roadmap](#-Roadmap)
- [🔍 Safety-Critical Scenario Generation algorithms (Stay Tuned)](#-Safety-Critical-Scenario-Generation-algorithms)
- [Acknowledgments]
- [📝 License]
- [🔖 Citation]

<!-- Introduction -->

## 🌍 Introduction of the Safety2Drive

- The benchmark are in the standard OpenSCENARIO format, including 100 carefully designed standard regulatory scenarios for functional testing, 30 safety-critical scenarios, and 30 adversarial attack scenarios. Each of these 100 functional test items can be generalized to multiple scenarios. Theoretically, the benchmark contains an infinite number of scenarios.

|            Subset            | Number | File List |
| :---------------------------: | :----: | :-------: |
| Functional Test |  70  | xosc file |
|   safety-critical scenarios   |   30   | xosc file |
| adversarial attack scenarios |   30   | xosc file |

Note that the file in Repo contains 15 representative scenarios. You can contact us via email to get the full scenario file.

<!-- Introduction -->

## 🍃 Running the Autonomous Driving Scenario

### Download and setup CARLA

- CARLA 0.9.15
  ```bash
      mkdir carla
      cd carla
      wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
      tar -xvf CARLA_0.9.15.tar.gz
      cd Import && wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz
      cd .. && bash ImportAssets.sh
      export CARLA_ROOT=YOUR_CARLA_PATH
      echo "$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg" >> YOUR_CONDA_PATH/envs/YOUR_CONDA_ENV_NAME/lib/python3.7/site-packages/carla.pth # python 3.8 also works well, please set YOUR_CONDA_PATH and YOUR_CONDA_ENV_NAME
  ```

### Use the ScenarioRunner

---

Please take a look at the [Getting started](scenario_ruuner/Docs/getting_scenariorunner.md)
documentation.

### Here are 5 driving scenarios

---

| Decelerating | Cut In | Cut Out Front | Pedestrian Crossing | Two Wheeler Riding |
| :----------: | :---: | :---------: | :----------------: | :--------------: |
| <img src="https://github.com/ZGC-Safety-critical-scenario-for-AD/Safety2Drive/blob/main/gif_files/Decelerating/Decelerating.gif" width="128" height="128"> | <img src="https://github.com/ZGC-Safety-critical-scenario-for-AD/Safety2Drive/blob/main/gif_files/CutIn/CutIn.gif" width="128" height="128"> | <img src="https://github.com/ZGC-Safety-critical-scenario-for-AD/Safety2Drive/blob/main/gif_files/CutOutFront/CutOutFront.gif" width="128" height="128"> | <img src="https://github.com/ZGC-Safety-critical-scenario-for-AD/Safety2Drive/blob/main/gif_files/PedestrianCrossing/PedestrianCrossing.gif" width="128" height="128"> | <img src="https://github.com/ZGC-Safety-critical-scenario-for-AD/Safety2Drive/blob/main/gif_files/TwoWheelerRiding/TwoWheelerRiding.gif" width="128" height="128"> |
| BEV | BEV | BEV | BEV | BEV |
| <img src="https://github.com/ZGC-Safety-critical-scenario-for-AD/Safety2Drive/blob/main/gif_files/Decelerating/Decelerating_Bev.gif" width="128" height="128"> | <img src="https://github.com/ZGC-Safety-critical-scenario-for-AD/Safety2Drive/blob/main/gif_files/CutIn/CutIn_Bev.gif" width="128" height="128"> | <img src="https://github.com/ZGC-Safety-critical-scenario-for-AD/Safety2Drive/blob/main/gif_files/CutOutFront/CutOutFront_Bev.gif" width="128" height="128"> | <img src="https://github.com/ZGC-Safety-critical-scenario-for-AD/Safety2Drive/blob/main/gif_files/PedestrianCrossing/PedestrianCrossing_Bev.gif" width="128" height="128"> | <img src="https://github.com/ZGC-Safety-critical-scenario-for-AD/Safety2Drive/blob/main/gif_files/TwoWheelerRiding/TwoWheelerRiding_Bev.gif" width="128" height="128"> |

### 16 types of environmental and weather: Fog, Rain, Midnight，Nightfall，Afternoon，Midday，Morning，Cloudless，Thick fog，Haze，Road surface water，Drizzle，Moderate rain，Heavy rain，Cloudy，Sunny.

---

|                           Sunny                          |                           Rain                           |                          midnight                           |                           Fog                           |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="https://github.com/ZGC-Safety-critical-scenario-for-AD/Safety2Drive/blob/main/gif_files/Environmental_interference/SunnyDay.gif" width="128" height="128"> | <img src="https://github.com/ZGC-Safety-critical-scenario-for-AD/Safety2Drive/blob/main/gif_files/Environmental_interference/RainyDay.gif" width="128" height="128"> | <img src="https://github.com/ZGC-Safety-critical-scenario-for-AD/Safety2Drive/blob/main/gif_files/Environmental_interference/NightTime.gif" width="128" height="128"> | <img src="https://github.com/ZGC-Safety-critical-scenario-for-AD/Safety2Drive/blob/main/gif_files/Environmental_interference/FoggyDay.gif" width="128" height="128"> |

### Here are 5 safety-critical scenarios

---

|                   Sudden Pedestrian Crossing                   |                       Opposing Lane Pass                       |                      Lane Change With Cone                      |                     Two wheeler Retrograde                     |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="https://github.com/ZGC-Safety-critical-scenario-for-AD/Safety2Drive/blob/main/gif_files/Riskmigration/SuddenPedestrianCrossing.gif" width="128" height="128"> | <img src="https://github.com/ZGC-Safety-critical-scenario-for-AD/Safety2Drive/blob/main/gif_files/Riskmigration/OpposingLanePass.gif" width="128" height="128"> | <img src="https://github.com/ZGC-Safety-critical-scenario-for-AD/Safety2Drive/blob/main/gif_files/Riskmigration/LaneChangeWithCone.gif" width="128" height="128"> | <img src="https://github.com/ZGC-Safety-critical-scenario-for-AD/Safety2Drive/blob/main/gif_files/Riskmigration/TwowheelerRetrograde.gif" width="128" height="128"> |

## 🏁 Intelligent Perception Tasks

|                                      Camera-based  Object Recognition                                        |                                          Lane Line Recognition                                          |
| :------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------: |
| ![Traffic Light](https://ucd-dare.github.io/cardreamer.github.io/static/gifs/bev/tl_right.gif) | ![Stop Sign](https://ucd-dare.github.io/cardreamer.github.io/static/gifs/bev/stop%20sign.gif) |

| Depth Estimation | Lidar-based Object Recognition|
| :-----------: | :-------: |
| ![Traffic Light](https://ucd-dare.github.io/cardreamer.github.io/static/gifs/bev/tl_right.gif) | ![Stop Sign](https://ucd-dare.github.io/cardreamer.github.io/static/gifs/bev/stop%20sign.gif) |

## 🏁 Adversarial Attack Scenarios
### 1.Pixel-based Digital Attack

|                                                                                Without Digital Attack vs. With Digital Attack                                                                                |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                                               **Without Digital Attack**                                                                                               |
| ![Right turn hard FOV](https://github.com/ZGC-Safety-critical-scenario-for-AD/Safety2Drive/blob/main/gif_files/Adversarial_Attack/before_digital_attack.jpg)|
**PGD Digital Attack** 
|![Right turn hard FOV](https://github.com/ZGC-Safety-critical-scenario-for-AD/Safety2Drive/blob/main/gif_files/Adversarial_Attack/after_digital_attack.jpg) |

### 2.Patch-based Digital/Physical Attack

|                                                                                Without Digital Attack vs. With Digital Attack                                                                                |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://github.com/ZGC-Safety-critical-scenario-for-AD/Safety2Drive/blob/main/gif_files/Adversarial_Attack/before_patch.png" width="256" height="256">  <img src="https://github.com/ZGC-Safety-critical-scenario-for-AD/Safety2Drive/blob/main/gif_files/Adversarial_Attack/after_patch.jpg" width="256" height="256">|

|                                                                                Without Lane Line Attack vs. With Lane Line Attack                                                                                |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                                               **Scenario: Right turn hard(选个场景)**                                                                                               |
| ![Right turn hard](https://ucd-dare.github.io/cardreamer.github.io/static/gifs/bev/right_turn_hard.gif)     ![Right turn hard FOV](https://ucd-dare.github.io/cardreamer.github.io/static/gifs/bev/right_turn_fov.gif) |
### 3.Camouflage-based Physical Attack

|                                                                                Without Camouflage Attack vs. With Camouflage Attack                                                                                |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://github.com/ZGC-Safety-critical-scenario-for-AD/Safety2Drive/blob/main/gif_files/Adversarial_Attack/before_p_attack.png" width="256" height="256">  <img src="https://github.com/ZGC-Safety-critical-scenario-for-AD/Safety2Drive/blob/main/gif_files/Adversarial_Attack/after_p_attack.png" width="256" height="256"> |

### 4.Backdoor Attack

|                                                                                Backdoor Attack                                                                                |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://github.com/ZGC-Safety-critical-scenario-for-AD/Safety2Drive/blob/main/gif_files/Adversarial_Attack/backdoor.png" width="256" height="256"> |
## 🏁 Leaderboard of Driving Agents

### Closed-loop Evaluation Leaderboard
#### Build your agent
  - Add your agent to leaderboard/team_code/your_agent.py & Link your model folder under the Safety2Drive directory.
    ```bash
        Safety2Drive\ 
          assets\
          xosc_files\
          leaderboard\
            team_code\
              --> Please add your agent HEAR
          scenario_runner\
          tools\
          --> Please link your model folder HEAR
    ```
#### Autopilot
|                                      Lane Change With Cone                                          |                                          Sudden Pedestrian Crossing                                          |
| :------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------: |
| ![Traffic Light](https://github.com/ZGC-Safety-critical-scenario-for-AD/Safety2Drive/blob/main/gif_files/Agent/auto_cone.gif) | ![Stop Sign](https://github.com/ZGC-Safety-critical-scenario-for-AD/Safety2Drive/blob/main/gif_files/Agent/auto_ghost.gif) |


<table>
    <tr style="background-color: #C7C7C7; color: white;">
        <th>Driving Agent</th>
        <th>Scenarios</th>
        <th>Collision</th>
        <th>Complete Route</th>
        <th>Driving Score</th>
    </tr>
    <tr>
        <td>AutoPolit</td>
        <td>Cone</td>
        <td>True</td>
        <td>False</td>
        <td>0.2746</td>
    </tr>
    <tr>
        <td>AutoPolit</td>
        <td>Ghost</td>
        <td>False</td>
        <td>True</td>
        <td>1.0</td>
    </tr>
</table>

#### Transfuser
|                                      Lane Change With Cone                                          |                                          Sudden Pedestrian Crossing                                          |
| :------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------: |
| ![Traffic Light](https://github.com/ZGC-Safety-critical-scenario-for-AD/Safety2Drive/blob/main/gif_files/Agent/transfuser_cone.gif) | ![Stop Sign](https://github.com/ZGC-Safety-critical-scenario-for-AD/Safety2Drive/blob/main/gif_files/Agent/transfuser_ghost.gif) |


<table>
    <tr style="background-color: #C7C7C7; color: white;">
        <th>Driving Agent</th>
        <th>Scenarios</th>
        <th>Collision</th>
        <th>Complete Route</th>
        <th>Driving Score</th>
    </tr>
    <tr>
        <td>Transfuser</td>
        <td>Cone</td>
        <td>True</td>
        <td>False</td>
        <td>0.1785</td>
    </tr>
    <tr>
        <td>Transfuser</td>
        <td>Ghost</td>
        <td>False</td>
        <td>True</td>
        <td>1.0</td>
    </tr>
</table>

#### Garage
|                                      Lane Change With Cone                                           |                                           Sudden Pedestrian Crossing                                          |
| :------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------: |
| ![Traffic Light](https://github.com/ZGC-Safety-critical-scenario-for-AD/Safety2Drive/blob/main/gif_files/Agent/garage_cone.gif) | ![Stop Sign](https://github.com/ZGC-Safety-critical-scenario-for-AD/Safety2Drive/blob/main/gif_files/Agent/garage_ghost.gif) |


<table>
    <tr style="background-color: #C7C7C7; color: white;">
        <th>Driving Agent</th>
        <th>Scenarios</th>
        <th>Collision</th>
        <th>Complete Route</th>
        <th>Driving Score</th>
    </tr>
    <tr>
        <td>Garage</td>
        <td>Cone</td>
        <td>True</td>
        <td>False</td>
        <td>0.4225</td>
    </tr>
    <tr>
        <td>Garage</td>
        <td>Ghost</td>
        <td>False</td>
        <td>True</td>
        <td>1.0</td>
    </tr>
</table>
<!-- ROADMAP -->
## 📌 Roadmap

- [X] Demo Website Release
- [X] V1.0 Release
  - [X] Benchmark
  - [X] Perception Task
  - [X] Driving Agent Support

- [ ]  V1.1 Release
  - [ ] Safety-Critical Scenario Generation algorithms
- [ ]  V1.2 Release
  - [ ] LLM-based Scenario Generation algorithms
