# Safety-Critical-Scenario-Benchmark-for-AD

### 🆕 Updates

- **`2025-01-06:`** The paper is in progress.
- **`2025-01-06:`** We've launched the safety-critical scenario benchmark for autonomous driving!

---

## Table of Contents:

- [Table of Contents:](#table-of-contents)
- Introduction of the Safety2Drive
- [🤩 Running the Autonomous Driving Scenario]
- [🔥 Highlights]
- [🏁 Intelligent Perception Tasks]
  - [1. Object Detection]
  - [2. Traffic Sign Recognition]
  - [3. Traffic light recognition]
  - [4. Lane Line Recognition]
- [🏁 Attack Algorithms]
  - [1. Digital Attacks]
  - [2. Physical Attacks]
- [🏁 Leaderboard of Driving Agents]
  - [1. Autopilot]
  - [2. Garage]
  - [3. Interfuser]
  - [4. UniAD]
- [📌 Roadmap]
- [🔍 Safety-Critical Scenario Generation algorithms (Coming Soon)]
- [Acknowledgments]
- [📝 License]
- [🔖 Citation]

<!-- Introduction -->

## Introduction of the Safety2Drive

- The dataset consists of files in the standard OpenSCENARIO format, including 70 carefully designed standard regulatory scenarios, 30 safety-critical scenarios, and 30 adversarial attack scenarios.

|            Subset            | Number | File List |
| :---------------------------: | :----: | :-------: |
| standard regulatory scenarios |  100  | XOSC File |
|   safety-critical scenarios   |   30   | XOSC File |
| adversarial attack scenarios |   30   | XOSC File |

Note that the documentation contains 15 representative scenarios. You can contact us via email to get the full scenario file.

<!-- Introduction -->

## 🤩 Running the Autonomous Driving Scenario

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

| Decelerating | CutIn | CutOutFront | PedestrianCrossing | TwoWheelerRiding |
| :----------: | :---: | :---------: | :----------------: | :--------------: |
| ![gif]() |      |            |                    |                  |
|              |      |            |                    |                  |

### Here are 4 types of environmental disturbances: fog, rain, night, exposure.

---

|                                           SunnyDay                                           |                                           RainyDay                                           |                                        NightTime                                        |                                   FoggyDay                                   |
| :-------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------: |
| ![SunnyDay_128x128](https://jc2001-1307981922.cos.ap-beijing.myqcloud.com/SunnyDay_128x128.gif) | ![RainyDay_128x128](https://jc2001-1307981922.cos.ap-beijing.myqcloud.com/RainyDay_128x128.gif) | ![Night_128x128](https://jc2001-1307981922.cos.ap-beijing.myqcloud.com/Night_128x128.gif) | ![](https://jc2001-1307981922.cos.ap-beijing.myqcloud.com/FoggyDay_128x128.gif) |

### Here are 5 safety-critical scenarios

---

|                                                   SuddenPedestrianCrossing                                                   |                                                MaliciousCuttingIn                                                |                                               RunningRedLight                                               |
| :---------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------: |
| ![SuddenPedestrianCrossing_128x128](https://jc2001-1307981922.cos.ap-beijing.myqcloud.com/SuddenPedestrianCrossing_128x128.gif) | ![MaliciousCuttingIn_128x128](https://jc2001-1307981922.cos.ap-beijing.myqcloud.com/MaliciousCuttingIn_128x128.gif) | ![RunningRedLight_128x128](https://jc2001-1307981922.cos.ap-beijing.myqcloud.com/RunningRedLight_128x128.gif) |

## 🏁 Intelligent Perception Tasks

|                                        Traffic Light                                        |                                          Lane line                                          |
| :------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------: |
| ![Traffic Light](https://ucd-dare.github.io/cardreamer.github.io/static/gifs/bev/tl_right.gif) | ![Stop Sign](https://ucd-dare.github.io/cardreamer.github.io/static/gifs/bev/stop%20sign.gif) |

## 🏁 Adversarial attack Tasks

### 1.Pixel-based Digital attack

|                                                                                Without Adversarial attack vs. With Adversarial attack                                                                                |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                                               **Right turn hard**                                                                                               |
| ![Right turn hard](https://ucd-dare.github.io/cardreamer.github.io/static/gifs/bev/right_turn_hard.gif)     ![Right turn hard FOV](https://ucd-dare.github.io/cardreamer.github.io/static/gifs/bev/right_turn_fov.gif) |

### 2.Patch-based Digital attack

|                                                                                Without Adversarial attack vs. With Adversarial attack                                                                                |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                                               **Right turn hard**                                                                                               |
| ![Right turn hard](https://ucd-dare.github.io/cardreamer.github.io/static/gifs/bev/right_turn_hard.gif)     ![Right turn hard FOV](https://ucd-dare.github.io/cardreamer.github.io/static/gifs/bev/right_turn_fov.gif) |

### 3.Camouflage-based physical attack

|                                                                                Without Adversarial attack vs. With Adversarial attack                                                                                |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                                               **Right turn hard**                                                                                               |
| ![Right turn hard](https://ucd-dare.github.io/cardreamer.github.io/static/gifs/bev/right_turn_hard.gif)     ![Right turn hard FOV](https://ucd-dare.github.io/cardreamer.github.io/static/gifs/bev/right_turn_fov.gif) |

### 4.Backdoor attack

|                                                                                Without Adversarial attack vs. With Adversarial attack                                                                                |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                                               **Right turn hard**                                                                                               |
| ![Right turn hard](https://ucd-dare.github.io/cardreamer.github.io/static/gifs/bev/right_turn_hard.gif)     ![Right turn hard FOV](https://ucd-dare.github.io/cardreamer.github.io/static/gifs/bev/right_turn_fov.gif) |

## 🏁 Leaderboard of Driving Agents

### Closed-loop Evaluation Leaderboard

#### UniAD

<table>
    <tr style="background-color: #C7C7C7; color: white;">
        <th>Driving Agent</th>
        <th>Scenario 1</th>
        <th>PDMS</th>
        <th>RC</th>
        <th>ADS</th>
    </tr>
    <tr>
        <td>UniAD</td>
        <td>Scenario 2</td>
        <td>0.7615</td>
        <td>0.1684</td>
        <td>0.1684</td>
    </tr>
    <tr>
        <td>UniAD</td>
        <td>Scenario 3</td>
        <td>0.7215</td>
        <td>0.169</td>
        <td>0.0875</td>
    </tr>
    <tr>
        <td>UniAD</td>
        <td>Scenario 4</td>
        <td>0.4952</td>
        <td>0.091</td>
        <td>0.0450</td>
    </tr>
    <tr>
        <td>UniAD</td>
        <td>Scenario 5</td>
        <td>0.6888</td>
        <td>0.121</td>
        <td>0.0835</td>
    </tr>
</table>
<!-- ROADMAP -->
## 📌 Roadmap

- [X] Demo Website Release
- [X] V1.0 Release
  - [X] Benchmark
  - [X] Perception Task
  - [X] Driving Agent Support

- []  V1.1 Release
  - [] Safety-Critical Scenario Generation algorithms
- []  V1.2 Release
  - [] LLM-based Scenario Generation algorithms
