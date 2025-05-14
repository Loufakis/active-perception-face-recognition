# Thesis Repository: Active Perception in Face Recognition using Navigation in Simulated Environments

This repository contains all the code and resources used for my Master's thesis at Aristotle University of Thessaloniki, School of Informatics, under the supervision of Prof. Anastasios Tefas. The thesis focuses on enhancing face recognition in robotics through active perception with multi-task learning, using simulations in the Webots environment.

## Overview

- **Thesis Title:** Active Perception in Face Recognition using Navigation in Simulated Environments
- **Author:** Michail Loufakis
- **Supervisor:** Prof. Anastasios Tefas
- **Institution:** Aristotle University of Thessaloniki
- **Year:** 2023

## Abstract
This project explores how robotic agents can improve face recognition performance through active perception—by intelligently navigating to acquire more informative viewpoints. A Webots-based simulation is developed, where a robot learns to move within an environment to optimize face recognition. The system combines navigation and perception using multi-task learning, demonstrating that task-aware movement leads to higher recognition accuracy compared to static observation.

## Repository Structure
- `Code/`: Python scripts used for training and evaluation of the recognition and navigation models.
- `Webots_Files/`: Webots simulation worlds, robot controller scripts, and configuration files.
- `figures/`: Diagrams and plots used in the thesis report.
- `Thesis.pdf`: The complete Master's thesis document detailing the research, methodology, experiments, and conclusions.


## Research Summary
- **Simulation:** Conducted in Webots, simulating a mobile robot navigating a 3D environment with face targets placed at various locations.
- **Recognition Model:** Uses CNN-based architecture to classify identities from RGB input frames.
- **Active Perception:** Trained reinforcement learning agent learns to reposition the robot based on current camera view quality to improve recognition outcomes.
- **Multi-task Learning:** Joint training of navigation and face recognition promotes efficient learning and cross-task benefit.
- **Evaluation:** Results show improved recognition performance when the robot actively navigates vs. remaining static.



## Usage
### Requirements

- Python 3.7+
- Webots (tested with R2022b)
- Dependencies:
  ```bash
  pip install numpy opencv-python matplotlib tensorflow scikit-lear
  ```

###  Running the Simulation
1. Open Webots and load the .wbt world file from Webots_Files/.
2. Start the simulation.
3. The robot will explore the environment, attempting to optimize its position for facial recognition.

### Training Scripts
If using Code/ for training:
```bash
cd Code
python train_face_recognition.py
python train_navigation_agent.py
```

## Citation

If you find this work useful, please cite:

Loufakis, M. (2023). Active Perception in Face Recognition using Navigation in Simulated Environments. Master's Thesis, Aristotle University of Thessaloniki, School of Informatics.

## Acknowledgments
Special thanks to Prof. Anastasios Tefas for his guidance throughout this project, and to the Aristotle University of Thessaloniki for supporting this research.



