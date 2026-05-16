Overview

DUDE (Dynamic UAV Deployment using dEep learning) is a deep learning-based framework for predicting the optimal 3D deployment location of a UAV in heterogeneous hotspot environments.

The framework addresses dynamic wireless coverage scenarios where:

hotspot sizes vary over time,
users follow heterogeneous spatial distributions,
users have different QoS/data-rate requirements,
real-time deployment decisions are required.

Instead of relying on computationally expensive iterative optimization during deployment, DUDE performs extensive offline learning and predicts near-optimal UAV positions in real time using a hybrid CNN architecture.

Key Features
Dynamic hotspot-aware UAV deployment
QoS-aware user connectivity modeling
Heterogeneous user distribution support
Hybrid CNN architecture based on ResNet
Real-time inference for UAV positioning
Synthetic dataset generation pipeline
Communication and energy-aware deployment model
Problem Statement

Given:

a hotspot region,
spatial distribution of ground terminals (GTs),
user QoS/data-rate requirements,

predict the optimal UAV:

horizontal location (x, y)
altitude (h)

such that:

path loss is minimized,
user connectivity is maximized,
QoS constraints are satisfied.
System Model

The framework includes:

Air-to-ground communication modeling
LoS/NLoS probabilistic path loss
UAV energy consumption model
QoS-aware bandwidth allocation
Dynamic hotspot generation
Repository Structure
DUDE/
│
├── dataset/
│   ├── generate_dataset.py
│   ├── distributions.py
│   └── utils.py
│
├── models/
│   ├── hybrid_resnet.py
│   ├── cnn_backbone.py
│   └── layers.py
│
├── training/
│   ├── train.py
│   ├── validate.py
│   └── losses.py
│
├── evaluation/
│   ├── evaluate.py
│   ├── metrics.py
│   └── visualization.py
│
├── experiments/
│   ├── configs/
│   └── saved_models/
│
├── figures/
│
├── requirements.txt
├── README.md
└── LICENSE
Dataset Generation

The dataset consists of:

dynamically generated hotspot regions,
heterogeneous GT distributions,
GT data-rate requirements,
brute-force generated optimal UAV labels.

Supported GT distributions:

Uniform
Positive skewed
Negative skewed
Center-biased
Corner-biased
Edge-biased

The optimal UAV location is generated using exhaustive 3D search under QoS constraints.

Hybrid CNN Architecture

The model uses:

ResNet-based CNN backbone
Spatial GT density maps as input
Additional scalar hotspot-size input
Fully connected regression head

The model predicts:

(x_position, y_position, altitude)
Installation
Clone Repository
git clone https://github.com/<username>/DUDE.git
cd DUDE
Create Environment
conda create -n dude python=3.10
conda activate dude
Install Dependencies
pip install -r requirements.txt
Training
Generate Dataset
python dataset/generate_dataset.py
Train Model
python training/train.py
Evaluation
python evaluation/evaluate.py

Metrics:

MAE
RMSE
R² Score
Connectivity Ratio
Example Results

The model achieves:

Mean Absolute Error (MAE): ~3.5
R² Score: >96%

across:

varying hotspot sizes,
heterogeneous user distributions,
different QoS requirements.
Simulation Parameters
Parameter	Value
Carrier Frequency	2 GHz
Pathloss Threshold	100 dB
UAV Bandwidth	20 MHz
CNN Grid Sizes	128, 256, 512
Optimizer	Adam
Loss Function	MSE
Epochs	40
Visualization

The framework supports:

GT spatial visualization
Predicted vs true UAV placement
Coverage region visualization
Distribution heatmaps
Citation

If you use this work, please cite:

@article{rajashekar2026dude,
  title={A Deep Learning-Based Approach for Heterogeneous Hotspot-Coverage in UAV Deployment},
  author={Rajashekar, Kolichala and Sunkara, Vamsi Krishna and Sidhanta, Subhajit},
  journal={Ad Hoc Networks},
  volume={183},
  pages={104127},
  year={2026},
  publisher={Elsevier}
}
Future Work
Multi-UAV deployment
Real-world mobility traces
Online adaptive learning
Federated UAV coordination
Wind-aware energy models
Edge-device optimization
License

This project is released under the MIT License.

Contact

Kolichala Rajashekar
University of Innsbruck, Austria
📧 rajashekar.kolichala@uibk.ac.at

Acknowledgement

This work was supported by research collaborations between:

University of Innsbruck
IIT Bhilai
IIT Kharagpur
