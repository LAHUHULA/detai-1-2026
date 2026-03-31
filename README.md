## 1. General Information

- **Title:** DDoS Attack Detection at IoT Gateways Using Federated Learning on Resource-Constrained Devices  
- **Authors:** Le Ngoc Lanh, Chu Ba Thanh  
- **Affiliation:** Faculty of Information Technology, Hungyen University of Technology and Education  
- **Corresponding Author:** Chu Ba Thanh  
- **Contact Email:** thanhcb.dce@gmail.com  

- **Research Domain:** IoT security, distributed intrusion detection, federated learning on edge devices  

- **Research Context:**  
  The rapid expansion of IoT systems has significantly increased the attack surface of modern networks. IoT gateways, which aggregate traffic from multiple devices, are critical points for detecting distributed denial-of-service (DDoS) attacks. However, these gateways typically operate under strict resource constraints, making conventional centralized intrusion detection approaches inefficient and privacy-invasive.

- **Research Objective:**  
  To design and evaluate a lightweight federated learning (FL) framework for DDoS detection at IoT gateways, focusing on detection performance, robustness under non-IID data, and deployment feasibility on resource-constrained edge devices.

- **Dataset:** CICIoT2023  

- **Experimental Platform:**  
  A cluster of 10 Raspberry Pi 4 devices acting as IoT gateways  

- **Evaluated Models:**  
  - LRNet-Lite  
  - MLPNet-Lite  
  - TabResNet-Lite  

- **Aggregation Methods:**  
  - FedAvg  
  - FedProx  

- **Data Distribution Settings:**  
  - IID  
  - non-IID (Dirichlet-based partitioning)  

- **Evaluation Metrics:**  
  - Macro F1-score  
  - Local training time  
  - Round time  
  - CPU usage  
  - RAM usage  
  - Model size  
  - Communication cost  
  - Inference latency  
  - Throughput  

- **Keywords:**  
  IoT security, DDoS detection, federated learning, edge computing, Raspberry Pi 4


## 2. Short Summary

This paper addresses the problem of detecting distributed denial-of-service (DDoS) attacks at IoT gateways, where devices operate under strict resource constraints and centralized data collection raises privacy and communication concerns. To overcome these limitations, the authors propose a lightweight federated learning (FL) framework that enables multiple gateways to collaboratively train intrusion detection models without sharing raw traffic data. The framework is evaluated using the CICIoT2023 dataset on a cluster of ten Raspberry Pi 4 devices, employing three lightweight models (LRNet-Lite, MLPNet-Lite, and TabResNet-Lite) under both IID and non-IID data distributions with FedAvg and FedProx aggregation. Experimental results show that FL achieves F1-scores of approximately 91–92% under IID settings and 88–89% under non-IID settings, while maintaining low inference latency (below 5 ms per sample) on edge devices. These findings demonstrate that federated learning is a practical, privacy-preserving, and efficient solution for distributed DDoS detection in real-world IoT environments.

## 3. Contributions

- This paper proposes a lightweight federated learning framework for gateway-level DDoS detection in IoT environments, specifically designed for deployment on resource-constrained edge devices.

- It evaluates three lightweight models—**LRNet-Lite**, **MLPNet-Lite**, and **TabResNet-Lite**—to compare their detection capability and deployment cost under federated learning settings.

- It investigates the impact of both **IID** and **non-IID** data distributions, showing how data heterogeneity affects model convergence and global detection performance.

- It compares two federated aggregation strategies, **FedAvg** and **FedProx**, to analyze their robustness under heterogeneous client data and partial client participation.

- It studies different client participation rates (**30%**, **50%**, **70%**, and **100%**) in the non-IID setting to assess how partial participation influences convergence behavior and final F1-score.

- It validates the practical feasibility of the proposed framework on a real cluster of **10 Raspberry Pi 4 devices**, rather than relying only on simulation-based evaluation.

- It provides a joint analysis of both **detection performance** and **deployment-oriented metrics**, including CPU usage, RAM usage, local training time, round time, model size, communication cost, inference latency, throughput, and device temperature.

- It demonstrates that federated learning can achieve near-centralized performance under IID conditions and remain effective under non-IID conditions, while preserving data locality and maintaining acceptable edge inference latency.

## 4. Main Results

### 4.1. Centralized vs. Federated Performance under IID Data

Table 1 shows that federated learning achieves performance very close to centralized training under the IID setting. Across all three lightweight models, the difference between centralized learning and federated learning remains below 1 percentage point in macro F1-score. This suggests that when client data are similarly distributed, the federated aggregation process can preserve most of the discriminative power of centralized learning while eliminating the need to share raw traffic data.

**Table 1. F1-score of centralized and federated learning under IID data**

| Model | Setting | Clients | F1-Score |
|---|---|---:|---:|
| LRNet-Lite | Federated | 10 | 91.10 |
| LRNet-Lite | Centralized | - | 92.05 |
| MLPNet-Lite | Federated | 10 | 91.87 |
| MLPNet-Lite | Centralized | - | 92.58 |
| TabResNet-Lite | Federated | 10 | 91.93 |
| TabResNet-Lite | Centralized | - | 92.79 |

**Key observations:**
- **TabResNet-Lite** achieves the highest F1-score in both centralized and federated settings.
- **MLPNet-Lite** performs slightly below TabResNet-Lite but still maintains strong detection capability.
- **LRNet-Lite** has the lowest F1-score among the three models, but remains competitive given its much lower model complexity.
- Overall, federated learning recovers most of the accuracy of centralized learning under IID data.


