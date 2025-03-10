# **Swarm Robotics with GNN-Based MARL**

This repository contains the implementation of a custom Multi-Agent Reinforcement Learning environment integrated with a Graph Neural Network architecture using `torch_geometric`. The project aims to explore the role of GNNs in enhancing cooperation and communication between agents in a swarm robotics system.

---

## **Overview**
Swarm robotics is a multi-agent system where robots collaborate to achieve complex goals. By leveraging GNNs, agents can effectively share information through graph structures, improving their decision-making capabilities.

In this project:  
- A custom MARL environment simulates swarm robotics behavior.  
- The environment includes dynamic agent interactions modeled as a graph.  
- Each agent observes its local environment and updates its policy using a GNN.  
- The performance of the agents is analyzed with visualizations.

## Architecture
![Screenshot 2025-03-10 115508](https://github.com/user-attachments/assets/0d82bad5-777a-47c2-ba39-9e9fc5d5228f)
The Above architecture was used from [this paper](https://arxiv.org/abs/2403.13093?utm_source=chatgpt.com) that introduced MAGEC.
MAGEC was tested in a multi-robot patrolling scenario using a ROS 2-based simulator and demonstrated improved performance compared to existing methods, particularly in scenarios involving agent attrition and communication disturbances.
