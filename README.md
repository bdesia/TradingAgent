### DDQN Trading Agent

**Overview**

The first part of this project implements a Double Deep Q-Learning (DDQN) agent for stock trading, developed for the Reinforcement Learning 1 course of the AI Specialization at FIUBA (2025). It uses historical AAPL stock data (2015–2025) to train an agent to make buy, sell, or hold decisions in a custom Gymnasium environment, incorporating transaction costs and technical indicators (EMA, RSI). The goal is to maximize portfolio value in a simulated market.


### Ensemble Trading Agent

**Overview**

The second one, presented as final assignment of the Reinforcement Learning 2 course of the AI Specialization at FIUBA (2025), continues working with the same objetive of applying reforcement learning in stock trading. Several agents such as PPO, A2C, DDPG, SAC and TD3 are trained using **Stable baselines 3** library and taken as base to build an ensemble model.