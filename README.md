# Stock Trading with A3C Reinforcement Learning

This project demonstrates how to use an Asynchronous Advantage Actor-Critic (A3C) reinforcement learning model to perform stock trading. The model is trained to make buy, sell, or hold decisions based on historical stock price data and various technical indicators.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Setup](#setup)

## Introduction
Automated stock trading is a complex and challenging task that requires careful consideration of market dynamics, price movements, and various economic indicators. The A3C model is a reinforcement learning algorithm that uses multiple agents working in parallel to learn policies for maximizing the expected reward.

This project utilizes the A3C model to make trading decisions based on historical stock data. The model observes stock prices, cash balance, and technical indicators like Simple Moving Averages (SMA) and Volume Moving Averages (VMA) to decide whether to buy, sell, or hold a stock.

## Features
- **Reinforcement Learning**: Utilizes A3C algorithm for training the model.
- **Technical Indicators**: Includes SMA and VMA as part of the observation space.
- **Real-time Trading Simulation**: Simulates real-time trading based on model predictions.
- **Performance Visualization**: Plots account value and stock price over time, including buy/sell actions.

## Setup
To set up the environment, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/ICTRLT.git
   cd ICTRLT

2. **Install dependencies**:
```bash
 pip install -r requirements.txt
```

3. **Google Drive Integration**: <br>
This project uses Google Colab for training. Ensure you have Google Drive mounted in your Colab environment.
