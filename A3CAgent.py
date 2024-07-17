import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import pandas as pd
from env.env import StockTradingEnv
import sys
import os
import random

try:
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    from utils import *
except ImportError:    
    from utils import *

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_space):
        """
        Actor-Critic 모델 초기화

        Args:
            input_dim (int): 입력 차원
            action_space (int): 행동 공간의 크기

        [입력층]                [은닉층]                  [출력층]
        (input_dim)  ->   (128개의 은닉 노드)  ->  (policy: action_space) 
                                              ->  (value: 1)
        """
        super(ActorCritic, self).__init__()
        log_manager.logger.info("Initializing ActorCritic model")
        self.fc = nn.Linear(input_dim, 256)  # 입력층 -> 은닉층

        # 정책 업데이트 (Actor)
        self.policy = nn.Linear(256, action_space)  # 은닉층 -> 정책 (행동)

        # 가치 업데이트 (Critic)
        self.value = nn.Linear(256, 1)  # 은닉층 -> 가치 (상태 가치)

    def forward(self, x):
        """
        전방향 신경망 연산 수행
        상태(x)를 입력으로 받아 정책(policy)과 가치(value)를 출력
        스텝마다 정책은 forward 하여 업데이트 됨, 손실 계산과 모델 업데이트는 에피소드마다 이루어짐.

        Args:
            x (torch.Tensor): 입력 텐서

        Returns:
            tuple: 정책과 가치
        """
        # log_manager.logger.debug("ActorCritic forward ...")
        x = torch.relu(self.fc(x))  # 은닉층 활성화 함수로 ReLU 사용
        policy = self.policy(x)  # 행동 확률 분포
        value = self.value(x)  # 상태 가치
        return policy, value

class A3CAgent:
    def __init__(self, env, gamma=0.99, epsilon=0.05):
        """
        A3C 에이전트 초기화

        Args:
            env (gym.Env): 강화 학습 환경
            gamma (float): 할인율(0~1)
            epsilon (float): 무작위 행동을 선택할 확률
        """
        log_manager.logger.info("Initializing A3CAgent")
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon  # 입실론 값 추가
        self.model = ActorCritic(input_dim=env.observation_space.shape[0], action_space=env.action_space.n)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def select_action(self, state):
        """
        현재 상태에 대한 행동 선택

        Args:
            state (np.ndarray): 현재 상태 (주식 가격, 현금 잔고)

        Returns:
            tuple: 선택된 행동과 행동의 로그 확률
        """
        # 입실론 탐욕 정책 적용
        if random.random() < self.epsilon:
            action = random.randint(0, self.env.action_space.n - 1)
            log_prob = torch.log(torch.tensor(1.0 / self.env.action_space.n))
            return action, log_prob
        else:
            state = torch.from_numpy(state).float()
            policy, _ = self.model(state)
            policy = torch.softmax(policy, dim=-1)
            policy = policy.clamp(min=1e-10, max=1-1e-10)
            policy = policy / policy.sum()

            m = Categorical(policy)
            action = m.sample()
            return action.item(), m.log_prob(action)

    def compute_returns(self, rewards, dones, next_value):
        """
        반환값 계산

        Args:
            rewards (list): 보상 리스트
            dones (list): 에피소드 종료 여부 리스트
            next_value (torch.Tensor): 다음 상태의 가치

        Returns:
            list: 반환값 리스트
        """
        returns = []
        R = next_value
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * (1 - dones[step])
            returns.insert(0, R)
        # log_manager.logger.debug(f"Computed returns: {returns}")
        return returns

    def update(self, trajectory):
        """
        모델 업데이트
        각 에피소드에서 스텝마다의 값들을 배열에 추가함
        이전 상태, 보상을 고려할 수 있음

        Args:
            trajectory (tuple): 상태, 행동, 보상, 종료 여부, 다음 상태를 포함한 튜플
        """
        # log_manager.logger.debug("Updating model")
        states, actions, rewards, dones, next_state = trajectory
        _, next_value = self.model(torch.from_numpy(next_state).float())
        returns = self.compute_returns(rewards, dones, next_value)

        log_probs = []
        values = []
        actions = torch.tensor(actions)
        for i, state in enumerate(states):
            policy, value = self.model(torch.from_numpy(state).float())
            values.append(value)
            m = Categorical(torch.softmax(policy, dim=-1))
            log_probs.append(m.log_prob(actions[i]))

        values = torch.stack(values)
        returns = torch.tensor(returns).float()
        log_probs = torch.stack(log_probs)

        advantage = returns - values.squeeze()

        policy_loss = -(log_probs * advantage.detach()).mean()
        value_loss = advantage.pow(2).mean()

        self.optimizer.zero_grad()
        (policy_loss + value_loss).backward()
        self.optimizer.step()
        # log_manager.logger.debug("Model updated")

    def save_model(self, path):
        """
        모델 저장

        Args:
            path (str): 모델을 저장할 경로
        """
        torch.save(self.model.state_dict(), path)
        log_manager.logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        모델 로드

        Args:
            path (str): 모델을 로드할 경로
        """
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        log_manager.logger.info(f"Model loaded from {path}")

def update_global_model(global_model, local_model, optimizer):
    """
    로컬 모델의 기울기를 글로벌 모델로 업데이트

    Args:
        global_model (nn.Module): 글로벌 모델
        local_model (nn.Module): 로컬 모델
        optimizer (optim.Optimizer): 글로벌 모델의 옵티마이저
    """
    for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
        if local_param.grad is not None:
            global_param._grad = local_param.grad  # 로컬 모델의 기울기를 글로벌 모델에 복사

    optimizer.step()  # 글로벌 모델의 가중치 업데이트
    # log_manager.logger.debug("Global model updated with local gradients")

def worker(global_agent, env, n_episodes, global_ep, global_ep_lock, optimizer):
    """
    학습 작업자 함수

    Args:
        global_agent (A3CAgent): 글로벌 에이전트
        env (gym.Env): 환경
        n_episodes (int): 에피소드 수
        global_ep (mp.Value): 글로벌 에피소드 카운터
        global_ep_lock (mp.Lock): 에피소드 카운터 잠금
        optimizer (optim.Optimizer): 글로벌 모델의 옵티마이저
    """
    local_agent = A3CAgent(env)
    for _ in range(n_episodes):
        state = env.reset()
        states, actions, rewards, dones = [], [], [], []
        while True:
            action, _ = local_agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            state = next_state
            if done:
                break

        trajectory = (states, actions, rewards, dones, next_state)
        local_agent.update(trajectory)  # 로컬 모델 업데이트

        # 글로벌 모델 업데이트
        optimizer.zero_grad()
        update_global_model(global_agent.model, local_agent.model, optimizer)
        
        # 이 블록 안에 있는 코드는 다른 프로세스에서 동시에 실행되지 않도록 보장(lock)
        with global_ep_lock:
            global_ep.value += 1
            log_manager.logger.info(f'Episode {global_ep.value} completed')

if __name__ == '__main__':
    # log_manager.logger.info("Starting training process")
    df = pd.read_csv('data/data_csv/samsung_stock_data.csv')
    env = StockTradingEnv(df)
    global_agent = A3CAgent(env)
    global_ep = mp.Value('i', 0)
    global_ep_lock = mp.Lock()
    optimizer = optim.Adam(global_agent.model.parameters(), lr=0.001)  # 글로벌 모델의 옵티마이저

    processes = []
    n_processes = 4
    n_episodes = 20 // n_processes
    for rank in range(n_processes):
        p = mp.Process(target=worker, args=(global_agent, env, n_episodes, global_ep, global_ep_lock, optimizer))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    global_agent.save_model('models/a3c_stock_trading_model.pth')
    log_manager.logger.info("Training process completed")
