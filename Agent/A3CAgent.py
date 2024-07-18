import torch
import torch.optim as optim
from torch.distributions import Categorical
import sys
import os
import random

try:
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    from utils import *
    from env import *
    from models import *
except ImportError:    
    from utils import *
    from env import *
    from models import *

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
        self.epsilon = epsilon
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
        # 입실론 탐욕 정책 적용(랜덤한 확률로 정책을 따르지 않고 바보같은 행동을 하는 것)
        if random.random() < self.epsilon:
            action = random.randint(0, self.env.action_space.n - 1)
            log_prob = torch.log(torch.tensor(1.0 / self.env.action_space.n))
            return action, log_prob
        else:
            state = torch.from_numpy(state).float()
            policy, _ = self.model(state)
            policy = torch.softmax(policy, dim=-1)
            # 클램핑(정책 값이 1에 너무 가깝거나, 0에 너무 가까운 극단적인 경우 제외)과 정규화 추가
            policy = policy.clamp(min=1e-10, max=1-1e-10)
            policy = policy / policy.sum()

            m = Categorical(policy)
            action = m.sample()
            return action.item(), m.log_prob(action)
        
    def compute_returns(self, rewards, dones, next_value):
        """
        주어진 보상과 에피소드 종료 여부 리스트, 그리고 다음 상태의 가치를 이용하여 반환값을 계산

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
        각 에피소드에서 각각의 스텝의 값들을 배열에 추가함
        이전 스텝 상태, 보상을 고려할 수 있음

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
    손실 함수(loss function)에 대한 각 가중치의 기울기를 계산하여, 이 기울기를 이용해 가중치를 업데이트

    Args:
        global_model (nn.Module): 글로벌 모델
        local_model (nn.Module): 로컬 모델
        optimizer (optim.Optimizer): 글로벌 모델의 옵티마이저
    """
    for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
        if local_param.grad is not None:
            global_param._grad = local_param.grad  # 로컬 모델의 기울기를 글로벌 모델에 복사

    optimizer.step()  # 글로벌 모델의 가중치 업데이트
