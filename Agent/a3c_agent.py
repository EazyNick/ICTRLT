"""
A3CAgent module for reinforcement learning.

에이전트 역할
"""

import sys
import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical

try:
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    from utils import log_manager
    from models import ActorCritic
    from config import ConfigLoader
except ImportError:
    print(ImportError)

class A3CAgent:
    """
    Actor-Critic (A3C) 에이전트 클래스
    """

    def __init__(self, env, gamma=0.99, epsilon=0.05):
        """
        A3C 에이전트 초기화

        Args:
            env (gym.Env): 강화 학습 환경
            gamma (float): 할인율(0~1)
            epsilon (float): 무작위 행동을 선택할 확률
        """
        log_manager.logger.info("Initializing A3CAgent")
        # ConfigLoader로 설정값 불러오기
        learning_rate = ConfigLoader.get_learning_rate() 
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.model = ActorCritic(
            input_dim=env.observation_space.shape[0],
            action_space=env.action_space.n
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

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
        Temporal Difference (TD) - 스텝마다 업데이트

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

    def update_batch(self, batch):
        """
        글로벌 모델에서 배치 업데이트를 수행합니다.

        Args:
            batch (list): 여러 워커에서 수집한 배치 데이터 (states, actions, rewards, dones, next_states).
        """
        states, actions, rewards, dones, next_states = zip(*batch)

        # 데이터를 텐서로 변환
        states = torch.tensor(np.vstack(states), dtype=torch.float32)
        actions = torch.tensor(np.hstack(actions), dtype=torch.int64)
        rewards = np.hstack(rewards)
        dones = np.hstack(dones)
        next_states = torch.tensor(np.vstack(next_states), dtype=torch.float32)

        # 다음 상태의 가치 계산
        _, next_value = self.model(next_states[-1])
        returns = self.compute_returns(rewards, dones, next_value)

        # 모델 예측 및 손실 계산
        log_probs = []
        values = []

        for i, state in enumerate(states):
            policy, value = self.model(state)
            m = Categorical(torch.softmax(policy, dim=-1))
            log_probs.append(m.log_prob(actions[i]))
            values.append(value)

        values = torch.stack(values).squeeze()
        returns = torch.tensor(returns, dtype=torch.float32)
        log_probs = torch.stack(log_probs)

        # 손실 계산
        advantage = returns - values
        policy_loss = -(log_probs * advantage.detach()).mean()
        value_loss = advantage.pow(2).mean()

        # 모델 업데이트
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
        log_manager.logger.info("Model saved to %s", path)

    def load_model(self, path):
        """
        모델 로드

        Args:
            path (str): 모델을 로드할 경로
        """
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        log_manager.logger.info("Model loaded from %s", path)

def sync_local_to_global(global_agent, local_agent):
    """
    로컬 에이전트의 기울기를 글로벌 에이전트로 동기화합니다.

    Args:
        global_agent (A3CAgent): 글로벌 에이전트
        local_agent (A3CAgent): 로컬 에이전트
    """
    for global_param, local_param in zip(
        global_agent.model.parameters(),
        local_agent.model.parameters()
    ):
        if local_param.grad is not None:
            global_param.grad = local_param.grad

    # 글로벌 모델 업데이트
    global_agent.optimizer.step()
    global_agent.optimizer.zero_grad()

    # 로컬 모델을 글로벌 모델의 파라미터로 동기화
    local_agent.model.load_state_dict(global_agent.model.state_dict())
