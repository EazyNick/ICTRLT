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
    from models import ActorCritic, EnhancedActorCritic
    from config import ConfigLoader
except ImportError:
    print(ImportError)

class A3CAgent:
    """
    Actor-Critic (A3C) 에이전트 클래스
    """

    def __init__(self, env, gamma=0.99, epsilon=0.4):
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

        MODEL_CLASSES = {
            "ActorCritic": ActorCritic,
            "EnhancedActorCritic": EnhancedActorCritic,
            # 필요하면 추가 모델 정의
        }

        # 모델 이름 가져오기
        model_name = ConfigLoader.get_model_name()

        # 모델 클래스 매핑에서 선택
        ModelClass = MODEL_CLASSES.get(model_name, ActorCritic)  # 기본값 ActorCritic

        self.model = ModelClass(
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
            self.epsilon = max(self.epsilon * 0.999, 0.1)  # 탐험 감소
            # log_manager.logger.debug(f"랜덤할 때: Process {os.getpid()} Action: {action}, Epsilon: {self.epsilon}")
            return action, log_prob
        else:
            state = torch.from_numpy(state).float()
            policy, _ = self.model(state)
           # 수정되어야 할 순서
            policy = torch.clamp(policy, -20, 20)  # 로짓값을 클램핑
            policy = torch.softmax(policy, dim=-1)  # 그 다음 확률로 변환
            policy = policy / policy.sum()  # 확률 정규화 보장
            if torch.isnan(policy).any() or torch.isinf(policy).any():
                log_manager.logger.error(f"NaN or Inf detected in policy: {policy}")
            # 값이 너무 작아 부동소수점들 모여있을 경우, 1보다 약간 클 수도 있음.
            if not torch.isclose(policy.sum(), torch.tensor(1.0), atol=1e-5):
                raise ValueError("Policy probabilities do not sum to 1.")

            m = Categorical(policy)
            action = m.sample()
            # log_manager.logger.debug(f"랜덤하지 않을 때: Process {os.getpid()} Action: {action}, Epsilon: {self.epsilon}")
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
        로컬 에이전트 모델에서 배치 업데이트를 수행합니다.

        Args:
            batch (list): 여러 워커에서 수집한 배치 데이터 (states, actions, rewards, dones, next_states).
        Returns:
            policy_loss.item() (float): 정책 손실의 스칼라 값.
            entropy.item() (float): 정책 엔트로피의 스칼라 값.
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
        entropies = []

        for i, state in enumerate(states):
            policy, value = self.model(state)
            # softmax 적용 전에 클리핑
            policy = torch.clamp(policy, -20, 20)  # 극단적인 값 방지, 신경망의 출력값이 너무 크거나 작은 극단적인 값(-20보다 작거나 20보다 큰)이 되는 것을 방지
            policy = torch.softmax(policy, dim=-1)
            policy = policy.clamp(min=1e-7, max=1-1e-7)
            policy = policy / policy.sum()
            # 두 값(스칼라나 텐서)이 특정 허용 오차(atol) 이내에서 같은지 비교
            if not torch.isclose(policy.sum(), torch.tensor(1.0), atol=1e-5):
                log_manager.logger.error(f"Invalid policy: {policy}, sum: {policy.sum()}")
                raise ValueError("Policy probabilities do not sum to 1.")
            m = Categorical(policy)
            log_probs.append(m.log_prob(actions[i]))
            values.append(value)
            entropies.append(m.entropy())  # 엔트로피 계산

            # m = Categorical(torch.softmax(policy, dim=-1))
            # log_probs.append(m.log_prob(actions[i]))
            # values.append(value)

        values = torch.stack(values).squeeze()
        returns = torch.tensor(returns, dtype=torch.float32)
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)

        # 손실 계산
        advantage = returns - values
        policy_loss = -(log_probs * advantage.detach()).mean()
        value_loss = advantage.pow(2).mean()
        entropy = entropies.mean()

        # 모델 업데이트
        self.optimizer.zero_grad()
        # 학습 초기 단계에서 에이전트가 탐험을 충분히 하도록 유도하고, 
        # 너무 빨리 특정 행동에 수렴하지 않도록 하기 위해 엔트로피를 보너스로 사용
        (policy_loss + value_loss - 0.01 * entropy).backward()
        self.optimizer.step()
        # log_manager.logger.debug("Model updated") 

        return policy_loss.item(), entropy.item()

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
    동기화 후 다시 로컬 에이전트에 적용합니다.

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
    log_manager.logger.info(f"로컬 -> 글로벌 모델 업데이트")

    # 로컬 모델을 글로벌 모델의 파라미터로 동기화
    local_agent.model.load_state_dict(global_agent.model.state_dict())
    log_manager.logger.info(f"글로벌 -> 로컬 파라미터 동기화")
