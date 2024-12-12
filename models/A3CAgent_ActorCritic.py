import torch
import torch.nn as nn
from utils import log_manager
from config import *

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_space):
        """
        Actor-Critic 모델 초기화
        정책 그라디언트 방법과 TD 방법을 결합한 Actor-Critic 구조
        A3C에서 사용

        Args:
            input_dim (int): 입력 차원
            action_space (int): 행동 공간의 크기

        [입력층]                [은닉층]                  [출력층]
        (input_dim)  ->   (128개의 은닉 노드)  ->  (policy: action_space) 
                                              ->  (value: 1)
        """
        super(ActorCritic, self).__init__()
        log_manager.logger.info("Initializing ActorCritic model")
        hidden_layer_size = ConfigLoader.get_hidden_layer_size()  # 기본값: 0.001
        
        self.fc = nn.Linear(input_dim, hidden_layer_size) # 입력층 -> 은닉층
        # 정책 업데이트 (Actor)
        self.policy = nn.Linear(hidden_layer_size, action_space)  # 은닉층 -> 정책 (행동)
        # 가치 업데이트 (Critic)
        self.value = nn.Linear(hidden_layer_size, 1)  # 은닉층 -> 가치 (상태 가치)

    def forward(self, x):
        """
        전방향 신경망 연산 수행
        상태(x)를 입력으로 받아 정책(policy)과 가치(value)를 출력

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
