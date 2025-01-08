"""
이 파일은 Actor-Critic 모델을 정의합니다.
이 모델은 A3C와 같은 강화 학습 알고리즘에서 사용되며,
정책 네트워크(Actor)와 가치 네트워크(Critic)로 구성됩니다.
"""

import torch
import torch.nn as nn
from utils import log_manager
from config import ConfigLoader
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_space):
        """
        Actor-Critic 모델 초기화
        정책 그라디언트 방법과 TD 방법을 결합한 Actor-Critic 구조
        A3C에서 사용하는 기본 모델

        Args:
            input_dim (int): 입력 차원
            action_space (int): 행동 공간의 크기

        [입력층]                [은닉층]                  [출력층]
        (input_dim)  ->   (128개의 은닉 노드)  ->  (policy: action_space) 
                                              ->  (value: 1)
        """
        super(ActorCritic, self).__init__()
        self.hidden_layer_size = ConfigLoader.get_hidden_layer_size()
        log_manager.logger.info("Initializing ActorCritic")

        self.fc = nn.Linear(input_dim, self.hidden_layer_size) # 입력층 -> 은닉층
        # 정책 업데이트 (Actor)
        self.policy = nn.Linear(self.hidden_layer_size, action_space)  # 은닉층 -> 정책 (행동)
        # 가치 업데이트 (Critic)
        self.value = nn.Linear(self.hidden_layer_size, 1)  # 은닉층 -> 가치 (상태 가치)

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

class EnhancedActorCritic(nn.Module):
    def __init__(self, input_dim, action_space):
        """
        향상된 Actor-Critic 모델 초기화
        - 더 깊은 신경망 구조
        - Batch Normalization 적용
        - Dropout 적용
        - Skip Connection 추가
        
        Args:
            input_dim (int): 입력 차원 (주가, 거래량 등의 특징)
            action_space (int): 행동 공간의 크기 (매수/매도/홀딩 등)
        """
        super(EnhancedActorCritic, self).__init__()
        log_manager.logger.info("Initializing EnhancedActorCritic")
        # 설정값
        self.hidden1_size = ConfigLoader.get_hidden1_size() 
        self.hidden2_size = ConfigLoader.get_hidden2_size()
        self.dropout_rate = ConfigLoader.get_dropout_rate()
        
        # 공통 신경망 계층
        self.fc1 = nn.Linear(input_dim, self.hidden1_size)
        self.norm1 = nn.LayerNorm(self.hidden1_size)
        self.fc2 = nn.Linear(self.hidden1_size, self.hidden2_size)
        self.norm2 = nn.LayerNorm(self.hidden2_size)
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Actor (정책) 신경망
        self.actor_fc1 = nn.Linear(self.hidden2_size, self.hidden2_size)
        self.actor_norm = nn.LayerNorm(self.hidden2_size)
        self.actor_output = nn.Linear(self.hidden2_size, action_space)
        
        # Critic (가치) 신경망
        self.critic_fc1 = nn.Linear(self.hidden2_size, self.hidden2_size)
        self.critic_norm = nn.LayerNorm(self.hidden2_size)
        self.critic_output = nn.Linear(self.hidden2_size, 1)
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """
        가중치 초기화 메서드
        He 초기화
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, x):
        """
        전방향 계산
        
        Args:
            x (torch.Tensor): 입력 텐서 (input_dim,) 또는 (배치, input_dim)
            
        Returns:
            tuple: (정책 확률 분포, 상태 가치)
        """
        # 1차원 입력 처리
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (input_dim,) -> (1, input_dim)

        # 공통 특징 추출
        x1 = F.relu(self.norm1(self.fc1(x)))
        x1 = self.dropout(x1)

        x2 = F.relu(self.norm2(self.fc2(x1)))
        x2 = self.dropout(x2)
        
        # Actor 경로 (Skip Connection 포함)
        actor = F.relu(self.actor_norm(self.actor_fc1(x2)) + x2)
        actor = self.dropout(actor)
        policy = F.softmax(self.actor_output(actor), dim=-1)
        
        # Critic 경로 (Skip Connection 포함)
        critic = F.relu(self.critic_norm(self.critic_fc1(x2)) + x2)
        critic = self.dropout(critic)
        value = self.critic_output(critic)
        
        # 1차원 입력이었다면 배치 차원 제거
        if x.size(0) == 1:
            policy = policy.squeeze(0)
            value = value.squeeze(0)
        
        return policy, value


# 개발 예정 #
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder

class StateOfTheArtActorCritic(nn.Module):
    def __init__(self, input_dim, action_space, seq_length=10):
        """
        최신 기술이 적용된 Actor-Critic 모델
        - Transformer 인코더로 시계열 패턴 포착
        - Multi-head Self-attention으로 중요 특징 강조
        - Gated Linear Units (GLU)로 비선형성 강화
        - Layer Normalization으로 학습 안정화
        
        Args:
            input_dim (int): 입력 특징 차원
            action_space (int): 행동 공간 크기
            seq_length (int): 시퀀스 길이 (과거 데이터 윈도우 크기)
        """
        super(StateOfTheArtActorCritic, self).__init__()
        
        # 하이퍼파라미터 설정
        self.hidden1_size = ConfigLoader.get_hidden1_size()
        self.hidden2_size = ConfigLoader.get_hidden2_size()
        self.dropout_rate = ConfigLoader.get_dropout_rate()
        self.num_heads = 4
        self.num_layers = 2
        
        # 특징 임베딩 계층
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_dim, self.hidden1_size),
            nn.LayerNorm(self.hidden1_size),
            nn.GELU()  # GELU activation (Gaussian Error Linear Units)
        )
        
        # Transformer 인코더 블록
        encoder_layer = TransformerEncoderLayer(
            d_model=self.hidden1_size,
            nhead=self.num_heads,
            dim_feedforward=self.hidden1_size * 4,
            dropout=self.dropout_rate,
            activation='gelu',
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # GLU 블록들
        self.glu1 = nn.Sequential(
            nn.Linear(self.hidden1_size, self.hidden1_size * 2),
            nn.GLU()
        )
        
        # Actor 네트워크
        self.actor_net = nn.Sequential(
            nn.Linear(self.hidden1_size, self.hidden2_size),
            nn.LayerNorm(self.hidden2_size),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden2_size, self.hidden2_size),
            nn.LayerNorm(self.hidden2_size),
            nn.GELU(),
            nn.Linear(self.hidden2_size, action_space)
        )
        
        # Critic 네트워크
        self.critic_net = nn.Sequential(
            nn.Linear(self.hidden1_size, self.hidden2_size),
            nn.LayerNorm(self.hidden2_size),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden2_size, self.hidden2_size),
            nn.LayerNorm(self.hidden2_size),
            nn.GELU(),
            nn.Linear(self.hidden2_size, 1)
        )
        
        # Attention Pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(self.hidden1_size, 1),
            nn.Softmax(dim=1)
        )
        
        # 가중치 초기화
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """가중치 초기화 - He 초기화와 xavier 초기화 조합"""
        if isinstance(module, nn.Linear):
            if getattr(module, "activation", None) == "gelu":
                torch.nn.init.kaiming_normal_(module.weight)
            else:
                torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
    def forward(self, x):
        """
        전방향 계산
        
        Args:
            x (torch.Tensor): 입력 텐서 (배치, 시퀀스 길이, 특징 수)
            
        Returns:
            tuple: (정책 확률 분포, 상태 가치)
        """
        # 특징 임베딩
        x = self.feature_embedding(x)
        
        # Transformer 인코딩
        x = self.transformer(x)
        
        # GLU 처리
        x = self.glu1(x)
        
        # Attention Pooling
        attention_weights = self.attention_pool(x)
        x = torch.sum(x * attention_weights, dim=1)
        
        # Actor와 Critic 분기
        policy = F.softmax(self.actor_net(x), dim=-1)
        value = self.critic_net(x)
        
        return policy, value

    def create_mask(self, seq_length):
        """미래 정보 유출 방지를 위한 마스크 생성"""
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
        return mask