import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import pandas as pd
from env.env import StockTradingEnv

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_space):
        """
        Actor-Critic 모델 초기화

        Args:
            input_dim (int): 입력 차원
            action_space (int): 행동 공간의 크기
        """
        super(ActorCritic, self).__init__()
        self.fc = nn.Linear(input_dim, 128)  # 입력층 -> 은닉층

        # 정책 업데이트 (Actor):
        # 에이전트는 현재 정책을 사용하여 행동을 선택합니다.
        # 선택된 행동에 대한 보상과 Critic 네트워크에서 계산된 가치를 사용하여 정책을 업데이트합니다.
        # 정책은 주어진 상태에서 더 나은 행동을 선택할 수 있도록 조정됩니다.
        self.policy = nn.Linear(128, action_space)  # 은닉층 -> 정책 (행동)

        # 가치 업데이트 (Critic):
        # 에이전트는 환경에서의 보상과 다음 상태를 기반으로 현재 상태의 가치를 평가합니다.
        # Critic 네트워크는 TD (Temporal Difference) 오류를 최소화하도록 업데이트
        self.value = nn.Linear(128, 1)  # 은닉층 -> 가치 (상태 가치)

    def forward(self, x):
        """
        전방향 신경망 연산 수행
        상태(x)를 입력으로 받아 정책(policy)과 가치(value)를 출력

        Args:
            x (torch.Tensor): 입력 텐서

        Returns:
            tuple: 정책과 가치
        """
        x = torch.relu(self.fc(x))  # 은닉층 활성화 함수로 ReLU 사용
        policy = self.policy(x)  # 행동 확률 분포
        value = self.value(x)  # 상태 가치
        return policy, value

class A3CAgent:
    def __init__(self, env, gamma=0.99):
        """
        A3C 에이전트 초기화

        Args:
            env (gym.Env): 강화 학습 환경
            gamma (float): 할인율(0~1)
        """
        self.env = env
        self.gamma = gamma

        # 상태에서 행동을 선택하고 가치를 평가
        # 환경의 관찰 공간과 행동 공간의 크기를 기반으로 모델의 입력 차원과 출력 차원을 설정
        self.model = ActorCritic(input_dim=env.observation_space.shape[0], action_space=env.action_space.n)
        
        # Adam 옵티마이저로, 모델의 파라미터를 업데이트하는 데 사용
        # self.model 객체는 ActorCritic 클래스의 인스턴스
        # self.model.parameters()는 self.fc, self.policy, self.value의 모든 가중치와 바이어스를 업데이트 하는 것
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def select_action(self, state):
        """
        현재 상태에 대한 행동 선택

        Args:
            state (np.ndarray): 현재 상태 (주식 가격(current_price), 현금 잔고(cash_in_hand))

        Returns:
            tuple: 선택된 행동과 행동의 로그 확률
        """
        state = torch.from_numpy(state).float()  # 상태를 텐서로 변환
        policy, _ = self.model(state)  # 정책 및 가치를 예측
        policy = torch.softmax(policy, dim=-1)  # 행동 확률 분포 계산(각 행동이 선택될 확률)

        m = Categorical(policy)  # 카테고리 분포 생성(각 행동이 선택될 확률을 만듬.)
        # ex) 주사위를 던질 때 각 면(1, 2, 3, 4, 5, 6)이 나올 확률이 동일하게 1/6인 경우, 이는 6개의 범주로 구성된 카테고리 분포

        action = m.sample()  # 행동 샘플링, 행동 확률 분포에서 하나의 행동을 샘플링
        
        # 로그 확률(Log Probability)은 확률 값을 로그 함수로 변환한 값입니다. 이는 확률 값이 매우 작은 경우, 
        # 계산의 안정성과 수학적 편리성을 위해 사용됩니다. 로그 확률은 특히 확률 값이 곱해지는 상황에서 유용
        return action.item(), m.log_prob(action)  # 선택된 행동과 행동의 로그 확률 반환

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
            R = rewards[step] + self.gamma * R * (1 - dones[step])  # 반환값 계산
            returns.insert(0, R)  # 반환값을 리스트 앞에 추가
        return returns

    def update(self, trajectory):
        """
        모델 업데이트

        Args:
            trajectory (tuple): 상태, 행동, 보상, 종료 여부, 다음 상태를 포함한 튜플
        """
        states, actions, rewards, dones, next_state = trajectory
        _, next_value = self.model(torch.from_numpy(next_state).float())  # 다음 상태의 가치를 예측
        returns = self.compute_returns(rewards, dones, next_value)  # 반환값 계산

        log_probs = []
        values = []
        actions = torch.tensor(actions)  # actions 리스트를 텐서로 변환
        for i, state in enumerate(states):
            policy, value = self.model(torch.from_numpy(state).float())  # 상태의 정책 및 가치를 예측
            values.append(value)  # 상태 가치를 리스트에 추가
            m = Categorical(torch.softmax(policy, dim=-1))  # 카테고리 분포 생성
            log_probs.append(m.log_prob(actions[i]))  # 행동의 로그 확률을 리스트에 추가

        values = torch.stack(values)  # 리스트를 텐서로 변환
        returns = torch.tensor(returns).float()  # 리스트를 텐서로 변환
        log_probs = torch.stack(log_probs)  # 리스트를 텐서로 변환

        advantage = returns - values.squeeze()  # 어드밴티지 계산

        policy_loss = -(log_probs * advantage.detach()).mean()  # 정책 손실 계산
        value_loss = advantage.pow(2).mean()  # 가치 손실 계산

        self.optimizer.zero_grad()  # 옵티마이저 초기화
        (policy_loss + value_loss).backward()  # 역전파
        self.optimizer.step()  # 모델 업데이트

    def save_model(self, path):
        """
        모델 저장

        Args:
            path (str): 모델을 저장할 경로
        """
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        """
        모델 로드

        Args:
            path (str): 모델을 로드할 경로
        """
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

def worker(global_agent, env, n_episodes, global_ep, global_ep_lock):
    """
    학습 작업자 함수

    Args:
        global_agent (A3CAgent): 글로벌 에이전트
        env (gym.Env): 환경
        n_episodes (int): 에피소드 수
        global_ep (mp.Value): 글로벌 에피소드 카운터
        global_ep_lock (mp.Lock): 에피소드 카운터 잠금
    """
    local_agent = A3CAgent(env)  # 로컬 에이전트 생성
    for _ in range(n_episodes):
        state = env.reset()  # 환경 리셋
        states, actions, rewards, dones = [], [], [], []  # trajectory 초기화
        while True:
            action, log_prob = local_agent.select_action(state)  # 행동 선택
            next_state, reward, done, _ = env.step(action)  # 환경에서 다음 상태와 보상 얻기
            
            # Collect trajectory
            states.append(state)  # 상태 추가
            actions.append(action)  # 행동 추가
            rewards.append(reward)  # 보상 추가
            dones.append(done)  # 종료 여부 추가
            
            state = next_state  # 상태 업데이트
            if done:
                break
        
        # Add next state to trajectory
        trajectory = (states, actions, rewards, dones, next_state)
        
        # Update the global agent
        global_agent.update(trajectory)  # 글로벌 에이전트 업데이트

        with global_ep_lock:  # 글로벌 에피소드 카운터 업데이트
            global_ep.value += 1
            print(f'Episode {global_ep.value} completed')

if __name__ == '__main__':
    df = pd.read_csv('data/data_csv/samsung_stock_data.csv')  # 주식 데이터 로드
    env = StockTradingEnv(df)  # 환경 생성
    global_agent = A3CAgent(env)  # 글로벌 에이전트 생성
    global_ep = mp.Value('i', 0)  # 글로벌 에피소드 카운터
    global_ep_lock = mp.Lock()  # 에피소드 카운터 잠금

    processes = []
    n_processes = 4  # 사용할 프로세스 수
    n_episodes = 1000 // n_processes  # 각 프로세스가 수행할 에피소드 수
    for rank in range(n_processes):
        p = mp.Process(target=worker, args=(global_agent, env, n_episodes, global_ep, global_ep_lock))
        p.start()  # 프로세스 시작
        processes.append(p)

    for p in processes:
        p.join()  # 모든 프로세스가 끝날 때까지 대기

    # 학습된 모델 저장
    global_agent.save_model('models/a3c_stock_trading_model.pth')
    print("Model saved!")
