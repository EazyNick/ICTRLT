import gym
from gym import spaces
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
    def __init__(self, df):
        """
        주식 데이터프레임 df를 입력으로 받아 환경을 초기화

        Args:
            df (pd.DataFrame): 주식 데이터프레임
        """
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.current_step = 0
        self.cash_in_hand = 10000  # 초기 현금
        self.stock_owned = 0  # 초기 주식 보유량
        self.action_space = spaces.Discrete(3)  # 세 가지 행동: 매수, 매도, 유지

        # low:
        # 관찰 공간의 각 차원에서 가능한 최솟값을 정의합니다.
        # low=0은 관찰 공간의 모든 차원의 최솟값이 0임을 의미합니다.
        # 이 경우, 주식 가격과 현금 잔고가 0 이상임을 나타냅니다.

        # high:
        # 관찰 공간의 각 차원에서 가능한 최댓값을 정의합니다.
        # high=np.inf는 관찰 공간의 모든 차원의 최댓값이 무한대임을 의미합니다.
        # 즉, 주식 가격과 현금 잔고가 이론적으로 무한대까지 증가할 수 있음을 나타냅니다.

        # shape:
        # 관찰 공간의 차원 수를 정의합니다.
        # shape=(2,)는 관찰 공간이 2차원 벡터로 구성됨을 의미합니다.
        # 이 경우, 관찰 공간은 두 개의 요소로 구성됩니다: 주식 가격과 현금 잔고.

        # dtype:
        # 관찰 공간의 데이터 타입을 정의합니다.
        # dtype=np.float32는 관찰 공간의 값이 32비트 부동 소수점(실수)으로 저장됨을 의미합니다.
        # 이는 주식 가격과 현금 잔고가 실수 값으로 표현됨을 나타냅니다.
        # 에이전트가 관찰할 수 있는 상태 공간, 주식 가격과 현금 잔고를 포함
        # 에이전트의 공간(low, max를 정의 하는 것), 실제 값 반환은 _next_observation에서 진행
        # _next_observation 값 범위는 observation_space에서 정의한 값 이내여야 함.
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32)  

    def reset(self, new_df=None):
        """
        환경을 초기 상태로 재설정
        
        Args:
            new_df (pd.DataFrame, optional): 새로운 주식 데이터프레임. 제공되지 않으면 기존 데이터프레임을 사용.
        
        Returns:
            np.ndarray: 초기 관찰값
        """
        self.current_step = 0
        self.cash_in_hand = 10000  # 초기 현금
        self.stock_owned = 0  # 초기 주식 보유량
        if new_df is not None:
            self.df = new_df
        return self._next_observation()

    def _next_observation(self):
        """
        현재 주식 가격(df['Close'])과 현금 잔고를 포함한 관찰값을 반환
        
        Returns:
            np.ndarray: 현재 관찰값
        """
        return np.array([self.df['Close'].values[self.current_step], self.cash_in_hand], dtype=np.float32)

    def step(self, action):
        """
        주어진 행동(action)에 따라 환경의 상태를 업데이트
        
        Args:
            action (int): 에이전트가 선택한 행동 (0: 매수, 1: 매도, 2: 유지)
        
        Returns:
            tuple: 다음 관찰값, 보상, 에피소드 종료 여부, 추가 정보
        """
        # 현재 주식 가격을 current_price로 설정
        current_price = self.df['Close'].values[self.current_step]

        # 0: 매수, 1: 매도, 2: 관망
        if action == 0:  # 매수
            self.stock_owned += 1
            self.cash_in_hand -= current_price
        elif action == 1:  # 매도
            self.stock_owned -= 1
            self.cash_in_hand += current_price

        # 시간 스텝을 1 증가
        self.current_step += 1
        # done 변수를 설정하여 에피소드가 끝났는지 확인
        done = self.current_step >= len(self.df) - 1

        # 보상(reward)을 계산, 현재 보유한 주식의 총 가치와 현금 잔고의 합계.
        reward = self.stock_owned * current_price + self.cash_in_hand

        return self._next_observation(), reward, done, {}

    def render(self, mode='human', close=False):
        """
        현재 시간 스텝과 총 수익을 출력
        
        Args:
            mode (str): 출력 모드. 'human'은 콘솔 출력.
            close (bool): 환경을 닫을지 여부. 기본값은 False.
        """
        profit = self.stock_owned * self.df['Close'].values[self.current_step] + self.cash_in_hand - 10000
        
        if mode == 'human':
            print(f'Step: {self.current_step}, Profit: {profit}')
        else:
            raise NotImplementedError(f"Render mode '{mode}' is not supported.")
