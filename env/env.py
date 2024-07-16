import gym
from gym import spaces
import numpy as np
import pandas as pd
import os
import sys

try:
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    from utils import *
except ImportError:
    from utils import *


class StockTradingEnv(gym.Env):
    def __init__(self, df):
        """
        주식 데이터프레임 df를 입력으로 받아 환경을 초기화

        Args:
            df (pd.DataFrame): 주식 데이터프레임
        """
        super(StockTradingEnv, self).__init__()
        log_manager.logger.info(f"StockTradingEnv initialized")
        self.df = df
        self.current_step = 0
        self.cash_in_hand = 50000000  # 초기 현금
        self.stock_owned = 0  # 초기 주식 보유량
        self.action_space = spaces.Discrete(3)  # 세 가지 행동: 매수, 매도, 유지
        # log_manager.logger.info(f"Action space: {self.action_space}")

        # 관찰 공간 정의
        num_indicators = len([col for col in df.columns if 'SMA' in col])  # 이동평균선의 개수
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(num_indicators + 2,), dtype=np.float32)
        # log_manager.logger.info(f"Observation space: {self.observation_space}")

    def reset(self, new_df=None):
        """
        환경을 초기 상태로 재설정
        
        Args:
            new_df (pd.DataFrame, optional): 새로운 주식 데이터프레임. 제공되지 않으면 기존 데이터프레임을 사용.
        
        Returns:
            np.ndarray: 초기 관찰값
        """
        log_manager.logger.info(f"Environment reset start")
        self.current_step = 0
        self.cash_in_hand = 50000000  # 초기 현금
        self.stock_owned = 0  # 초기 주식 보유량
        if new_df is not None:
            self.df = new_df
            log_manager.logger.info(f"New data frame provided")
        initial_observation = self._next_observation()
        log_manager.logger.debug(f"Initial observation: {initial_observation}")
        return initial_observation

    def _next_observation(self):
        """
        현재 주식 가격(df['Close'])과 현금 잔고를 포함한 관찰값을 반환
        
        Returns:
            np.ndarray: 현재 관찰값
        """
        indicators = self.df.iloc[self.current_step].filter(like='SMA').values
        current_price = self.df['Close'].values[self.current_step]
        next_observation = np.concatenate(([current_price, self.cash_in_hand], indicators)).astype(np.float32)
        next_observation = np.nan_to_num(next_observation, nan=0.0)  # NaN 값을 0으로 대체
        # log_manager.logger.debug(f"Next observation: {next_observation}, current_price: {current_price}")
        return next_observation

    def step(self, action):
        """
        주어진 행동(action)에 따라 환경의 상태를 업데이트
        
        Args:
            action (int): 에이전트가 선택한 행동 (0: 매수, 1: 매도, 2: 유지)
        
        Returns:
            tuple: 다음 관찰값, 보상, 에피소드 종료 여부, 추가 정보
        """
        # log_manager.logger.info(f"Step {self.current_step}, Action: {action}")
        current_price = self.df['Close'].values[self.current_step]
        # log_manager.logger.debug(f"Current price: {current_price}")

        # 0: 매수, 1: 매도, 2: 관망
        if action == 0:  # 매수
            log_manager.logger.info(f"Action: Buy")
            self.stock_owned += 1
            self.cash_in_hand -= current_price
        elif action == 1 and self.stock_owned > 0:  # 매도 (보유 주식이 있을 때만)
            log_manager.logger.info(f"Action: Sell")
            self.stock_owned -= 1
            self.cash_in_hand += current_price
        else:  # 관망
            log_manager.logger.debug(f"Action: Hold")

        # log_manager.logger.debug(f"Stock owned: {self.stock_owned}, Cash in hand: {self.cash_in_hand}")

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        if done:
            log_manager.logger.info(f"Episode finished")

        reward = self.stock_owned * current_price + self.cash_in_hand
        # log_manager.logger.debug(f"Reward: {reward}")

        next_observation = self._next_observation()
        # log_manager.logger.debug(f"Next observation: {next_observation}")

        return next_observation, reward, done, {}

    def render(self, mode='human', close=False):
        """
        현재 시간 스텝과 총 수익을 출력
        
        Args:
            mode (str): 출력 모드. 'human'은 콘솔 출력.
            close (bool): 환경을 닫을지 여부. 기본값은 False.
        """
        profit = self.stock_owned * self.df['Close'].values[self.current_step] + self.cash_in_hand - 50000000
        # log_manager.logger.info(f"Profit: {profit}")

        if mode == 'human':
            log_manager.logger.info(f'Step: {self.current_step}, Profit: {profit}')
        else:
            raise NotImplementedError(f"Render mode '{mode}' is not supported.")
