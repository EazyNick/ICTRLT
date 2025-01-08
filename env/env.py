"""
학습 환경(env)
"""

import os
import sys
import gym
from gym import spaces
import numpy as np
import pandas as pd
import random

try:
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    from utils import log_manager
    from config import ConfigLoader
except ImportError:
    print(ImportError)

class StockTradingEnv(gym.Env):
    """
    주식 거래 환경을 정의하는 클래스.
    """
    def __init__(self, df):
        """
        주식 데이터프레임 df를 입력으로 받아 환경을 초기화

        Args:
            df (pd.DataFrame): 주식 데이터프레임
        """
        super(StockTradingEnv, self).__init__()
        log_manager.logger.info(f"StockTradingEnv initialized")

        self.config = ConfigLoader()
        self.df = df
        self.current_step = 0
        self.cash_in_hand = random.uniform(self.config.get_cash_in_hand() * 0.8, self.config.get_cash_in_hand() * 1.2)
        self.previous_cash_in_hand = self.cash_in_hand  # 이전 현금 초기화
        self.stock_owned = 0  # 초기 주식 보유량
        self.max_stock = self.config.get_max_stock()  # 한 번에 매수/매도 가능한 최대 주식 수
        self.trading_charge = self.config.get_trading_charge()  # 거래 수수료
        self.trading_tax = self.config.get_trading_tax()  # 거래세

        # 행동: 0~(2*max_stock) (매도 0~max_stock, 유지 max_stock, 매수 max_stock+1~2*max_stock)
        self.action_space = spaces.Discrete(2 * self.max_stock + 1)
        # log_manager.logger.info(f"Action space: {self.action_space}")

        # 관찰 공간 정의
        num_sma = len([col for col in df.columns if 'SMA' in col])  # 이동평균선의 개수
        num_vma = len([col for col in df.columns if 'VMA' in col])
        num_high = len([col for col in df.columns if 'High' in col])
        num_low = len([col for col in df.columns if 'Low' in col])

        # 잔고, 주식 보유량을 포함한 관찰 공간 정의
        self.observation_space = spaces.Box(
            low=0, high=np.inf,
            shape=(5 + num_sma + num_vma + num_high + num_low,),
            dtype=np.float32
        )

        # 매수/매도 기록을 위한 리스트 초기화
        self.buy_sell_log = []

    def reset(self, new_df=None):
        """
        환경을 초기 상태로 재설정
        
        Args:
            new_df (pd.DataFrame, optional): 새로운 주식 데이터프레임. 제공되지 않으면 기존 데이터프레임을 사용.
        
        Returns:
            np.ndarray: 초기 관찰값
        """
        # log_manager.logger.info(f"Environment reset start")
        self.current_step = 0
        self.cash_in_hand = self.config.get_cash_in_hand()  # 초기 현금
        self.previous_cash_in_hand = self.cash_in_hand  # 초기 이전 현금을 현재 현금으로 설정
        self.stock_owned = 0  # 초기 주식 보유량

        if new_df is not None:
            if isinstance(new_df, pd.DataFrame) and not new_df.empty:
                self.df = new_df
            else:
                raise ValueError("Invalid dataframe provided for reset.")
            
        initial_observation = self._next_observation()
        # log_manager.logger.debug(f"Initial observation: {initial_observation}")
        self.buy_sell_log = []  # 매수/매도 기록 초기화
        return initial_observation

    def _next_observation(self):
        """
        현재 주식 가격(df['Close'])과 현금 잔고를 포함한 관찰값을 반환
        
        Returns:
            np.ndarray: 현재 관찰값
        """
        sma_values = self.df.iloc[self.current_step].filter(like='SMA').values
        vma_values = self.df.iloc[self.current_step].filter(like='VMA').values
        high_values = self.df.iloc[self.current_step].filter(like='High').values
        low_values = self.df.iloc[self.current_step].filter(like='Low').values
        current_price = self.df['Close'].values[self.current_step]
        volume = self.df['Volume'].values[self.current_step]

        # 데이터 정규화 (0~1 스케일)
        # sma_values = self.df.iloc[self.current_step].filter(like='SMA').values / self.df['Close'].max()
        # vma_values = self.df.iloc[self.current_step].filter(like='VMA').values / self.df['Volume'].max()
        # high_values = self.df.iloc[self.current_step].filter(like='High').values / self.df['Close'].max()
        # low_values = self.df.iloc[self.current_step].filter(like='Low').values / self.df['Close'].max()
        # current_price = self.df['Close'].values[self.current_step] / self.df['Close'].max()
        # volume = self.df['Volume'].values[self.current_step] / self.df['Volume'].max()

        next_observation = np.concatenate((
            [current_price, volume, self.cash_in_hand, self.previous_cash_in_hand, self.stock_owned],
            sma_values, vma_values, high_values, low_values
        )).astype(np.float32)

        # next_observation = np.concatenate((
        #     [current_price, self.cash_in_hand, self.stock_owned],
        #     sma_values, vma_values, high_values, low_values
        # )).astype(np.float32)
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

        if action < self.max_stock:
            # 매수: 현금 내에서만 매수 가능
            num_stocks_to_buy = max(0, min(action, 
                self.cash_in_hand // (current_price * (1 + self.trading_charge))
            ))
            if num_stocks_to_buy > 0:  # 실제 매수가 발생한 경우에만 로그 기록
                cost = num_stocks_to_buy * current_price * (1 + self.trading_charge)
                self.stock_owned += num_stocks_to_buy
                self.cash_in_hand -= cost
                self.buy_sell_log.append((self.df.index[self.current_step], 'buy', num_stocks_to_buy, current_price))
                # log_manager.logger.debug(f"Step: {self.current_step}, Action: Buy, Stocks Bought: {num_stocks_to_buy}, Stock Owned: {self.stock_owned}, Cash: {self.cash_in_hand}")

        elif action > self.max_stock:
            # 매도: 보유한 주식 내에서만 매도 가능
            num_stocks_to_sell = max(0, min(action - self.max_stock, self.stock_owned))
            if num_stocks_to_sell > 0:
                self.cash_in_hand += num_stocks_to_sell * current_price * (1 - self.trading_charge - self.trading_tax)
                self.stock_owned -= num_stocks_to_sell
                self.buy_sell_log.append((self.df.index[self.current_step], 'sell', num_stocks_to_sell, current_price))
                # log_manager.logger.debug(f"Step: {self.current_step}, Action: Sell, Stocks Sold: {num_stocks_to_sell}, Stock Owned: {self.stock_owned}, Cash: {self.cash_in_hand}")
                # if num_stocks_to_buy != 0:
                # log_manager.logger.info(f"{num_stocks_to_buy}주 매도")

        # log_manager.logger.debug(f"Stock owned: {self.stock_owned}, Cash in hand: {self.cash_in_hand}")

        # 이전 총 자산
        previous_total_asset = self.stock_owned * self.df['Close'].values[self.current_step - 1] + self.previous_cash_in_hand

        # 현재 총 자산
        current_total_asset = self.stock_owned * current_price + self.cash_in_hand

        # log_manager.logger.debug(f"current_total_asset: {current_total_asset}, previous_total_asset: {previous_total_asset}")

        # 보상을 총 자산의 변화로 계산
        reward = current_total_asset - previous_total_asset

        reward = max(reward, -1.0)  # 최대 손실 -1로 제한

        # 현재 현금을 이전 현금으로 업데이트
        self.previous_cash_in_hand = self.cash_in_hand

        # log_manager.logger.debug(f"Step: {self.current_step}, Action: {action}, Reward: {reward}")

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1 or current_total_asset <= 0
        # log_manager.logger.debug(f"done {done}")
        # if done:
        #     log_manager.logger.info(f"Episode finished")

        next_observation = self._next_observation()
        # log_manager.logger.debug(f"Next observation: {next_observation}")

        return next_observation, reward, done, {}

    def render(self, mode='human', close=False):
        """
        현재 시간 스텝과 총 수익을 출력
        
        Args:
            mode (str): 출력 모드. 'human'은 콘솔 출력. (기본값: 'human')
            close (bool): 환경을 닫을지 여부 (기본값: False)
        """
        profit = (
            self.stock_owned * self.df['Close'].values[self.current_step] + self.cash_in_hand - 10000000
        )
        # log_manager.logger.info(f"Profit: {profit}")

        if mode == 'human':
            pass
            # log_manager.logger.info(f'Step: {self.current_step}, Profit: {profit}')
        else:
            raise NotImplementedError(f"Render mode '{mode}' is not supported.")

if __name__ == '__main__':
    import random
    from pathlib import Path
    import pandas as pd
    import numpy as np
    from config import ConfigLoader
    import random
    file_path = Path(__file__).resolve().parent.parent / 'data/data_csv/sp500_training_data.csv'
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)  # 주식 데이터 로드
    test = StockTradingEnv(df)
    for i in range(10000):
        random_action = test.action_space.sample()  # 랜덤 액션 생성
        log_manager.logger.debug(f"Random action: {random_action}")