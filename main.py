import torch
import pandas as pd
from A3CAgent import A3CAgent  # A3CAgent 클래스 불러오기
from env.env import StockTradingEnv
import sys
import os
import matplotlib.pyplot as plt

try:
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    from utils import *
except ImportError:
    from utils import *

# 저장된 모델을 로드하고 새 데이터를 기반으로 매수, 매도를 수행하는 함수
def run_trading(agent, env, new_data):
    state = env.reset(new_df=new_data)  # 새로운 데이터를 사용하여 환경 초기화
    done = False
    account_values = []  # 계좌 잔고 기록
    stock_prices = []  # 주식 가격 기록
    
    while not done:
        action, _ = agent.select_action(state)  # 행동 선택
        next_state, reward, done, _ = env.step(action)  # 다음 상태와 보상 얻기
        state = next_state  # 상태 업데이트
        
        account_value = state[1] + state[0] * env.stock_owned  # 현금 잔고 + (보유 주식 * 주식 가격)
        account_values.append(account_value)
        stock_prices.append(state[0])  # 현재 주식 가격 기록
        
        # 추가 로그
        # log_manager.logger.info(f"Step: {env.current_step}, Stock Price: {state[0]}, Account Value: {account_value}")
        
        env.render()
    
    return account_values, stock_prices

if __name__ == '__main__':
    log_manager.logger.info("Starting trading process")
    # 모델 로드
    model_path = 'models/a3c_stock_trading_model.pth'
    df = pd.read_csv('data/data_csv/samsung_stock_data.csv')  # 주식 데이터 로드
    env = StockTradingEnv(df)  # 환경 생성
    agent = A3CAgent(env)  # 에이전트 생성
    agent.load_model(model_path)  # 학습된 모델 로드

    # 새 데이터를 기반으로 거래 수행
    new_data = pd.read_csv('data/data_csv/samsung_stock_data.csv')  # 새로운 주식 데이터 로드
    account_values, stock_prices = run_trading(agent, env, new_data)
    
    # 결과를 그래프로 시각화
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(account_values, label='Account Value', color='b')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Account Value', color='b')

    ax2 = ax1.twinx()
    ax2.plot(stock_prices, label='Samsung Stock Price', color='orange')
    ax2.set_ylabel('Samsung Stock Price', color='orange')
    
    plt.title('Account Value and Samsung Stock Price Over Time')
    fig.tight_layout()

    # 두 레전드를 하나로 병합하여 왼쪽 상단에 배치
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    
    plt.show()
