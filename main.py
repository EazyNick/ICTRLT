import torch
import pandas as pd
from A3CAgent import A3CAgent  # A3CAgent 클래스 불러오기
from env.env import StockTradingEnv

# 저장된 모델을 로드하고 새 데이터를 기반으로 매수, 매도를 수행하는 함수
def run_trading(agent, env, new_data):
    state = env.reset(new_data)  # 새로운 데이터를 사용하여 환경 초기화
    done = False
    while not done:
        action, _ = agent.select_action(state)  # 행동 선택
        state, reward, done, _ = env.step(action)  # 다음 상태와 보상 얻기
        env.render()

if __name__ == '__main__':
    # 모델 로드
    model_path = 'models/a3c_stock_trading_model.pth'
    df = pd.read_csv('data/data_csv/samsung_stock_data.csv')  # 주식 데이터 로드
    env = StockTradingEnv(df)  # 환경 생성
    agent = A3CAgent(env)  # 에이전트 생성
    agent.load_model(model_path)  # 학습된 모델 로드

    # 새 데이터를 기반으로 거래 수행
    new_data = pd.read_csv('data/data_csv/samsung_stock_data.csv')  # 새로운 주식 데이터 로드
    run_trading(agent, env, new_data)
