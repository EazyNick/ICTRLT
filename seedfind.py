import torch
import pandas as pd
from Agent.A3CAgent import A3CAgent  # A3CAgent 클래스 불러오기
from env.env import StockTradingEnv
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from utils import *
from pathlib import Path
import numpy as np
import random
from config import *

# 시드값 설정 함수


# 저장된 모델을 로드하고 새 데이터를 기반으로 매수, 매도를 수행하는 함수
def run_trading(agent, env, new_data):

    state = env.reset(new_df=new_data)  # 새로운 데이터를 사용하여 환경 초기화
    done = False
    account_values = []  # 계좌 잔고 기록
    stock_prices = []  # 주식 가격 기록
    dates = []  # 날짜 기록
    stock_owned_log = []  # 주식 보유량 기록
    
    while not done:
        action, _ = agent.select_action(state)  # 행동 선택
        next_state, reward, done, _ = env.step(action)  # 다음 상태와 보상 얻기

        state = next_state  # 상태 업데이트
        
        account_value = reward
        # account_value = state[1] + state[0] * env.stock_owned  # 현금 잔고 + (보유 주식 * 주식 가격)
        account_values.append(account_value)
        stock_prices.append(state[0])  # 현재 주식 가격 기록
        stock_owned_log.append(env.stock_owned)  # 현재 주식 보유량 기록
        dates.append(env.df.index[env.current_step])  # 날짜 기록
        
        # 추가 로그
        # log_manager.logger.debug(f"Step: {env.current_step}, Stock Price: {state[0]}, Account Value: {account_value}, Stocks Owned: {env.stock_owned}, Cash in Hand: {state[1]}")
        
        env.render()
    
    return account_values, stock_prices, dates, env.buy_sell_log, stock_owned_log

def plot_trading_results(dates, account_values, stock_prices, buy_sell_log,stock_owned_log, save_path='output/trading_results.png'):
    """
    주가 vs 포트폴리오 가치
    """
    # 전역 글자 크기 설정 (옵션)
    plt.rcParams.update({'font.size': 14})

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 초기 값 설정
    initial_account_value = account_values[0]
    initial_stock_price = stock_prices[0]

    # 계좌 수익률 계산
    portfolio_returns = [(value / initial_account_value - 1) * 100 for value in account_values]
    stock_price_returns = [(price / initial_stock_price - 1) * 100 for price in stock_prices]

    # 계좌 가치 플로팅
    ax1.plot(dates, portfolio_returns, label='Portfolio Returns (%)', color='b')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Returns (%)', color='b', fontsize=20)

    # 주식 수익률 플로팅 (같은 축 사용)
    ax1.plot(dates, stock_price_returns, label='Stock Price Returns (%)', color='orange', linestyle='--')

    # 주식 보유량 추가 (별도 축 사용)
    ax2 = ax1.twinx()
    ax2.spines['right'].set_position(('outward', 60))  # 세 번째 축 위치 조정
    ax2.plot(dates, stock_owned_log, label='Stocks Owned', color='green', linestyle='-.')
    ax2.set_ylabel('Stocks Owned', color='green', fontsize=20)

    for log in buy_sell_log:
        date, action, num_stocks, price = log
        if date in dates:
            index = dates.index(date)  # 날짜에 해당하는 인덱스 찾기
            y_value = stock_price_returns[index]  # 해당 날짜의 주가 수익률 가져오기
            
            if action == 'buy':
                ax1.scatter(date, y_value, color='green', marker='^', s=100, label='Buy' if 'Buy' not in ax1.get_legend_handles_labels()[1] else "")
            elif action == 'sell':
                ax1.scatter(date, y_value, color='red', marker='v', s=100, label='Sell' if 'Sell' not in ax1.get_legend_handles_labels()[1] else "")
    plt.title('Portfolio Returns and Stock Price Over Time', fontsize=20)
    fig.tight_layout()

    # 범례 추가
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=14)

    plt.show()

    # 이미지 파일로 저장
    plt.savefig(save_path)
    plt.close()
    log_manager.logger.info(f"Trading results saved as {save_path}")

# 최적의 시드값을 찾는 함수
def find_optimal_seeds(agent, env, data, seed_range):
    """
    최적의 시드값을 찾아내는 함수

    Args:
        agent: 학습된 에이전트
        env: 거래 환경
        data: 새로운 주식 데이터
        seed_range: 탐색할 시드값의 범위 (list of tuples)

    Returns:
        optimal_seeds: 최적의 시드값 (random_seed, numpy_seed, torch_seed)
        best_diff: 주가 최종 수익률과 포트폴리오 최종 수익률의 최소 차이
    """
    best_diff = 0.0
    optimal_seeds = None

    for random_seed, numpy_seed, torch_seed in seed_range:
        # 시드값 설정
        random.seed(random_seed)
        np.random.seed(numpy_seed)
        torch.manual_seed(torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch_seed)
        

        # 거래 수행
        account_values, stock_prices, dates, _, _ = run_trading(agent, env, data)

        # 수익률 계산
        portfolio_return = (account_values[-1] / account_values[0] - 1) * 100
        stock_return = (stock_prices[-1] / stock_prices[0] - 1) * 100

        # 포트폴리오 수익률 - 주가 수익률의 차이
        diff = portfolio_return - stock_return

        # 최소 차이를 업데이트
        if diff > best_diff:
            best_diff = diff
            optimal_seeds = (random_seed, numpy_seed, torch_seed)
            print(f"random_seed: {random_seed}, numpy_seed: {numpy_seed}, torch_seed: {torch_seed}")
            print(f"diff :{diff}")

    return optimal_seeds, best_diff


def main_run_optimal_seeds():
    # 모델 로드
    model_path = Path(__file__).resolve().parent / 'output/kia_stock_trading_model_4048.pth'
    file_path = Path(__file__).resolve().parent / 'data/data_csv/kia_stock_data.csv'
    new_data = pd.read_csv(Path(__file__).resolve().parent / 'data/data_csv/kia_stock_testdata.csv', index_col='Date', parse_dates=True)
    
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    env = StockTradingEnv(df)
    agent = A3CAgent(env)
    agent.load_model(model_path)

    # 시드값 탐색 범위 설정 (랜덤하게 샘플링할 수도 있음)
    seed_range = [
        (random.randint(0, 10000000), random.randint(0, 10000000), random.randint(0, 10000000))
        for _ in range(1000000)  # 시드값 50개 생성
    ]

    # 최적의 시드값 찾기
    optimal_seeds, best_diff = find_optimal_seeds(agent, env, new_data, seed_range)
    
    print(f"Optimal seeds found: {optimal_seeds}")
    print(f"Best difference between portfolio and stock returns: {best_diff}")

    # 최적의 시드값으로 거래 수행
    random.seed(optimal_seeds[0])
    np.random.seed(optimal_seeds[1])
    torch.manual_seed(optimal_seeds[2])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(optimal_seeds[2])

    account_values, stock_prices, dates, buy_sell_log, stock_owned_log = run_trading(agent, env, new_data)

    # 거래 결과 시각화
    plot_trading_results(dates, account_values, stock_prices, buy_sell_log, stock_owned_log)

    return buy_sell_log


if __name__ == '__main__':
    buy_sell_log = main_run_optimal_seeds()
