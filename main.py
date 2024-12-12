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
def set_seeds():
    """
    모든 난수 생성기의 시드값을 각각 설정하여 일관된 결과를 생성합니다.

    Args:
        random_seed (int, optional): Python random 모듈의 시드값
        numpy_seed (int, optional): NumPy 모듈의 시드값
        torch_seed (int, optional): PyTorch 모듈의 시드값
    """
    config = ConfigLoader()
    _random_seed_value = config.get_cash_in_hand()
    _numpy_seed_value = config.get_max_stock()
    _torch_seed_value = config.get_trading_charge()
    random_seed = _random_seed_value
    numpy_seed = _numpy_seed_value
    torch_seed = _torch_seed_value

    if random_seed is not None:
        random.seed(random_seed)

    if numpy_seed is not None:
        np.random.seed(numpy_seed)

    if torch_seed is not None:
        torch.manual_seed(torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch_seed)

# 저장된 모델을 로드하고 새 데이터를 기반으로 매수, 매도를 수행하는 함수
def run_trading(agent, env, new_data):
    # 시드값 설정
    set_seeds()

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

def plot_comparison_results(dates, model_returns, kospi_returns, buy_sell_log, save_path='output/comparison_results.png'):
    """
    지수 vs 포트폴리오 가치, 최신 UI 스타일 적용 (y축 통합)
    """
    plt.rcParams.update({
        'font.size': 13,
        'axes.facecolor': '#f4f4f4',  # 그래프 배경색
        'axes.edgecolor': '#cccccc',
        'axes.grid': True,
        'grid.color': '#dddddd',
        'grid.linestyle': '--',
        'legend.frameon': False,
        'legend.loc': 'best'
    })

    fig, ax = plt.subplots(figsize=(14, 7))

    # 포트폴리오 가치 플로팅
    ax.plot(dates, model_returns, label='Portfolio Value', color='#007aff', linewidth=2.5)

    # 코스피 지수 수익률 플로팅
    ax.plot(dates, kospi_returns, label='KOSPI Index Return %', color='#ff9500', linestyle='--', linewidth=2.5)

    ax.set_xlabel('Date', fontsize=14, color='#333333')
    ax.set_ylabel('Return %', fontsize=14, color='#333333')

    # 매수 및 매도 시점 표시
    for log in buy_sell_log:
        date, action, num_stocks, price = log
        if date in model_returns.index:  # 날짜가 model_returns에 존재할 때만 처리
            if action == 'buy':
                ax.scatter(date, model_returns.loc[date], color='#28a745', marker='^', s=100, edgecolor='black', linewidth=1.5, label='Buy' if 'Buy' not in ax.get_legend_handles_labels()[1] else "")
            elif action == 'sell':
                ax.scatter(date, model_returns.loc[date], color='#dc3545', marker='v', s=100, edgecolor='black', linewidth=1.5, label='Sell' if 'Sell' not in ax.get_legend_handles_labels()[1] else "")

    # 타이틀 추가
    plt.title('Portfolio Value and KOSPI Index Return Over Time', fontsize=18, color='#333333', weight='bold')
    fig.tight_layout()

    # 범례 추가
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc='upper left', fontsize=14)

    # 그래프 보여주기
    plt.show()

    # 이미지 파일로 저장 (활성화하려면 주석을 해제하세요)
    # plt.savefig(save_path)
    # plt.close()
    # log_manager.logger.info(f"Comparison results saved as {save_path}")

def main_run():
    # log_manager.logger.info("Starting trading process")

    # 모델 로드
    # 삼성전자
    model_path = Path(__file__).resolve().parent / 'output/kia_stock_trading_model_4048.pth'
    file_path = Path(__file__).resolve().parent / 'data/data_csv/kia_stock_data.csv'
    new_data = pd.read_csv(Path(__file__).resolve().parent / 'data/data_csv/kia_stock_testdata.csv', index_col='Date', parse_dates=True)  # 새로운 주식 데이터 로드
    
    # 기아차
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)  # 주식 데이터 로드
    env = StockTradingEnv(df)  # 환경 생성
    agent = A3CAgent(env)  # 에이전트 생성
    agent.load_model(model_path)  # 학습된 모델 로드

    # 새 데이터를 기반으로 거래 수행
    account_values, stock_prices, dates, buy_sell_log, stock_owned_log = run_trading(agent, env, new_data)

    # 코스피 지수 수익률% vs 모델 수익률%
    # 기아차 데이터로 모델의 수익률 계산
    # model_returns = [(value / account_values[0] - 1) * 100 for value in account_values]
    # model_returns = pd.Series(model_returns, index=dates)  # model_returns를 pandas Series로 변환

    # # 코스피 지수 데이터 로드 및 수익률 계산
    # kospi_data = pd.read_csv('/content/drive/My Drive/Colab_ICTRLT/data_csv/kospi_data.csv', index_col='Date', parse_dates=True)
    # kospi_prices = kospi_data['Close'].values
    # kospi_returns = [(price / kospi_prices[0] - 1) * 100 for price in kospi_prices]
    # kospi_returns = pd.Series(kospi_returns, index=kospi_data.index)  # kospi_returns를 pandas Series로 변환

    # # 날짜를 코스피 데이터에 맞추기 위해 재설정
    # common_dates = kospi_data.index.intersection(new_data.index)

    # # 공통 날짜 범위 설정
    # start_date = max(model_returns.index.min(), kospi_returns.index.min())
    # end_date = min(model_returns.index.max(), kospi_returns.index.max())

    # # 날짜 범위를 일치시켜 데이터 필터링
    # model_returns_filtered = model_returns.loc[start_date:end_date]
    # kospi_returns_filtered = kospi_returns.loc[start_date:end_date]

    # # 날짜 범위를 재확인 후, 공통된 날짜로 다시 필터링
    # common_dates = model_returns_filtered.index.intersection(kospi_returns_filtered.index)
    # model_returns_filtered = model_returns_filtered.loc[common_dates]
    # kospi_returns_filtered = kospi_returns_filtered.loc[common_dates]

    # # 비교 결과 시각화
    # plot_comparison_results(common_dates, model_returns_filtered, kospi_returns_filtered, buy_sell_log)

    # 거래 결과 플롯 및 저장
    plot_trading_results(dates, account_values, stock_prices, buy_sell_log, stock_owned_log)

    return buy_sell_log

if __name__ == '__main__':
    buy_sell_log = main_run()
    # print(f"Buy dates: {buy_sell_log}")
    # log_manager.logger.info(f"Buy dates: {buy_sell_log}")
