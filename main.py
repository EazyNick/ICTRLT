import pandas as pd
from Agent.A3CAgent import A3CAgent  # A3CAgent 클래스 불러오기
from env.env import StockTradingEnv
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from utils import *
from pathlib import Path

# 저장된 모델을 로드하고 새 데이터를 기반으로 매수, 매도를 수행하는 함수
def run_trading(agent, env, new_data):
    state = env.reset(new_df=new_data)  # 새로운 데이터를 사용하여 환경 초기화
    done = False
    account_values = []  # 계좌 잔고 기록
    stock_prices = []  # 주식 가격 기록
    dates = []  # 날짜 기록
    
    while not done:
        action, _ = agent.select_action(state)  # 행동 선택
        next_state, reward, done, _ = env.step(action)  # 다음 상태와 보상 얻기
        state = next_state  # 상태 업데이트
        
        account_value = state[1] + state[0] * env.stock_owned  # 현금 잔고 + (보유 주식 * 주식 가격)
        account_values.append(account_value)
        stock_prices.append(state[0])  # 현재 주식 가격 기록
        dates.append(env.df.index[env.current_step])  # 날짜 기록
        
        # 추가 로그
        log_manager.logger.info(f"Step: {env.current_step}, Stock Price: {state[0]}, Account Value: {account_value}, Stocks Owned: {env.stock_owned}, Cash in Hand: {state[1]}")
        
        env.render()
    
    return account_values, stock_prices, dates, env.buy_sell_log

def plot_trading_results(dates, account_values, stock_prices, buy_sell_log, save_path='output/trading_results.png'):
    """
    주가 vs 포트폴리오 가치
    """
    # 전역 글자 크기 설정 (옵션)
    plt.rcParams.update({'font.size': 14})

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 초기 값 설정
    initial_account_value = account_values[0]
    initial_stock_price = stock_prices[0]

    # 계좌 가치와 주식 가격의 상대적인 변화 계산
    relative_account_values = [value / initial_account_value for value in account_values]
    relative_stock_prices = [price / initial_stock_price for price in stock_prices]

    # 계좌 가치 플로팅
    ax1.plot(dates, relative_account_values, label='Account Value', color='b')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Relative Account Value', color='b', fontsize=20)

    # 주식 가격 플로팅을 위한 두 번째 y축 설정
    ax2 = ax1.twinx()
    ax2.plot(dates, stock_prices, label='Stock Price', color='orange', linestyle='--')
    ax2.set_ylabel('Stock Price', color='orange', fontsize=20)

    for log in buy_sell_log:
        date, action, num_stocks, price = log
        if action == 'buy':
            ax2.scatter(date, price, color='green', marker='^', label='Buy' if 'Buy' not in ax2.get_legend_handles_labels()[1] else "")
        elif action == 'sell':
            ax2.scatter(date, price, color='red', marker='v', label='Sell' if 'Sell' not in ax2.get_legend_handles_labels()[1] else "")

    plt.title('Account Value and Stock Price Over Time', fontsize=20)
    fig.tight_layout()

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
    model_path = Path(__file__).resolve().parent / 'output/a3c_stock_trading_model_SamSung_2048.pth'
    file_path = Path(__file__).resolve().parent / 'data/data_csv/samsung_stock_data.csv'
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)  # 주식 데이터 로드
    env = StockTradingEnv(df)  # 환경 생성
    agent = A3CAgent(env)  # 에이전트 생성
    agent.load_model(model_path)  # 학습된 모델 로드

    # 새 데이터를 기반으로 거래 수행
    new_data = pd.read_csv(Path(__file__).resolve().parent / 'data/data_csv/Samsung_stock_testdata.csv', index_col='Date', parse_dates=True)  # 새로운 주식 데이터 로드
    account_values, stock_prices, dates, buy_sell_log = run_trading(agent, env, new_data)

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
    plot_trading_results(dates, account_values, stock_prices, buy_sell_log)

    return buy_sell_log

if __name__ == '__main__':
    buy_sell_log = main_run()
    print(f"Buy dates: {buy_sell_log}")
    log_manager.logger.info(f"Buy dates: {buy_sell_log}")
