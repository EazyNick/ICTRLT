import torch
import torch.multiprocessing as mp
import pandas as pd
from Agent import A3CAgent, sync_local_to_global
from env import StockTradingEnv
from utils import *
import numpy as np
import random

# 시드값 설정 함수
def set_seeds(random_seed=None, numpy_seed=None, torch_seed=None):
    """
    모든 난수 생성기의 시드값을 각각 설정하여 일관된 결과를 생성합니다.

    Args:
        random_seed (int, optional): Python random 모듈의 시드값
        numpy_seed (int, optional): NumPy 모듈의 시드값
        torch_seed (int, optional): PyTorch 모듈의 시드값
    """
    if random_seed is not None:
        random.seed(random_seed)

    if numpy_seed is not None:
        np.random.seed(numpy_seed)

    if torch_seed is not None:
        torch.manual_seed(torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch_seed)

# 시드값 설정 (각각의 시드값을 다르게 설정)
random_seed_value = 42
numpy_seed_value = 2023
torch_seed_value = 1234

def worker(global_agent, env, n_episodes, global_ep, global_ep_lock, batch_size=32):
    """
    학습 작업자 함수

    Args:
        global_agent (A3CAgent): 글로벌 에이전트
        env (gym.Env): 환경
        n_episodes (int): 에피소드 수
        global_ep (mp.Value): 글로벌 에피소드 카운터
        global_ep_lock (mp.Lock): 에피소드 카운터 잠금
        batch_size (int): 배치 크기
    """
    local_agent = A3CAgent(env)
    batch = []  # 배치 데이터를 저장할 리스트

    for _ in range(n_episodes):
        state = env.reset()
        states, actions, rewards, dones = [], [], [], []
        while True:
            action, _ = local_agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            # 데이터를 배치에 저장
            batch.append((state, action, reward, done, next_state))

            # 배치 크기만큼 데이터가 모이면 로컬 업데이트 수행
            if len(batch) >= batch_size:
                # 로컬 에이전트 업데이트
                local_agent.update_batch(batch)
                batch = []  # 배치 초기화

                # 글로벌 에이전트로 동기화
                sync_local_to_global(global_agent, local_agent)

            state = next_state
            if done:
                break
        
        # 이 블록 안에 있는 코드는 다른 프로세스에서 동시에 실행되지 않도록 보장(lock)
        with global_ep_lock:
            global_ep.value += 1
            log_manager.logger.info(f'Episode {global_ep.value} completed')

    # 남은 배치 데이터 처리
    if batch:
        local_agent.update_batch(batch)
        sync_local_to_global(global_agent, local_agent)

# --------------------------------------------------------------------------------------------------------
# 모델 학습 시작과 관련된 코드

def initialize_environment_and_agent(data_path):
    """
    주식 데이터를 로드하고 트레이딩 환경과 글로벌 A3C 에이전트를 초기화합니다.

    Args:
        data_path (str): 주식 데이터가 포함된 CSV 파일 경로.

    Returns:
        env (StockTradingEnv): 초기화된 트레이딩 환경.
        global_agent (A3CAgent): 초기화된 글로벌 A3C 에이전트.
        df (pd.DataFrame): 주식 데이터를 포함한 DataFrame.
    """
    # 주식 데이터를 CSV 파일에서 로드
    df = pd.read_csv(data_path)
    
    # 트레이딩 환경을 초기화
    env = StockTradingEnv(df)
    
    # 글로벌 A3C 에이전트를 초기화
    global_agent = A3CAgent(env)
    
    return env, global_agent, df


def start_training(global_agent, env, n_processes=4, n_episodes=8, batch_size=32):
    """
    여러 프로세스를 사용하여 학습 프로세스를 시작합니다.

    Args:
        global_agent (A3CAgent): 글로벌 A3C 에이전트.
        env (StockTradingEnv): 트레이딩 환경.
        n_processes (int): 학습에 사용할 프로세스 수.
        n_episodes (int): 학습할 총 에피소드 수.
        batch_size (int): 배치 크기.
    """
    global_ep = mp.Value('i', 0)
    global_ep_lock = mp.Lock()

    processes = []
    episodes_per_process = n_episodes // n_processes
    for process_num in range(n_processes):
        p = mp.Process(target=worker, args=(global_agent, env, episodes_per_process, global_ep, global_ep_lock, batch_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

def save_trained_model(global_agent, model_path):
    """
    학습된 모델을 지정된 경로에 저장합니다.

    Args:
        global_agent (A3CAgent): 학습된 글로벌 A3C 에이전트.
        model_path (str): 모델을 저장할 경로.
    """
    global_agent.save_model(model_path)

if __name__ == '__main__':
    data_path = 'data/data_csv/samsung_stock_data.csv'
    model_path = 'output/a3c_stock_trading_model.pth'

    # 환경과 에이전트 초기화
    env, global_agent, df = initialize_environment_and_agent(data_path)

    # 학습 프로세스 시작
    start_training(global_agent, env, 4, 8, 32)

    # 학습된 모델 저장
    save_trained_model(global_agent, model_path)

    log_manager.logger.info("Training process completed")
