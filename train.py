"""
A3C Stock Trading Training Script

이 스크립트는 Asynchronous Advantage Actor-Critic (A3C) 알고리즘을 사용하여
주식 거래 모델을 학습하는 데 사용됩니다. 주요 기능:
- 멀티프로세싱을 활용한 병렬 학습
- 배치 업데이트 및 글로벌-로컬 에이전트 동기화
- 학습된 모델 저장

구성:
1. 데이터 및 환경 초기화
2. 학습 프로세스 병렬 실행
3. 모델 저장
"""

import random
import pandas as pd
import numpy as np
import torch
import torch.multiprocessing as mp

from Agent import A3CAgent, sync_local_to_global
from env import StockTradingEnv
from utils import log_manager
from config import ConfigLoader

def set_seeds():
    """
    모든 난수 생성기의 시드값을 각각 설정하여 일관된 결과를 생성합니다.
    """
    config = ConfigLoader()
    random_seed = config.get_random_seed()
    numpy_seed = config.get_numpy_seed()
    torch_seed = config.get_torch_seed()

    if random_seed is not None:
        random.seed(random_seed)

    if numpy_seed is not None:
        np.random.seed(numpy_seed)

    if torch_seed is not None:
        torch.manual_seed(torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch_seed)

def worker(global_agent, data_path, n_episodes, global_ep, global_ep_lock, batch_size=32):
    """
    학습 작업자 함수

    Args:
        global_agent (A3CAgent): 글로벌 에이전트
        data_path: env 생성을 위한 데이터 저장 경로
        n_episodes (int): 에피소드 수
        global_ep (mp.Value): 글로벌 에피소드 카운터
        global_ep_lock (mp.Lock): 에피소드 카운터 잠금
        batch_size (int): 배치 크기
    """
    # set_seeds()
    df = pd.read_csv(data_path)
    env = StockTradingEnv(df)
    local_agent = A3CAgent(env)

    # env에 노이즈 추가
    # with torch.no_grad():
    #     for param in local_agent.model.parameters():
    #         noise = torch.randn_like(param) * 0.01
    #         param.add_(noise)

    batch = []  # 배치 데이터를 저장할 리스트
    sync_interval = 8  # 5 에피소드마다 글로벌 동기화 수행

    for _ in range(n_episodes):
        state = env.reset()
        # if hasattr(env, 'np_random'):
        #     env.np_random.seed(random.randint(0, 10000))

        # state = env._get_observation()  # 현재 상태 얻기
        episode_reward = 0  # 에피소드 동안 얻은 총 보상
        while True:
            action, _ = local_agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
        
            # 보상을 누적
            episode_reward += reward

            # 데이터를 배치에 저장
            batch.append((state, action, reward, done, next_state))

            # 배치 크기만큼 데이터가 모이면 로컬 업데이트 수행
            if len(batch) >= batch_size:
                # 로컬 에이전트 업데이트
                local_agent.update_batch(batch)
                # 글로벌 에이전트로 동기화
                # sync_local_to_global(global_agent, local_agent) # 로컬 -> 글로벌
                # local_agent.model.load_state_dict(global_agent.model.state_dict())  # 글로벌 -> 로컬 동기화 보장
                batch = []  # 배치 초기화

            state = next_state
            if done:
                break
        
        # 이 블록 안에 있는 코드는 다른 프로세스에서 동시에 실행되지 않도록 보장(lock)
        with global_ep_lock:
            global_ep.value += 1
            log_manager.logger.info(f"Episode {global_ep.value} completed with reward: {episode_reward}")
            # sync_local_to_global(global_agent, local_agent)
            if global_ep.value % sync_interval == 0:
                sync_local_to_global(global_agent, local_agent) # 로컬 -> 글로벌
                local_agent.model.load_state_dict(global_agent.model.state_dict())  # 글로벌 -> 로컬 동기화 보장
        
    # 남은 배치 데이터 처리
    if batch:
        local_agent.update_batch(batch)
        sync_local_to_global(global_agent, local_agent)
        # local_agent.model.load_state_dict(global_agent.model.state_dict())  # 글로벌 -> 로컬 동기화 보장

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

    return global_agent


def start_training(global_agent, data_path, n_processes=4, n_episodes=8, batch_size=32):
    """
    여러 프로세스를 사용하여 학습 프로세스를 시작합니다.

    Args:
        global_agent (A3CAgent): 글로벌 A3C 에이전트.
        data_path: 학습 데이터 저장 경로.
        n_processes (int): 학습에 사용할 프로세스 수.
        n_episodes (int): 학습할 총 에피소드 수.
        batch_size (int): 배치 크기.
    """
    global_ep = mp.Value('i', 0)
    global_ep_lock = mp.Lock()

    processes = []
    episodes_per_process = n_episodes // n_processes
    for process_num in range(n_processes):
        p = mp.Process(target=worker, args=(global_agent, data_path, episodes_per_process, global_ep, global_ep_lock, batch_size))
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

    set_seeds()

    data_path = 'data/data_csv/sp500_training_data.csv'
    model_path = 'output/sp500_trading_model_9192.pth'

    # 환경과 에이전트 초기화
    global_agent = initialize_environment_and_agent(data_path)

    config = ConfigLoader()
    n_processes = config.get_n_processes()
    n_episodes = config.get_n_episodes()
    batch_size = config.get_batch_size()

    # 학습 프로세스 시작
    start_training(global_agent, data_path, n_processes, n_episodes, batch_size)

    # 학습된 모델 저장
    save_trained_model(global_agent, model_path)

    log_manager.logger.info("Training process completed")
