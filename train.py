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
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import queue 

from Agent import A3CAgent, sync_local_to_global
from env import StockTradingEnv
from utils import log_manager
from config import ConfigLoader

# tensorboard --logdir=utils/Log/tensorboard_logs/ --port=6006

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

def worker(global_agent, data_path, n_episodes, global_ep, global_ep_lock, writer_queue, process_id=0, batch_size=32):
    """
    학습 작업자 함수

    Args:
        global_agent (A3CAgent): 글로벌 에이전트
        data_path: env 생성을 위한 데이터 저장 경로
        n_episodes (int): 에피소드 수
        global_ep (mp.Value): 글로벌 에피소드 카운터
        global_ep_lock (mp.Lock): 에피소드 카운터 잠금
        batch_size (int): 배치 크기
        writer_queue (SummaryWriter): Queue를 통해 각 워커 프로세스의 로깅 데이터를 수집
        process_id (int): 프로세스 ID (TensorBoard 로깅 구분용)
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
    sync_interval = 24  # 5 에피소드마다 글로벌 동기화 수행

    for _ in range(n_episodes):
        state = env.reset()
        # if hasattr(env, 'np_random'):
        #     env.np_random.seed(random.randint(0, 10000))

        # state = env._get_observation()  # 현재 상태 얻기
        episode_reward = 0  # 에피소드 동안 얻은 총 보상
        step_count = 0
        episode_losses = []  # 에피소드 동안의 손실 기록
        episode_entropies = []  # 에피소드 동안의 엔트로피 기록
        actions_taken = []  # 액션 분포 추적

        while True:
            action, action_probs = local_agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
        
            # 보상을 누적
            actions_taken.append(action)
            episode_reward += reward
            step_count += 1

            # 데이터를 배치에 저장
            batch.append((state, action, reward, done, next_state))

            # 배치 크기만큼 데이터가 모이면 로컬 업데이트 수행
            if len(batch) >= batch_size:
                # 로컬 에이전트 업데이트 및 손실 계산
                loss, entropy = local_agent.update_batch(batch)
                episode_losses.append(loss)
                episode_entropies.append(entropy)
                # 글로벌 에이전트로 동기화
                sync_local_to_global(global_agent, local_agent) # 로컬 -> 글로벌
                # local_agent.model.load_state_dict(global_agent.model.state_dict())  # 글로벌 -> 로컬 동기화 보장
                batch = []  # 배치 초기화

            state = next_state
            if done:
                break
        
        # 에피소드 기록
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        avg_entropy = np.mean(episode_entropies) if episode_entropies else 0
        action_distribution = np.bincount(actions_taken, minlength=env.action_space.n) / len(actions_taken)

        # 이 블록 안에 있는 코드는 다른 프로세스에서 동시에 실행되지 않도록 보장(lock)
        with global_ep_lock:
            global_ep.value += 1
            log_manager.logger.info(f"Episode {global_ep.value} completed with reward: {episode_reward}")
            # sync_local_to_global(global_agent, local_agent)
            # TensorBoard에 로그 추가
            current_ep = global_ep.value
            
            writer_queue.put({
                'process_id': process_id,
                'episode': current_ep,
                'reward': episode_reward,
                'steps': step_count,
                'loss': avg_loss,
                'entropy': avg_entropy,
                'action_distribution': action_distribution
            })

            if global_ep.value % sync_interval == 0:
                sync_local_to_global(global_agent, local_agent) # 로컬 -> 글로벌
        
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
        global_agent (A3CAgent): 초기화된 글로벌 A3C 에이전트.
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
    writer_queue = mp.Queue()
    stop_event = mp.Event()

    # 현재 시간을 포함한 로그 디렉토리 생성
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join("utils/Log/tensorboard_logs", f"sp500_training_{current_time}")
    os.makedirs(log_dir, exist_ok=True)

    # 로깅 프로세스 시작 (글로벌 에이전트 전달)
    logging_proc = mp.Process(
        target=logging_process, 
        args=(writer_queue, stop_event, log_dir, global_agent, data_path, n_processes, n_episodes, batch_size)
    )
    logging_proc.start()

    processes = []
    episodes_per_process = n_episodes // n_processes
    for process_num in range(n_processes):
        # 올바른 순서
        p = mp.Process(
            target=worker,
            args=(global_agent, data_path, episodes_per_process, global_ep, 
                global_ep_lock, writer_queue, process_num, batch_size)
        )
        p.start()
        processes.append(p)

    try:
        for p in processes:
            p.join()
        stop_event.set()
        logging_proc.join()
    except Exception as e:
        log_manager.logger.error(f"Error during process termination: {e}")


def save_trained_model(global_agent, model_path):
    """
    학습된 모델을 지정된 경로에 저장합니다.

    Args:
        global_agent (A3CAgent): 학습된 글로벌 A3C 에이전트.
        model_path (str): 모델을 저장할 경로.
    """
    global_agent.save_model(model_path)

def evaluate_global_agent(global_agent, env, n_episodes=5):
    """
    글로벌 에이전트의 성능을 평가
    """
    total_rewards = []
    total_steps = []
    action_counts = []
    
    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        actions = []
        
        while True:
            action, action_probs = global_agent.select_action(state)
            actions.append(action)
            next_state, reward, done, _ = env.step(action)
            
            episode_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        total_steps.append(steps)
        action_counts.append(actions)
    
    # 평균 계산
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)
    all_actions = np.concatenate(action_counts)
    action_distribution = np.bincount(all_actions, minlength=env.action_space.n) / len(all_actions)
    
    return {
        'reward': avg_reward,
        'steps': avg_steps,
        'action_distribution': action_distribution
    }

def logging_process(writer_queue, stop_event, log_dir, global_agent, data_path, n_processes, n_episodes, batch_size):
    """
    별도의 프로세스에서 TensorBoard 로깅을 처리하며 글로벌 에이전트도 평가
    """
    writer = SummaryWriter(log_dir)
    
    # 글로벌 에이전트 평가를 위한 환경 생성
    df = pd.read_csv(data_path)
    eval_env = StockTradingEnv(df)
    
    # 글로벌 에이전트 평가 주기 설정
    eval_interval = 10  # 10 에피소드마다 글로벌 에이전트 평가
    last_eval_episode = 0
    writer.add_text('Hyperparameters', f"Processes: {n_processes}, Episodes: {n_episodes}, Batch Size: {batch_size}")

    while not stop_event.is_set() or not writer_queue.empty():
        try:
            data = writer_queue.get(timeout=1.0)
            current_episode = data['episode']
            
            # 로컬 에이전트 메트릭 로깅
            writer.add_scalar(f'Local/Process_{data["process_id"]}/Reward', data['reward'], current_episode)
            writer.add_scalar(f'Local/Process_{data["process_id"]}/Steps', data['steps'], current_episode)
            writer.add_scalar(f'Local/Process_{data["process_id"]}/Loss', data['loss'], current_episode)
            writer.add_scalar(f'Local/Process_{data["process_id"]}/Entropy', data['entropy'], current_episode)
            
            # 글로벌 에이전트 주기적 평가
            if current_episode - last_eval_episode >= eval_interval:
                global_metrics = evaluate_global_agent(global_agent, eval_env)
                
                # 글로벌 에이전트 메트릭 로깅
                writer.add_scalar('Global/Average_Reward', global_metrics['reward'], current_episode)
                writer.add_scalar('Global/Average_Steps', global_metrics['steps'], current_episode)
                
                last_eval_episode = current_episode
            
        except queue.Empty:
            continue
    
    writer.close()


# TensorBoard 로그 초기화
def initialize_tensorboard(log_dir):
    """
    TensorBoard SummaryWriter를 초기화합니다.

    Args:
        log_dir (str): TensorBoard 로그 파일을 저장할 디렉터리 경로.

    Returns:
        SummaryWriter: TensorBoard writer 객체.
    """
    return SummaryWriter(log_dir)

if __name__ == '__main__':
    import os

    set_seeds()

    data_path = 'data/data_csv/sp500_training_data.csv'
    model_path = 'output/sp500_trading_model_1024.pth'

    # 환경과 에이전트 초기화
    global_agent = initialize_environment_and_agent(data_path)

    config = ConfigLoader()
    n_processes = config.get_n_processes()
    n_episodes = config.get_n_episodes()
    batch_size = config.get_batch_size()

    # TensorBoard 초기화
    # log_dir = "utils/Log/tensorboard_logs/sp500_training"
    # os.makedirs(log_dir, exist_ok=True)
    # writer = initialize_tensorboard(log_dir)

    # 학습 프로세스 시작
    start_training(
        global_agent, 
        data_path,
        n_processes=config.get_n_processes(),
        n_episodes=config.get_n_episodes(),
        batch_size=config.get_batch_size()
    )
    # 학습된 모델 저장
    save_trained_model(global_agent, model_path)

    # TensorBoard writer 닫기
    # writer.close()

    log_manager.logger.info("Training process completed")
