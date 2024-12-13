"""
이 파일은 YAML 설정 파일(config.yaml)을 로드하고, 각종 설정 값을 제공하는 ConfigLoader 클래스를 정의합니다.
ConfigLoader는 학습 환경, A3C 에이전트, Actor-Critic 네트워크, 학습 설정 및 시드 값 등을 제공합니다.
"""

import yaml

class ConfigLoader:
    """
    YAML 설정 파일을 로드하고 필요한 설정 값을 제공하는 클래스입니다.
    """
    DEFAULT_CONFIG_PATH = "config/config.yaml"  # 기본 설정 파일 경로

    @classmethod
    def _load_config(cls):
        """
        YAML 설정 파일을 로드합니다.

        Returns:
            dict: 파싱된 설정 값.
        """
        with open(cls.DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    @classmethod
    def get_env(cls):
        """환경(env) 관련 설정 값을 반환합니다."""
        return cls._load_config().get("env", {})

    @classmethod
    def get_a3c_agent(cls):
        """A3C 에이전트 관련 설정 값을 반환합니다."""
        return cls._load_config().get("A3CAgent", {})

    @classmethod
    def get_actor_critic(cls):
        """Actor-Critic 네트워크 관련 설정 값을 반환합니다."""
        return cls._load_config().get("ActorCritic", {})

    @classmethod
    def get_training(cls):
        """학습(training) 관련 설정 값을 반환합니다."""
        return cls._load_config().get("training", {})

    @classmethod
    def get_seeds(cls):
        """시드(seed) 관련 설정 값을 반환합니다."""
        return cls._load_config().get("seeds", {})

    @classmethod
    def get_logger(cls):
        """로거(logger) 관련 설정 값을 반환합니다."""
        return cls._load_config().get("Logger", {})

    # 환경 설정값 접근 메서드
    @classmethod
    def get_cash_in_hand(cls):
        """초기 현금(cash in hand) 값을 반환합니다."""
        return cls.get_env().get("cash_in_hand")

    @classmethod
    def get_max_stock(cls):
        """최대 보유 주식 수(max stock) 값을 반환합니다."""
        return cls.get_env().get("max_stock")

    @classmethod
    def get_trading_charge(cls):
        """거래 수수료(trading charge) 값을 반환합니다."""
        return cls.get_env().get("trading_charge")

    @classmethod
    def get_trading_tax(cls):
        """거래 세금(trading tax) 값을 반환합니다."""
        return cls.get_env().get("trading_tax")

    # 학습 설정값 접근 메서드
    @classmethod
    def get_n_processes(cls):
        """학습에 사용할 프로세스 개수(n_processes) 값을 반환합니다."""
        return cls.get_training().get("n_processes")

    @classmethod
    def get_n_episodes(cls):
        """학습 에피소드 수(n_episodes) 값을 반환합니다."""
        return cls.get_training().get("n_episodes")

    @classmethod
    def get_batch_size(cls):
        """배치 크기(batch size) 값을 반환합니다."""
        return cls.get_training().get("batch_size")

    # Actor-Critic 설정값 접근 메서드
    @classmethod
    def get_hidden_layer_size(cls):
        """히든 레이어(hidden layer) 크기 값을 반환합니다."""
        return cls.get_actor_critic().get("hidden_layer", {}).get("size")

    # A3CAgent 설정값 접근 메서드
    @classmethod
    def get_learning_rate(cls):
        """A3C 에이전트의 학습률(learning rate) 값을 반환합니다."""
        return cls.get_a3c_agent().get("Learning Rate")

    # 시드 설정값 접근 메서드
    @classmethod
    def get_random_seed(cls):
        """랜덤(random) 시드 값을 반환합니다."""
        return cls.get_seeds().get("random")

    @classmethod
    def get_numpy_seed(cls):
        """NumPy 시드 값을 반환합니다."""
        return cls.get_seeds().get("numpy")

    @classmethod
    def get_torch_seed(cls):
        """PyTorch 시드 값을 반환합니다."""
        return cls.get_seeds().get("torch")

if __name__ == "__main__":
    # 환경 설정값
    print("Cash in Hand:", ConfigLoader.get_cash_in_hand())
    print("Max Stock:", ConfigLoader.get_max_stock())
    print("Trading Charge:", ConfigLoader.get_trading_charge())
    print("Trading Tax:", ConfigLoader.get_trading_tax())

    # 학습 설정값
    print("Number of Processes:", ConfigLoader.get_n_processes())
    print("Number of Episodes:", ConfigLoader.get_n_episodes())
    print("Batch Size:", ConfigLoader.get_batch_size())

    # Actor-Critic 설정값
    print("Hidden Layer Size:", ConfigLoader.get_hidden_layer_size())

    # A3CAgent 설정값
    print("Learning Rate:", ConfigLoader.get_learning_rate())

    # 시드 설정값
    print("Random Seed:", ConfigLoader.get_random_seed())
    print("Numpy Seed:", ConfigLoader.get_numpy_seed())
    print("Torch Seed:", ConfigLoader.get_torch_seed())

    # 기타 설정값
    print("A3C Agent Configuration:", ConfigLoader.get_a3c_agent())
    print("Logger Configuration:", ConfigLoader.get_logger())
