import yaml
from functools import cached_property

class ConfigLoader:
    DEFAULT_CONFIG_PATH = "config\\config.yaml"  # 고정된 config 경로

    @classmethod
    def _load_config(cls):
        """
        YAML 파일을 로드하여 설정 값을 반환
        """
        with open(cls.DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    @classmethod
    def get_env(cls):
        return cls._load_config().get("env", {})

    @classmethod
    def get_a3c_agent(cls):
        return cls._load_config().get("A3CAgent", {})

    @classmethod
    def get_actor_critic(cls):
        return cls._load_config().get("ActorCritic", {})

    @classmethod
    def get_training(cls):
        return cls._load_config().get("training", {})

    @classmethod
    def get_seeds(cls):
        return cls._load_config().get("seeds", {})

    @classmethod
    def get_logger(cls):
        return cls._load_config().get("Logger", {})

    # 각각의 env 값을 출력하는 함수들
    @classmethod
    def get_cash_in_hand(cls):
        return cls.get_env().get("cash_in_hand", None)

    @classmethod
    def get_max_stock(cls):
        return cls.get_env().get("max_stock", None)

    @classmethod
    def get_trading_charge(cls):
        return cls.get_env().get("trading_charge", None)

    @classmethod
    def get_trading_tax(cls):
        return cls.get_env().get("trading_tax", None)

    # 각각의 training 값을 출력하는 함수들
    @classmethod
    def get_n_processes(cls):
        return cls.get_training().get("n_processes", None)

    @classmethod
    def get_n_episodes(cls):
        return cls.get_training().get("n_episodes", None)

    @classmethod
    def get_batch_size(cls):
        return cls.get_training().get("batch_size", None)

    # ActorCritic의 size 값을 출력하는 함수
    @classmethod
    def get_hidden_layer_size(cls):
        return cls.get_actor_critic().get("hidden_layer", {}).get("size", None)

    # A3CAgent의 Learning Rate 값을 출력하는 함수
    @classmethod
    def get_learning_rate(cls):
        return cls.get_a3c_agent().get("Learning Rate", None)

    # 각각의 seeds 값을 출력하는 함수들
    @classmethod
    def get_random_seed(cls):
        return cls.get_seeds().get("random", None)

    @classmethod
    def get_numpy_seed(cls):
        return cls.get_seeds().get("numpy", None)

    @classmethod
    def get_torch_seed(cls):
        return cls.get_seeds().get("torch", None)

# Example usage
if __name__ == "__main__":
    # Environment values
    print("Cash in Hand:", ConfigLoader.get_cash_in_hand())
    print("Max Stock:", ConfigLoader.get_max_stock())
    print("Trading Charge:", ConfigLoader.get_trading_charge())
    print("Trading Tax:", ConfigLoader.get_trading_tax())

    # Training values
    print("Number of Processes:", ConfigLoader.get_n_processes())
    print("Number of Episodes:", ConfigLoader.get_n_episodes())
    print("Batch Size:", ConfigLoader.get_batch_size())

    # ActorCritic values
    print("Hidden Layer Size:", ConfigLoader.get_hidden_layer_size())

    # A3CAgent values
    print("Learning Rate:", ConfigLoader.get_learning_rate())

    # Seeds values
    print("Random Seed:", ConfigLoader.get_random_seed())
    print("Numpy Seed:", ConfigLoader.get_numpy_seed())
    print("Torch Seed:", ConfigLoader.get_torch_seed())

    # Other configurations
    print("A3C Agent Configuration:", ConfigLoader.get_a3c_agent())
    print("Logger Configuration:", ConfigLoader.get_logger())
