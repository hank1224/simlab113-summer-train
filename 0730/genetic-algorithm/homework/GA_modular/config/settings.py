class GeneticAlgorithmConfig:
    POP_SIZE = 5000  # 種群大小
    GENERATIONS = 100  # 世代數

    @classmethod
    def get_pop_size(cls):
        return cls.POP_SIZE

    @classmethod
    def get_generations(cls):
        return cls.GENERATIONS


class CrossoverConfig:
    CROSSOVER_TYPE = "blx_alpha"  # "uniform", "blx_alpha", "linear_interpolation"
    BLX_ALPHA = 0.5  # 模糊交叉(Blend Crossover, BLX-α)

    @classmethod
    def get_crossover_type(cls):
        return cls.CROSSOVER_TYPE

    @classmethod
    def get_blx_alpha(cls):
        return cls.BLX_ALPHA


class MutationConfig:
    MUTATION_TYPE = "uniform"  # "uniform", "gaussian", "non-uniform"
    MUTATION_RATE = 0.6  # 變異率
    MUTATION_DELTA = 0.5  # 變異幅度

    @classmethod
    def get_mutation_type(cls):
        return cls.MUTATION_TYPE

    @classmethod
    def get_mutation_rate(cls):
        return cls.MUTATION_RATE

    @classmethod
    def get_mutation_delta(cls):
        return cls.MUTATION_DELTA


class SelectionConfig:
    SELECTION_STRATEGY_TYPE = "tournament"  # "roulette", "tournament", "rank_based"
    TOURNAMENT_SIZE = 5  # 錦標賽選擇(tournament)

    @classmethod
    def get_selection_strategy_type(cls):
        return cls.SELECTION_STRATEGY_TYPE

    @classmethod
    def get_tournament_size(cls):
        return cls.TOURNAMENT_SIZE
