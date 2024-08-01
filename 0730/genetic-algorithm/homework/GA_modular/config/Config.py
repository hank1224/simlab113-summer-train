class GeneticAlgorithmConfig:
    POP_SIZE = 500  # 種群大小
    GENERATIONS = 100  # 世代數


class CrossoverConfig:
    CROSSOVER_TYPE = "uniform"  # "uniform", "blx_alpha", "linear_interpolation

    # 模糊交叉(Blend Crossover, BLX-α)
    BLX_ALPHA = 0.5


class MutationConfig:
    MUTATION_TYPE = "gaussian"  # "uniform", "gaussian", "non-uniform"

    MUTATION_RATE = 0.6  # 變異率
    MUTATION_DELTA = 0.5  # 變異幅度


class SelectionConfig:
    SELECTION_STRATEGY_TYPE = "roulette"  # "roulette", "tournament", "rank_based"

    # 輪盤選擇(roulette)

    # 錦標賽選擇(tournament)
    TOURNAMENT_SIZE = 5

    # 排序選擇(rank-based)
