from .definition import *
from .mcts_algorithm import *
from .minimax_algorithm import *
from .opening_moves import *
from .constant import *
from .heuristic import *
import json
from random import randint
import os

class MatchStatistic:
    def __init__(self):
        self.nodes_expanded:list[int] = []
        self.hits:list[int] = []
        self.time_taken:list[float] = []
        self.simulations:list[int] = []
        self.scores_diff:list[int] = []
    def dump(self):
        stats = {
            'nodes': self.nodes_expanded,
            'time_taken':self.time_taken,
            'simulations': self.simulations,
            'score_diff': self.scores_diff,
        }
        print(stats)
        #json.dump(stats, os.open(f'stats_{randint(0,10000000)}.json',flags=))


class StrategyController:
    
    def __init__(self):
        self.strategies:list[Strategy] = []
        self.strategy_count = 0
        self.match_stats = MatchStatistic()

    def dump_stats(self):
        self.match_stats.dump()
        
    def __call__(self, *args, **kwds)->LightAction:
        return self._compute_action(*args,**kwds)

    def _compute_action(self) -> LightAction:
        moves_index = Strategy.my_step
        strategy = self[moves_index]

        # Get stats
        best_action = strategy.search()
        if isinstance(strategy,MinimaxTypeASearch):
            self.match_stats.nodes_expanded.append(strategy.node_expanded)
            self.match_stats.hits.append(strategy.hit)
            self.match_stats.time_taken.append(strategy.time_taken)
        
        if isinstance(strategy,MCTSSearch):
            self.match_stats.simulations.append(strategy.n_simulation)
            self.match_stats.time_taken.append(strategy.time_taken)
        
        self.match_stats.scores_diff.append(strategy.score_diff)
    
        return best_action
        
    def add_strategy(self,strategy:Strategy | type[Strategy],number_of_moves:int | None = None):
        # NOTE add **kwards if we want to another way to create a strategy
        try:
            if isinstance(strategy,type):
                strategy = strategy()
        except:
            raise KeyError('This Strategy needs args, define it before passing to the class')

        if number_of_moves == None or number_of_moves > MAX_MOVES- len(self.strategies):
            number_of_moves = MAX_MOVES - len(self.strategies)

        self.strategies.extend([strategy for _  in range(number_of_moves)])
        return self
        
    def __getitem__(self,move_index) -> Strategy:
        return self.strategies[move_index]
    
    def _setitem__(self,move_index,strategy:Strategy):
        self.strategies[move_index] = strategy

    def strategy_from_dict(self, strategy:dict[int,Strategy],clear=False):
        if clear:
            self.strategies.clear()

        for move_step,algo in strategy.items():
            self.add_strategy(move_step,algo)
    
    def to_json(self):
        return {}
    
    
    def __del__(self):
        self.dump_stats()
############################################### PREDEFINED STRATEGY ##############################################
