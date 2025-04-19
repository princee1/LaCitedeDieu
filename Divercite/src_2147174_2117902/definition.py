from abc import abstractmethod
from typing import Callable, Literal, Any
from .tools import Time
from game_state_divercite import GameStateDivercite
import numpy as np
from seahorse.game.light_action import LightAction
from .constant import *
from .helper import *
from typing import Generator
from game_state_divercite import GameStateDivercite
from cachetools import LRUCache,Cache
from gc import collect
from seahorse.utils.custom_exceptions import ActionNotPermittedError
from enum import Enum
from inspect import signature
#from scipy.stats import norm
from numpy.random import normal,uniform
from time import time


L = 4.1

class Optimization(Enum):
    MAXIMIZE = 1
    MINIMIZE = -1


ARGS_KEYS= Literal['opponent_score','my_score','last_move','my_pieces','opponent_pieces','moves','is_first_to_play','my_id','opponent_id','current_env']
NormalizationType = Literal['range_scaling','sigmoid','tanh']
LossFunction = Literal['potential','evolution','evolution_no_cross_diff','raw_eval','diff','raw_eval_opp','dispersion','delta','opp_delta']
DistributionType = Literal['random','gaussian','uniform']

############################################  Exception class  #############################################

class AlphaOutOfRangeException(Exception):
    def __init__(self,):
        super().__init__('Make sure the alpha is within [0,1] !')

class MissingOrderingHeuristicException(Exception):
    def __init__(self,):
        super().__init__('Missing the ordering heuristic for the non-distribution type! ')

class MissingStdException(Exception):
    def __init__(self,):
        super().__init__('Missing standard deviation for the Gaussian Distribution !')

class TimeConvertException(Exception):
    def __init__(self, message):
        super().__init__(message)

class NegativeOrNullTimeException(Exception):

    def __init__(self, time_provided:int,):    
        super().__init__(f"The time returned cannot be {'null' if  time_provided == 0 else 'negative'}: {time_provided}")
       
            
class OptimizationTypeNotPermittedException(Exception):
    def __init__(self,class_name:str,loss_func:LossFunction):
        super().__init__(f'Optimization Type ({loss_func}) not permitted in {class_name}')

class ActionNotFoundException(Exception):
    def __init__(self,my_step:int,step:int, search_type:str):
        super().__init__('Action not found')

############################################ Base Heuristic class  #############################################

class Heuristic:
    def evaluate(self, current_state: GameStateDivercite,**kwargs) -> Any:
        ...

    def __call__(self, *args, **kwds) -> LightAction | float:
        return self.evaluate(*args, **kwds)


class AlgorithmHeuristic(Heuristic):
    
    #NOTE needs to be same nam
    def __init__(self,normalization_type:NormalizationType,loss_func:LossFunction, min_value: float, max_value: float,L=L,weight=1,heuristic_list=None,optimization = Optimization.MAXIMIZE):
        self.heuristic_list: list[AlgorithmHeuristic] = [self] if heuristic_list ==None else heuristic_list
        self.min_value = min_value
        self.max_value = max_value
        self.weight = weight
        self.total_weight=weight
        self.L = L
        self.optimization = optimization
        self.normalization_type:NormalizationType = normalization_type
        self.loss_func:LossFunction = loss_func
        self.k = 1/(.60*max_value+1)


    def __call__(self, *args, **kwds) -> float: 
        if len(self.heuristic_list) == 1:
            return self.evaluate(*args,**kwds)
        
        vals = [hrstc.weight * hrstc.evaluate(*args,**kwds) for hrstc in self.heuristic_list]
        return sum(vals)/self.total_weight

    def evaluate(self, current_state: GameStateDivercite,**kwargs) -> float:
        state_eval = self._evaluation(current_state,**kwargs)
        return self.normalize(state_eval)

    def _evaluation(self,current_state:GameStateDivercite,**kwargs)-> float:
        ...

    def _sigmoid(self, x: float,min_value,max_value,L=L):
        x_scaled = (x - (min_value + max_value) / 2) / \
            ((max_value - min_value) / 2) * L
        return 2 / (1 + np.exp(-x_scaled)) - 1

    def _tanh(self,x:float):
        return float(np.tanh(self.k*x))
    
    def normalize(self,x:float):
        if self.normalization_type == 'sigmoid':
            return self._sigmoid(x,self.min_value,self.max_value,self.L)

        if self.normalization_type == 'tanh':
            return self._tanh(x)

        return self._range_scaling(x,self.min_value,self.max_value)
        
    def _range_scaling(self,value, min_value, max_value):
        if min_value == max_value:
            raise ValueError("min_value and max_value must be different for scaling.")
        
        normalized_value = (value - min_value) / (max_value - min_value)
        scaled_value = 2 * normalized_value - 1
        return scaled_value
        
    def __mul__(self,weight):
        # changing the weight or the heuristic
        return self._clone(weight)

    def __truediv__(self,weight):
        # changing the weight or the heuristic

        return self._clone(weight)
    
    def _clone(self, weight):
        # Cloning the heuristic
        temp_args = self._compute_added_args()
        clone = self.__class__(**temp_args)
        clone.weight = weight
        clone.total_weight = weight
        #clone.h_list = self.h_list
        return clone
 
    def _compute_added_args(self):
        temp_args = self.__dict__.copy()
        for param in signature(AlgorithmHeuristic.__init__).parameters.values():
            if param.name == 'self':
                continue
            del temp_args[param.name]
            
        del temp_args['total_weight']
        del temp_args['k']
        return temp_args
    
    def __add__(self, other):
        other:AlgorithmHeuristic = other
        total_weight =self.total_weight + other.weight
        temp = self.heuristic_list + other.heuristic_list
        clone = AlgorithmHeuristic(None,None,1,1,heuristic_list=temp)
        clone.total_weight = total_weight
        return clone
    
    def __repr__(self):
        if len(self.heuristic_list) <= 1:
            return f'{self.__class__.__name__}(weight = {self.weight}, loss_func: {self.loss_func}, optimization = {self.optimization.name})'
        return f'{self.__class__.__name__}:{self.total_weight} - {self.heuristic_list}'
    
    @staticmethod
    def _maximize_score_diff(my_current,opp_current,my_state,opp_state,optimization:Optimization,cross_diff =True):
        my_delta_score = my_state - my_current
        delta_state = my_state - opp_state
        cross_diff = (my_state - opp_current) - \
            (opp_state-my_current) if cross_diff else 0

        return optimization.value*(my_delta_score+delta_state+cross_diff)
    @staticmethod
    def _maximized_potential(opp_state,my_state,optimization:Optimization,):
        return optimization.value*((my_state - opp_state) + my_state)
    
    @staticmethod
    def compute_optimization(my_current_score,opponent_current_score,my_state_score,opponent_state_score,loss_func:LossFunction,optimization:Optimization)->float | int:
        match loss_func:
            case 'evolution_no_cross_diff':
                return AlgorithmHeuristic._maximize_score_diff(my_current_score, opponent_current_score, my_state_score, opponent_state_score,optimization,False) 
            case 'evolution':
                        # BUG Might want to revisit the way we quantify this heuristic and remove the cross diff
                return AlgorithmHeuristic._maximize_score_diff(my_current_score, opponent_current_score, my_state_score, opponent_state_score,optimization) 
            case 'potential':
                return AlgorithmHeuristic._maximized_potential(opponent_state_score,my_state_score,optimization)

            case 'diff':
                return optimization.value*(my_state_score - opponent_state_score)

            case 'raw_eval':
                return optimization.value*my_state_score

            case 'raw_eval_opp':
                return optimization.value*-1 *opponent_state_score

            case 'delta':
                return optimization.value*(my_state_score-my_current_score)

            case 'dispersion':
                return optimization.value*(my_state_score-opponent_current_score)
            
            case 'opp_delta':
                return optimization.value*-1*(opponent_state_score-opponent_current_score)

############################################# Base Strategy Classes ##############################################

class Strategy:

    # Meta Data de l'État actuel
    is_first_to_play: bool = None
    current_state: GameStateDivercite = None
    my_id: int = None
    opponent_id: int = None
    remaining_time: float = None
    my_step: int = -1  # [0,19]
    my_piece_type:str=None
    opponent_piece_type:str=None
                
    @staticmethod
    def set_current_state(current_state: GameStateDivercite, remaining_time: float):
        Strategy.current_state = current_state
        Strategy.remaining_time = remaining_time
        Strategy.my_step += 1
        
        if Strategy.my_step == 0:
            Strategy.init_game_state()

    @staticmethod
    def init_game_state():
        Strategy.opponent_id = Strategy.current_state.compute_next_player().id
        temp = [player.id for player in Strategy.current_state.players]
        temp.remove(Strategy.opponent_id)
        Strategy.my_id = temp[0]
        Strategy.is_first_to_play = Strategy.current_state.step == 0
        Strategy.my_piece_type = W_PIECE if Strategy.is_first_to_play else B_PIECE
        Strategy.opponent_piece_type = list(PIECE_TYPE.difference([Strategy.my_piece_type]))[0]

    def __init__(self, heuristic: Heuristic|AlgorithmHeuristic = None):
        self.main_heuristic = heuristic

    @staticmethod
    def greedy_fallback_move():
        '''
        Code taken from the template
        '''

        possible_actions = Strategy.current_state.get_possible_light_actions().copy()
        best_action = possible_actions[0]
        best_score = Strategy.current_state.scores[Strategy.my_id]

        for action in possible_actions:
            state = Strategy.current_state.apply_action(action)
            score = state.scores[Strategy.my_id]
            if score > best_score:
                best_action = action
        
        return best_action

    def _search(self) -> LightAction:
        ...

    def search(self)-> LightAction:
        collect()
        try:
            return self._search()
        
        except ActionNotPermittedError as e:
            print(e.__class__.__name__,f': {e.args}')
        
        except ActionNotFoundException as e:
            print(e.__class__.__name__,f': {e.args}')

        except Exception as e:
            print('Warning... !:',e.__class__.__name__,f': {e.args}')
        
        try:
            return Strategy.greedy_fallback_move()
        except:
            print('Error in fallback moves! Testing Last resort',)
        
        return choice(list(self.current_state.get_possible_light_actions().copy()))

    @property
    def my_pieces(self):
        return self.current_state.players_pieces_left[self.my_id]
    
    @property
    def score_diff(self):
        return self.my_score - self.opponent_score    

    @property
    def opponent_pieces(self):
        return self.current_state.players_pieces_left[self.opponent_id]

    @property
    def my_score(self):
        return self.current_state.scores[self.my_id]

    @property
    def opponent_score(self):
        return self.current_state.scores[self.opponent_id]

    @property
    def last_move(self):
        temp =list(self.current_state.rep.env)
        if not temp:
            return None
        return temp[-1]

    @property
    def moves(self):
        return list(self.current_state.rep.env)
    
    @property
    def current_env(self):
        return self.current_state.rep.env


class Algorithm(Strategy):

    def __init__(self,utility_type:LossFunction,heuristic: AlgorithmHeuristic, cache: int | Cache | None=None, allowed_time: float = None,keep_cache: bool = False,skip_symmetric=True):
        super().__init__(heuristic)
        if isinstance(cache,Cache):
            self.cache = cache
        else:
            self.cache = None if cache == None else LRUCache(cache)
        self.allowed_time = allowed_time
        self.keep_cache = keep_cache # ERROR can be source of heuristic evaluation error, only uses if a deeper search was done prior a less deeper search
        self.utility_type:LossFunction = utility_type
        self.skip_symmetric = skip_symmetric
        self.hit = 0
        self.node_expanded = 0
        self.time_taken:float = -1

        
    def _utility(self, state: GameStateDivercite) ->float:        
        state_scores = state.get_scores()
        my_state_score = state_scores[self.my_id]
        opponent_state_score = state_scores[self.opponent_id]

        current_scores = self.current_state.get_scores()
        my_current_score = current_scores[self.my_id]
        opponent_current_score = current_scores[self.opponent_id]

        return AlgorithmHeuristic.compute_optimization(my_current_score, opponent_current_score,my_state_score,opponent_state_score,self.utility_type,Optimization.MAXIMIZE)
        
    def _pred_utility(self,state:GameStateDivercite)->float:
        ...

    def _is_our_turn(self, step = None):
        if step ==None:
            step = self.current_state.step 

        if self.is_first_to_play and step % 2 == 0:
            return True

        if not self.is_first_to_play and step % 2 == 1:
            return True

        return False

    def _transition(self, state: GameStateDivercite, action) -> GameStateDivercite:
        return state.apply_action(action)
    
    def rotate_moves_90(self,moves:dict[tuple[int,int],Any]):
        temp_moves = {}
        for pos, pieces in moves.items():
            x,y = pos
            pos = rotate_position_90_clockwise(x,y)
            temp_moves[pos] = pieces
        return temp_moves

    def _hash_state(self, state_env: dict) -> int:
        temp_env ={pos:piece.piece_type for pos, piece in state_env.items()}
        return frozenset(temp_env.items())
    
    def check_symmetric_moves_in_cache(self, state_env:dict) -> tuple[bool, None | frozenset]:
        
        if self.skip_symmetric:
            temp_env = state_env.copy()
            for _ in range(3):
                temp_env= self.rotate_moves_90(temp_env) 
                temp_env_hash = self._hash_state(temp_env)
                if temp_env_hash in self.cache:
                    return True, temp_env_hash              
        return False, None
    
    def _clear_cache(self):
        try:
            if self.cache != None and not self.keep_cache:
                self.cache.clear()
        except AttributeError:
            print('Warning: Trying to clear cache when None is provided')
        except:
            ...
        
    def search(self):
        # NOTE See comments in line 170
        start_time = time()
        self._clear_cache()
        best_action = super().search()
        self.time_taken = time()- start_time
        return best_action

############################################## Interface ###########################################

class ActionOrderInterface():

    def __init__(self, order_heuristic:AlgorithmHeuristic):
        self.order_heuristic = order_heuristic

    def _order_actions(self, actions: Generator | list, current_state: GameStateDivercite,**kwargs) -> tuple[np.ndarray,np.ndarray]:
        def _apply(a):
            return self.order_heuristic(current_state.apply_action(a[0]), **kwargs)
        
        return self._iter_state(actions,_apply)


    def _iter_state(self, _iterator, apply_func:Callable):
        _iterator_compute = np.fromiter(_iterator, dtype=np.object_)
        vals = np.apply_along_axis(
            apply_func, axis=0, arr=_iterator_compute.reshape(1, -1))
        return vals,_iterator_compute

class StochasticActionInterface(ActionOrderInterface):
    
    def __init__(self, ordering_heuristic,distribution_type:DistributionType,std:float=None):
        super().__init__(ordering_heuristic)
        self.distribution_type:DistributionType = distribution_type
        self.std =std
        if self.std == None and distribution_type =='gaussian':
            raise MissingStdException()
        if self.order_heuristic == None and distribution_type != 'random':
            raise MissingOrderingHeuristicException()
        
    def _simulate(self, state: GameStateDivercite)->float:
        current_state = state
        while not current_state.is_done():
            action = self._compute_action(current_state)
            current_state = self._transition(current_state, action)

        return self._utility(current_state)
    

    def _compute_action(self,state:GameStateDivercite):
        actions_ = list(state.get_possible_light_actions())
        if self.distribution_type == 'random':
            return choice(actions_)
        
        vals, returned_actions = self._order_actions(actions_,state,my_id=self.my_id, opponent_id=self.opponent_id,
                                        my_pieces=self.my_pieces, opponent_pieces=self.opponent_pieces,
                                        last_move=self.last_move, is_first_to_play=self.is_first_to_play, moves=self.moves,
                                        my_score=self.my_score, opponent_score=self.opponent_score, current_env=self.current_env,
                                        my_piece_type=self.my_piece_type, opponent_piece_type=self.opponent_piece_type)
    
        moves_index  = self._stochastic_selection(vals)
        return returned_actions[moves_index]
    
    def _stochastic_selection(self,vals:np.ndarray):
        mean = vals.mean()
        if self.distribution_type == 'gaussian':
            target = normal(mean,self.std,1)
        if self.distribution_type == 'uniform':   
            target = uniform(0,1,1)

        target = float(target[0])
        
        diff = float('inf')
        selected_index = None

        for index,action_eval in enumerate(vals):
            if np.abs(action_eval-target) < diff:
                selected_index = index
                diff = np.abs(action_eval-target)
        return selected_index