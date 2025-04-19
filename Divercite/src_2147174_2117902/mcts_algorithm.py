from dataclasses import dataclass
from typing import Any

from cachetools import Cache
from .minimax_algorithm import MinimaxTypeASearch
from game_state_divercite import GameStateDivercite
from .definition import ActionNotFoundException, Algorithm, AlgorithmHeuristic, LossFunction, TimeConvertException, NegativeOrNullTimeException,StochasticActionInterface,DistributionType,AlphaOutOfRangeException
from seahorse.game.light_action import LightAction
from random import choice
from time import time
from pytimeparse import parse
from .constant import MAX_MOVES, MAX_STEP
import math
import numpy as np

# @dataclass
class Node:

    def __init__(self, state: GameStateDivercite,action_taken:LightAction=None, parent=None):
        self.value = 0
        self.visit = 0
        self.action_taken = action_taken
        self.parent: Node = parent
        self.children: list[Node] = []
        self.state: GameStateDivercite = state
        self.max_children: int = len(self.state.get_possible_light_actions())

    @property
    def is_fully_expanded(self):
        return len(self.children) >= self.max_children

    def _compute_uct(self,Q,C):
        return Q + C*(math.log10(self.parent.visit/self.visit)**0.5)

    def UCB1(self,C) -> float:
        Q = self.value/self.visit
        return self._compute_uct(Q,C)
    
    def UCB1_Minimax(self,C,alpha,v0)->float:
        Q = (1-alpha)*self.value/self.visit + alpha*v0
        return self._compute_uct(Q,C)


class MCTSSearch(Algorithm,StochasticActionInterface):
    def __init__(self, typeB_heuristic:AlgorithmHeuristic,allowed_time: str, distribution_type:DistributionType,C:float,std:float=None,n_playouts:int=1,cache: int | Cache = 5000,skip_symmetric=True):
        allowed_time = self._convert_to_seconds(allowed_time)
        super().__init__('diff', typeB_heuristic, cache, allowed_time,skip_symmetric=skip_symmetric)
        self.n_simulation = 0
        StochasticActionInterface.__init__(self,typeB_heuristic,distribution_type,std)
        self.n_playouts = n_playouts
        self.C = C
        self.ucb_flag = False
    
    def _pred_utility(self, state:GameStateDivercite):
        total_pred =0
        for _ in range(self.n_playouts):
            total_pred+= 1 if self._simulate(state)>0 else -1
        
        return total_pred/self.n_playouts

    def _search(self) -> LightAction:
        self.n_simulation = 0
        print(f'Allowed time {self.allowed_time} (s)')
        print(f'Game Step: {self.current_state.step}/{MAX_STEP} - My Step: {self.my_step}/{MAX_MOVES} - {self.__class__.__name__} - Distribution: {self.distribution_type}')
        root_node = Node(self.current_state)
        start_time = time()
        self.ucb_flag = False

        while time() - start_time <= self.allowed_time:
            node: Node = self._select(root_node)
            reward = self._pred_utility(node.state)
            self._back_propagate(node, reward)
            self.n_simulation += 1
        self.ucb_flag = True
        print(f'Search Done with {self.n_simulation} simulations')
        # TODO best_child based on some type of score, UCT1,UCTtuned, UCTminimax, 
        return self._best_action(root_node.children).action_taken

    def _select(self,node:Node) -> Node:
        while not node.state.is_done():
            if not node.is_fully_expanded:
                return self._expand(node)
            else:
                node = self._best_action(node.children)
        return node

    def _expand(self,node:Node):
        tried_actions = {child.action_taken for child in node.children if child.action_taken != None }
        legal_actions = node.state.get_possible_light_actions()
        for action in legal_actions:
            if action not in tried_actions:
                next_state = self._transition(node.state,action)
                
                # Use cache to avoid redundant node creation

                if self.cache != None:
                    hash_state = self._hash_state(next_state.rep.env)
                    if hash_state not in self.cache:
                        flag, _ = self.check_symmetric_moves_in_cache(
                            next_state.rep.env)
                        if flag:
                            self.hit +=1
                            continue
                        else:
                            self.cache[hash_state] = Node(next_state, action_taken=action,parent=node)
                            child_node = self.cache[hash_state]        
                            node.children.append(child_node)
                            return child_node

                else :
                    child_node = Node(next_state, action_taken=action,parent=node)
                    node.children.append(child_node)
                    return child_node
                
        
        raise ActionNotFoundException()

    def _back_propagate(self, node: Node, reward: float):
        while node is not None:
            node.visit += 1
            node.value += reward
            node = node.parent

    def _best_action(self,children:list[Node])->Node: 
        vals,_action_state = self._iter_state(children,self._apply)
        best_indices = vals.argmax(axis=0)
        return _action_state[best_indices]
    
    def _apply(self,node_):
        n:Node= node_[0]
        return n.UCB1(self.C)

    def _convert_to_seconds(self, time_: str | int | float):
        try:
            if not isinstance(time_, str):
                return time_
            seconds = parse(time_)
            if seconds is not None:
                if seconds <= 0:
                    raise NegativeOrNullTimeException(seconds)
                return seconds
            raise TimeConvertException(f'Could not parse time string: {time_}')

        except Exception as e:
            raise TimeConvertException(
                f'Error while parsing time string: {time_}')


class MCTSHybridMinimaxBackupsSearch(MCTSSearch):
    
    def __init__(self, typeB_heuristic:AlgorithmHeuristic, allowed_time:str, distribution_type:DistributionType,C:float,alpha:float,minimax_details:dict[str,Any], filter_heuristic:AlgorithmHeuristic=None, n_max_filtered:int=None,std = None, n_playouts = 1, cache = 5000):
        super().__init__(typeB_heuristic, allowed_time, distribution_type,C, std, n_playouts, cache)
        self.filter_heuristic = typeB_heuristic if filter_heuristic == None else filter_heuristic
        self.n_max_filtered = n_max_filtered
        if alpha>1 or alpha<0:
            raise AlphaOutOfRangeException()
        self.alpha = alpha
        self.minimax_details = minimax_details

    def _apply(self, node_):
        n:Node = node_[0]
        if self.ucb_flag:
            return n.UCB1(self.C)
        
        if isinstance(self.minimax_details,dict):
            _t = self.minimax_details['type']
            args = self.minimax_details['args']
            minimax_search:MinimaxTypeASearch =  _t(**args)
        else:
            minimax_search = self.minimax_details

        v0,_ = minimax_search._minimax(n.state,True,float('-inf'), float('inf'), 0, minimax_search.max_depth)
        return n.UCB1_Minimax(self.C,self.alpha,v0)
    
    def _best_action(self, children):
        # GEt the best action
        if self.n_max_filtered != None:
            children =self._filter_children(children)
        return super()._best_action(children)
    
    def _filter_children(self, children):
        # Filter the children by an heuristic evaluation
        def _apply(node_):
            n:Node = node_[0]
            return self.filter_heuristic(n.state,my_id=self.my_id, opponent_id=self.opponent_id,
                                        my_pieces=self.my_pieces, opponent_pieces=self.opponent_pieces,
                                        last_move=self.last_move, is_first_to_play=self.is_first_to_play, moves=self.moves,
                                        my_score=self.my_score, opponent_score=self.opponent_score, current_env=self.current_env,
                                        my_piece_type=self.my_piece_type, opponent_piece_type=self.opponent_piece_type)

        vals,children_iter = self._iter_state(children,_apply)
        n_max = self.n_max_filtered if self.n_max_filtered >=1 else math.ceil(self.n_max_filtered*len(vals))
        vals_indice = vals.argsort(axis=0)[::-1][:n_max]
        return children_iter[vals_indice]
    