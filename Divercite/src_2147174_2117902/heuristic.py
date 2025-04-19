from typing import Any
from game_state_divercite import GameStateDivercite
from .definition import AlgorithmHeuristic, Heuristic, ARGS_KEYS, Optimization,NormalizationType, LossFunction,OptimizationTypeNotPermittedException
from .constant import *
from .helper import *
import numpy as np
from random import random


class ScoreHeuristic(AlgorithmHeuristic):

    def __init__(self,normalization_type:NormalizationType='sigmoid',loss_func:LossFunction ='evolution' ):
        
        match loss_func:
            case 'diff':
                min_value,max_value,l = -30,30,10.5
            case 'potential':
                min_value,max_value,l=-42,42,5.5
            case 'raw_eval':
                min_value,max_value,l = -40,40,5.2
            case 'raw_eval_opp':
                min_value,max_value,l=-50,0,4.3
            case 'evolution':
                min_value,max_value,l=-81,81,5.7
            case 'evolution_no_cross_diff':
                min_value,max_value,l=-30,30,4.8
            
            case _:
                raise  OptimizationTypeNotPermittedException(self.__class__.__name__,loss_func)
        
        super().__init__(normalization_type,loss_func,min_value,max_value, L=l)

    def _evaluation(self, current_state, **kwargs):
        my_state_score = current_state.get_scores()[kwargs['my_id']]
        opponent_state_score = current_state.get_scores()[
            kwargs['opponent_id']]

        my_current_score = kwargs['my_score']
        opponent_current_score = kwargs['opponent_score']

        return self.compute_optimization(my_current_score, opponent_current_score,my_state_score,opponent_state_score,self.loss_func,self.optimization)



class ControlIndexHeuristic(AlgorithmHeuristic):

    def __init__(self,normalization_type:NormalizationType='range_scaling',loss_func:LossFunction = 'raw_eval',ctrl_weight=.35,dist_weight=.65,):
        super().__init__(normalization_type,loss_func,-4.3, 4.3,L=5.65)
        self.ctrl_weight = ctrl_weight
        self.dist_weight= dist_weight
        self.total_weight = ctrl_weight + dist_weight

    def _evaluation(self, current_state, **kwargs):
        my_piece_type = kwargs['my_piece_type']
        original_env: dict = kwargs['current_env']
        opp_piece_type = kwargs['opponent_piece_type']
        state_env = current_state.rep.env

        (my_state_moves, my_ic_state), (opp_state_moves, opp_ic_state) = self._filter_moves_control_index(
            current_state.rep.env, my_piece_type)
        (my_current_moves, my_current_ic), (opp_current_moves, opp_current_ic) = self._filter_moves_control_index(
            original_env, my_piece_type)

        control_index = self._maximize_score_diff(
            my_current_ic, opp_current_ic, my_ic_state, opp_ic_state,self.optimization)

        my_current_dist = self._compute_neighborhood_distances(
            my_piece_type, my_current_moves, original_env) # at the original state
        opp_current_dist = self._compute_neighborhood_distances(
            opp_piece_type, opp_current_moves, original_env)  # at the original state

        my_state_dist = self._compute_neighborhood_distances(
            my_piece_type, my_state_moves, state_env)  # at the new state 
        opp_state_dist = self._compute_neighborhood_distances(
            opp_piece_type, opp_state_moves, state_env) # at the new state 

        dist_index = self._maximize_score_diff(
            my_current_dist, opp_current_dist, my_state_dist, opp_state_dist,self.optimization)

        control_index =self._compute_control_index(control_index)
        dist_index = self._compute_distance_index(dist_index)

        return (control_index*self.ctrl_weight + dist_index*self.dist_weight)/self.total_weight  #NOTE we can add weight it depends on which is more important


    def _compute_neighborhood_distances(self, pieces_type: str, added_moves: dict[tuple, str], state_env: dict[tuple, str]):
        score = 0
        index = 0
        for move_pos, pieces in added_moves.items():

            m_x, m_y = move_pos
            c, _ = pieces

            for pos in horizontal_vertical_compute + diagonal_compute:
                index += 1
                in_horizontal = index <= 4

                i, j = pos # GEt the new positions when it is in horizontal or vertical or digonale

                computed_pos = m_x+i, m_y + j
                if not is_in_board(computed_pos): # Skip is not in board
                    continue
                if not computed_pos in state_env:  # check piece neighbours exists
                    continue
                p_color, _, p_owner = state_env[computed_pos].piece_type

                if c == p_color and pieces_type == p_owner:  # same color and same ownership
                    score += (10 if in_horizontal else 15)
                
                elif c != p_color and pieces_type == p_owner:  # diff color but same ownership
                    score += (15 if in_horizontal else 10)

                elif c == p_color and pieces_type != p_owner:  # same color but diff ownership
                    score += (25 if in_horizontal else 20)      
                else:                                           # different ownership and diff_color
                    score += ( 35 if in_horizontal else 30)

        return score

    def _filter_moves_control_index(self, env: dict[tuple, Any], my_piece_type: str):
        my_moves = {}
        opp_moves = {}
        my_ic = 0
        opp_ic = 0

        for pos, piece in env.items():
            piece_type = piece.piece_type
            if is_ressource(piece_type):
                continue
            if piece_type.endswith(my_piece_type):
                my_moves[pos] = piece_type[:2]
                my_ic += control_index(pos)
            else:
                opp_moves[pos] = piece_type[:2]
                opp_ic += control_index(pos)

        return (my_moves, my_ic), (opp_moves, opp_ic)

    def _compute_control_index(self,x:float):
        return self._range_scaling(x,3,52)
    
    def _compute_distance_index(self,x:float):
        return self._range_scaling(x,100,700)

class PiecesVarianceHeuristic(AlgorithmHeuristic):

    def __init__(self,normalization_type:NormalizationType='range_scaling',loss_func:LossFunction='potential', city_weight=.7, ress_weight=.3):
        match loss_func:
            case 'diff':
                min_value,max_value,l = -286,286,7
            case 'potential':
                min_value,max_value,l=-400,400,7
            case 'raw_eval_opp':
                min_value,max_value,l=20,330,5
            case 'raw_eval':
                min_value,max_value,l = -330,-20,5

            case _:
                raise  OptimizationTypeNotPermittedException(self.__class__.__name__,loss_func)
            
        super().__init__(normalization_type,loss_func,min_value, max_value, L=l,optimization=Optimization.MINIMIZE) # URGENT-TODO Recheck the scaling depends on minimize or maximize
        self.city_weight = city_weight
        self.ress_weight = ress_weight

    def _evaluation(self, current_state, **kwargs):
        opponent_pieces = current_state.players_pieces_left[kwargs['opponent_id']]
        my_pieces = current_state.players_pieces_left[kwargs['my_id']]
        #####
        my_state_var = self._pieces_var(my_pieces)
        opp_state_var = self._pieces_var(opponent_pieces)
        
        return self.compute_optimization(None, None,my_state_var,opp_state_var,self.loss_func,self.optimization)
             
    def _pieces_var(self, pieces: dict[str, int]):
        # Computing the variance for each type

        city_val = np.array([pieces[cn] if cn in pieces else 0 for cn in CityNames._member_names_])
        ress_val = np.array([pieces[rn] if rn in pieces else 0 for rn in RessourcesNames._member_names_])
        
        is_tuning = ress_val.sum() > city_val.sum() # Scaling up the city by one when its less than ressources
        if is_tuning:
            city_val +=1

        score_city = float(np.var(city_val))*100
        score_ress = float(np.var(ress_val))*100
        city_penalty = 30 if is_tuning else 20
        ress_penalty = 20 if is_tuning else 30

        score_city = city_penalty if score_city ==0 else score_city
        score_ress = ress_penalty if score_ress ==0 else score_ress

        return score_city + score_ress

    @property
    def _h_tot_weight(self):
        return self.city_weight+self.ress_weight

class DiverciteHeuristic(AlgorithmHeuristic):
    def __init__(self,normalization_type:NormalizationType='sigmoid',loss_func:LossFunction = 'potential'):
        
        match loss_func:
            case 'potential':
                min_value,max_value,l = -22000, 16000, 5
            case 'raw_eval':
                min_value, max_value,l = -8000,4000,5.2
            case 'diff':
                min_value,max_value,l = -20000,12000,5
            # case 'evolution':
            #     min_value,max_value,l = -7500,7500,9.5   
            case _:
                raise  OptimizationTypeNotPermittedException(self.__class__.__name__,loss_func)             

        super().__init__(normalization_type,loss_func,min_value,max_value, L=l)
    
    def get_placed_cities_by_player(self, state: GameStateDivercite, player_symbol: str) -> dict:
        """
        Get all cities placed by a specific player on the board.

        Args:
            state (GameStateDivercite): The current game state.
            player (str): The symbol representing the player (e.g., 'W' for White or 'B' for Black).

        Returns:
            dict[tuple, str]: A dictionary mapping city positions (as tuples) to their respective piece types.
        """
        player_cities = {}
        board = state.get_rep().get_env() 
        # Iterate through the board to find cities belonging to the player  
        for pos, piece in board.items():
            piece_type = piece.get_type()
            if piece_type[1] == 'C' and piece_type[2] == player_symbol:
                player_cities[pos] = piece_type  # Store position as key and piece type as value
        
        return player_cities

    def get_total_pieces_on_board(self, state: GameStateDivercite) -> int:
        """
        Get the total number of placed pieces (non-empty) on the board.

        Args:
            state (GameStateDivercite): The current game state.

        Returns:
            int: The total number of pieces on the board.
        """
        board = state.get_rep().get_env()
        return len([piece for piece in board.values() if piece != 'EMPTY'])
    
    def get_colors_around_city(self, state: GameStateDivercite, city_pos: tuple) -> list:
        """
        Get the colors of all resources adjacent to a specified city.

        Args:
            state (GameStateDivercite): The current game state.
            city_pos (tuple[int, int]): The position of the city on the board.

        Returns:
            list[str]: A list of resource colors adjacent to the specified city.
        """
        adjacent_positions = state.get_neighbours(city_pos[0], city_pos[1])
        adjacent_colors = []
        for _, (piece, _) in adjacent_positions.items():
            if piece != 'EMPTY' and hasattr(piece, 'get_type'):
                piece_type = piece.get_type()
                if piece_type[1] == 'R':
                    adjacent_colors.append(piece_type[0])
        
        return adjacent_colors
           
    def get_missing_colors_for_divercite(self, current_colors: set) -> str:
        """
        Identify a missing color needed to complete a divercité.

        Args:
            current_colors (set[str]): A set of colors already present around a city.

        Returns:
            Optional[str]: A missing color needed for divercité, or None if all colors are present.
        """
        missing_colors = COLORS - current_colors
        return missing_colors.pop() if missing_colors else None

    def get_cities_affected_by_ressource(self, state: GameStateDivercite, ressource_pos: tuple) -> dict:
        """
        Identify all cities affected by a resource placed at a specific position.

        Args:
            state (GameStateDivercite): The current game state.
            ressource_pos (tuple[int, int]): The position of the resource on the board.

        Returns:
            dict[tuple[int, int], str]: A dictionary mapping city positions (tuples) to their respective piece types.
        """
        adjacent_positions = state.get_neighbours(ressource_pos[0], ressource_pos[1])
        adjacent_cities = {}
        # Collect cities affected by the resource
        for _, (piece, pos) in adjacent_positions.items():
            if piece != 'EMPTY' and hasattr(piece, 'get_type'):
                piece_type = piece.get_type()
                if piece_type[1] == 'C':
                    adjacent_cities.update({pos: piece_type})
        
        return adjacent_cities

    def evaluate_resource_scarcity(self, state: GameStateDivercite,player_id:str) -> float:
        """
        Evaluate the scarcity of the player's remaining resources and penalize excessive usage.

        Args:
            state (GameStateDivercite): The current game state.

        Returns:
            float: A score penalizing resource scarcity. A lower score indicates more scarce or depleted resources.
        """
        # Get the player's remaining pieces
        score = 0
        player_pieces_left = state.players_pieces_left[player_id]
        total_pieces = sum(player_pieces_left.values())
        # Constants
        DEPLETION_PENALTY = 150  # Penalty for completely depleting a resource color
        SCARCITY_MULTIPLIER = 100  # Penalty scaling for scarce resources

        # Penalize if a specific resource color is disproportionately used
        for piece, count in player_pieces_left.items():
            color = piece[0]  # Extract the color
            if count == 0:  # Resource depleted
                score -= SCARCITY_MULTIPLIER  # Heavy penalty for completely depleting a color
            else:
                scarcity_ratio = count / total_pieces
                score -= (1 - scarcity_ratio) * DEPLETION_PENALTY  # Penalize if resource is scarce

        return score

    def evaluate_ressource_placement(self, state: GameStateDivercite,player_symbol:str) -> float:
        """
        Evaluate the placement of resources based on their proximity to cities and strategic importance.

        Args:
            state (GameStateDivercite): The current game state.

        Returns:
            float: A score reflecting the quality of resource placement. Higher scores indicate better placements.
        """
         # Constants
        FRIENDLY_CITY_BONUS = 20
        OPPONENT_CITY_PENALTY = 25
        OPPONENT_ADVANTAGE_PENALTY = 75
        CORNER_OPPONENT_PENALTY = 250
        COLOR_MATCH_PENALTY = 30
        COMPLETE_DIVERCITE_PENALTY = 200
        NEARLY_COMPLETE_DIVERCITE_PENALTY = 100
        NO_ADJACENT_CITIES_PENALTY = 20

        score = 0
        board = state.get_rep().get_env()

        for pos, piece in board.items():
            piece_type = piece.get_type()
            if piece_type[1] == 'R':
                adjacent_cities = self.get_cities_affected_by_ressource(state, pos)
                if adjacent_cities:
                    opponent_cities = [city for city in adjacent_cities.values() if city[2] != player_symbol]
                    friendly_cities = [city for city in adjacent_cities.values() if city[2] == player_symbol]
                    friendly_city_count = len(friendly_cities)
                    opponent_city_count = len(opponent_cities)
                    
                    # Reward placements benefiting friendly cities
                    score += FRIENDLY_CITY_BONUS * friendly_city_count
                    # Penalize placements helping opponent cities
                    score -= OPPONENT_CITY_PENALTY * opponent_city_count
                    
                    if friendly_city_count == 0:
                        score -= OPPONENT_ADVANTAGE_PENALTY * (opponent_city_count ** 2)
                    
                    if pos in corner_ressource_position and opponent_city_count > 0:
                        score -= CORNER_OPPONENT_PENALTY
                    
                    opponent_cities_dict = {city_pos: city_type for city_pos, city_type in adjacent_cities.items() if city_type[2] != player_symbol}
                    for opponent_city_pos, opponent_city_type in opponent_cities_dict.items():
                        opponent_city_color = opponent_city_type[0]
                        piece_color = piece_type[0]
                        
                        if opponent_city_color == piece_color:
                            score -= COLOR_MATCH_PENALTY * len([city for city in adjacent_cities.values() if city[0] == piece_color])
                        
                        adjacent_colors = self.get_colors_around_city(state, opponent_city_pos)
                        unique_colors = set(adjacent_colors)
                        
                        if len(unique_colors) == 4 : 
                            score -= COMPLETE_DIVERCITE_PENALTY
                        elif len(unique_colors) == 3 and len(adjacent_colors) == 3:
                            score -= NEARLY_COMPLETE_DIVERCITE_PENALTY
                    
                elif not adjacent_cities:
                    score -= NO_ADJACENT_CITIES_PENALTY
                
        return score
    
    def evaluate_city_placement(self, state: GameStateDivercite) -> float:
        """
        Evaluate the placement of cities based on their proximity to resources and strategic value.

        Args:
            state (GameStateDivercite): The current game state.

        Returns:
            float: A score reflecting the quality of city placements. Higher scores indicate better placements.
        """
        FULL_DIVERCITE_BONUS = 200
        INEFFICIENT_RESOURCES_PENALTY = 50
        SINGLE_COLOR_BONUS = 20
        MULTI_COLOR_BONUS = 30
        ISOLATED_CITY_PENALTY = 100
        MIN_RESOURCES_THRESHOLD = 5

        score = 0
        board = state.get_rep().get_env()
        
        for pos, piece in board.items():
            piece_type = piece.get_type()
            if piece_type[1] == 'C':
                adjacent_colors = self.get_colors_around_city(state, pos)
                adjacent_resource_count = len(adjacent_colors)
                unique_colors = set(adjacent_colors)
                city_color = piece_type[0]
                
                # Case 1: Full divercité (4 unique colors)
                if len(unique_colors) == 4:
                    score += FULL_DIVERCITE_BONUS

                # Case 2: Inefficient resource placement (2 unique colors, not matching city color)
                elif len(unique_colors) == 2 and adjacent_resource_count > 2 and city_color not in unique_colors:
                    score -= INEFFICIENT_RESOURCES_PENALTY

                # Case 3: Single color matches city color
                elif len(unique_colors) == 1 and city_color in unique_colors:
                    score += SINGLE_COLOR_BONUS * adjacent_resource_count
                    
                 # Case 4: Multiple colors with city color included
                elif city_color in unique_colors: 
                    score += MULTI_COLOR_BONUS * len(unique_colors)
                
                 # Case 5: Isolated city with no adjacent resources (after early game)
                elif adjacent_resource_count == 0 and self.get_total_pieces_on_board(state) > 5:
                    score -= ISOLATED_CITY_PENALTY
                    
        return score
        
    def calculate_blocking_score(self, state: GameStateDivercite,player_symbol:str,opponent_id:str) -> float:
        """
        Calculate a score for blocking opponent cities from achieving DiverCité.

        Args:
            state (GameStateDivercite): The current game state.

        Returns:
            float: A score reflecting the blocking effectiveness. Higher scores indicate better blocking.
        """ 

        # Constants
        BLOCKING_BONUS_FULL = 200
        BLOCKING_BONUS_PARTIAL = 100
        BLOCKING_PENALTY = 50

        score = 0
        opponent_symbol = 'B' if player_symbol == 'W' else 'W'
        opponent_cities = self.get_placed_cities_by_player(state, opponent_symbol)
        
        for city_pos in opponent_cities.keys():
            adjacent_colors = self.get_colors_around_city(state, city_pos)
            unique_colors = set(adjacent_colors)
            city_color = opponent_cities[city_pos][0]
            blocking_color = next(
                (color for color in unique_colors if color != city_color),
                None
            )
            
            if len(unique_colors) == 3 and len(adjacent_colors) == 4 and adjacent_colors.count(blocking_color) == 2:
                # Count occurrences of each color
                color_counts = {color: adjacent_colors.count(color) for color in unique_colors}
                
                # Identify if there's a repeated color that makes DiverCité impossible
                repeated_colors = [color for color, count in color_counts.items() if count > 1]
                
                if len(repeated_colors) > 0:
                    # DiverCité impossible due to repeated colors
                    continue  # Skip this city as no threat exists
                
                ennemy_color_ressource_left = state.players_pieces_left[opponent_id]
                keys_to_remove = [key for key, value in ennemy_color_ressource_left.items() if (value == 0 and key[1] == "R") or key[1] == 'C']
                for key in keys_to_remove:
                    ennemy_color_ressource_left.pop(key)
                
                missing_color = self.get_missing_colors_for_divercite(unique_colors)
                
                if missing_color + "R" in ennemy_color_ressource_left:
                    print("Missing color: ", missing_color)
                    score += BLOCKING_BONUS_FULL
                
            elif len(unique_colors) == 2 and len(adjacent_colors) == 4 and adjacent_colors.count(blocking_color) == 1:
                # Check for blocking potential with only 2 resources
                # Ensure no color is repeated that prevents DiverCité
                color_counts = {color: adjacent_colors.count(color) for color in unique_colors}
                repeated_colors = [color for color, count in color_counts.items() if count > 1]
                
                if len(repeated_colors) == 0:
                    score += BLOCKING_BONUS_PARTIAL
                    
            elif len(unique_colors) == 3 and len(adjacent_colors) == 3 and adjacent_colors.count(blocking_color) == 1:
                # Check for blocking potential with only 3 resources
                # Ensure no color is repeated that prevents DiverCité
                color_counts = {color: adjacent_colors.count(color) for color in unique_colors}
                repeated_colors = [color for color, count in color_counts.items() if count > 1]
                
                if len(repeated_colors) == 0:
                    # DiverCité is still theoretically possible, valid blocking
                    score -= BLOCKING_PENALTY
                    
        return score
         
    def calculate_divercite_score(self, state: GameStateDivercite,player_symbol:str,player_id:str) -> float:
        """
        Calculate a score for the player's cities based on their progress towards DiverCité.

        Args:
            state (GameStateDivercite): The current game state.

        Returns:
            float: A score reflecting the progress of the player's cities towards DiverCité.
        """
        # Constants for scoring
        FULL_DIVERCITE_SCORE = 600
        SINGLE_COLOR_SCORE = 100
        RESOURCE_BONUS = 15
        CLOSE_TO_DIVERCITE_BONUS = 60
        OPPONENT_ADVANTAGE_PENALTY = 100
        INEFFICIENT_PLACEMENT_PENALTY = 50
        DIVERCIET_NOT_POSSIBLE_PENALTY = 50
        LACK_OF_RESOURCES_PENALTY = 10

        score = 0
        player_cities = self.get_placed_cities_by_player(state, player_symbol)

        for city_pos in player_cities.keys():
            adjacent_colors = self.get_colors_around_city(state, city_pos)
            unique_colors = set(adjacent_colors)
            city_color = player_cities[city_pos][0]
            
            # Case 1: Full divercité (4 unique colors + 4 resources => 5 points)
            if len(unique_colors) == 4 and len(adjacent_colors) == 4:
                score += FULL_DIVERCITE_SCORE  # Full divercité, highest score

            elif len(unique_colors) == 1 and city_color in unique_colors:
                if len(adjacent_colors) == 4:
                    score += SINGLE_COLOR_SCORE
                else:
                    score += RESOURCE_BONUS * len(adjacent_colors) 
                
            elif len(unique_colors) == 3 and len(adjacent_colors) == 3 and city_color in unique_colors:
                missing_color = self.get_missing_colors_for_divercite(unique_colors)
                empty_positions = [
                            pos for _, (piece, pos) in state.get_neighbours(city_pos[0], city_pos[1]).items() if piece == 'EMPTY'
                        ]
                if empty_positions:
                    for empty_pos in empty_positions:
                        adjacent_cities = self.get_cities_affected_by_ressource(state, empty_pos)
                        opponent_cities = [city for city in adjacent_cities.values() if city[2] != player_symbol]
                        opponent_cities_colors = [city[0] for city in opponent_cities]
                        
                        color_ressources_left = state.players_pieces_left[player_id]
                        # Collect keys to remove in a separate list+
                        keys_to_remove = [key for key, value in color_ressources_left.items() if (value == 0 and key[1] == "R") or key[1] == 'C']

                        # Remove the keys after the iteration
                        for key in keys_to_remove:
                            color_ressources_left.pop(key)
                        
                        if missing_color in color_ressources_left and opponent_cities:
                            score -= OPPONENT_ADVANTAGE_PENALTY * len(opponent_cities)
                            
                        elif missing_color + "R" in color_ressources_left :
                            score += CLOSE_TO_DIVERCITE_BONUS
                            
                        else:
                            score -= OPPONENT_ADVANTAGE_PENALTY
                
            # # Case 4: 2 unique colors + 2 resources => early progress
            elif len(unique_colors) == 2 and len(adjacent_colors) == 2 and city_color in unique_colors:
                missing_color = self.get_missing_colors_for_divercite(unique_colors)
                color_ressources_left = state.players_pieces_left[player_id]
                keys_to_remove = [key for key, value in color_ressources_left.items() if (value == 0 and key[1] == "R") or key[1] == 'C']
                for key in keys_to_remove:
                    color_ressources_left.pop(key)
                
                if missing_color in color_ressources_left:
                    score += INEFFICIENT_PLACEMENT_PENALTY
                
            # # Case 5: 1 unique color + 1 resource => initial placement
            # elif len(unique_colors) == 1 and len(adjacent_colors) == 1 and city_color in unique_colors:
            #     score += 10  # Minimal progress, lowest reward
                
            # Case 6: 3 unique colors + 4 resources => incomplete divercité
            elif len(unique_colors) == 3 and len(adjacent_colors) == 4 and adjacent_colors.count(city_color) < 3:
                score -= INEFFICIENT_PLACEMENT_PENALTY  # Penalize since divercité is not possible
    
            # Case 7: 2 unique colors + 4 resources (Locked city)
            elif len(unique_colors) == 2 and len(adjacent_colors) == 4 and city_color not in unique_colors:
                score -= INEFFICIENT_PLACEMENT_PENALTY  # Penalize lightly for inefficiency
            
            # Case 6: City with no matching resources
            elif len(unique_colors) == 2 and len(adjacent_colors) == 3 and adjacent_colors.count(city_color) < 2:
                score -= OPPONENT_ADVANTAGE_PENALTY
            
            elif city_color not in unique_colors and unique_colors:
                score -= DIVERCIET_NOT_POSSIBLE_PENALTY  # Penalize for placing a city with no adjacent resources of the same color
            # Default penalty for poor placement     
            else :
                score -= LACK_OF_RESOURCES_PENALTY
                
        return score
    
    def _compute_divercite_score(self, state: GameStateDivercite,player_symbol:str,player_id:str,opponent_id:str) -> float:
        """
        Evaluate the current game state using a heuristic that combines multiple scores.

        Args:
            state (GameStateDivercite): The current game state.

        Returns:
            float: A total heuristic score representing the favorability of the game state.
        """
        score_divercite = self.calculate_divercite_score(state,player_symbol,player_id)
        score_bloquage = self.calculate_blocking_score(state,player_symbol,opponent_id)
        player_actual_score = state.scores[player_id]
        score_resource_scarcity = self.evaluate_resource_scarcity(state,player_id)
        score_ressource_placement = self.evaluate_ressource_placement(state,player_symbol)
        city_placement_score = self.evaluate_city_placement(state)

        progress = ( self.get_total_pieces_on_board(state) / 42 ) * 100

        # Adjust weights dynamically    
        w_divercite = 1.0 if progress < 80 else 0.5
        # w_bloquage = 0.6 if progress < 50 else 1
        # w_divercite = 1
        w_bloquage = 1
        w_ressource_scarcity = 0.8 if progress < 40 else 0.3

        
        total_score = (
            w_divercite * score_divercite +
            w_bloquage * score_bloquage +
            score_ressource_placement +
            player_actual_score +
            city_placement_score +
            w_ressource_scarcity * score_resource_scarcity
        )

        return total_score
    
    def _evaluation(self, state: GameStateDivercite,**kwargs) -> float:
        opp_id = kwargs['opponent_id']
        my_id = kwargs['my_id']
        my_symbol = kwargs['my_piece_type']
        opponent_symbol = kwargs['opponent_piece_type']

        my_total_score = self._compute_divercite_score(state,my_symbol,my_id,opp_id)

        if self.loss_func == 'raw_eval':
            return my_total_score

        opp_total_score = self._compute_divercite_score(state,opponent_symbol,opp_id,my_id)

        if self.loss_func == 'potential':
            return self._maximized_potential(opp_total_score,my_total_score,self.optimization)

        if self.loss_func == 'diff':
            return my_total_score - opp_total_score
        
    
        return my_total_score
            
