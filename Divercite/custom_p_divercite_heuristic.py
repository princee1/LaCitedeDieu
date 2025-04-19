from src_2147174_2117902.tools import Monitor
from src_2147174_2117902.constant import *
from player_divercite import PlayerDivercite
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_divercite import GameStateDivercite
from seahorse.utils.custom_exceptions import MethodNotImplementedError
from board_divercite import BoardDivercite

class MyPlayer(PlayerDivercite):
    """
    Player class for Divercite game that makes random moves.

    Attributes:
        piece_type (str): piece type of the player
    """

    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        """
        Initialize the PlayerDivercite instance.

        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player (default is "bob")
            time_limit (float, optional): the time limit in (s)
        """
        super().__init__(piece_type, name)
    
    def get_placed_cities_by_player(self, state: GameStateDivercite, player: str) -> dict:
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
            if piece_type[1] == 'C' and piece_type[2] == player:
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
        for direction, (piece, pos) in adjacent_positions.items():
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
        adjacent_positions = state.get_neighbours(ressource_pos[0], ressource_pos[1])
        adjacent_cities = {}
        
        for direction, (piece, pos) in adjacent_positions.items():
            if piece != 'EMPTY' and hasattr(piece, 'get_type'):
                piece_type = piece.get_type()
                if piece_type[1] == 'C':
                    adjacent_cities.update({pos: piece_type})
        
        return adjacent_cities

    def evaluate_resource_scarcity(self, state: GameStateDivercite) -> float:
        """
        Penalize excessive use of scarce resources.
        """
        score = 0
        player_pieces_left = state.players_pieces_left[self.get_id()]
        total_pieces = sum(player_pieces_left.values())

        # Penalize if a specific resource color is disproportionately used
        for piece, count in player_pieces_left.items():
            color = piece[0]  # Extract the color
            if count == 0:  # Resource depleted
                score -= 150  # Heavy penalty for completely depleting a color
            else:
                scarcity_ratio = count / total_pieces
                score -= (1 - scarcity_ratio) * 100  # Penalize if resource is scarce

        return score

    def evaluate_ressource_placement(self, state: GameStateDivercite) -> float:
        # Heuristic to boost the placement of ressources next to cities
        score = 0
        board = state.get_rep().get_env()
        corner_ressource_position = set([
            (4, 0), (0, 4), (8, 4), (4, 8)
        ])
        
        for pos, piece in board.items():
            piece_type = piece.get_type()
            if piece_type[1] == 'R':
                adjacent_cities = self.get_cities_affected_by_ressource(state, pos)
                if adjacent_cities:
                    opponent_cities = [city for city in adjacent_cities.values() if city[2] != self.piece_type]
                    friendly_cities = [city for city in adjacent_cities.values() if city[2] == self.piece_type]
                    friendly_city_count = len(friendly_cities)
                    opponent_city_count = len(opponent_cities)
                    
                    # Reward placements benefiting friendly cities
                    score += 20 * friendly_city_count
                    # Penalize placements helping opponent cities
                    score -= 25 * opponent_city_count
                    
                    if friendly_city_count == 0:
                        score -= 75 * (opponent_city_count ** 2)
                    
                    if pos in corner_ressource_position and opponent_city_count > 0:
                        score -= 250
                    
                    opponent_cities_dict = {city_pos: city_type for city_pos, city_type in adjacent_cities.items() if city_type[2] != self.piece_type}
                    for opponent_city_pos, opponent_city_type in opponent_cities_dict.items():
                        opponent_city_color = opponent_city_type[0]
                        piece_color = piece_type[0]
                        
                        if opponent_city_color == piece_color:
                            score -= 30 * len([city for city in adjacent_cities.values() if city[0] == piece_color])
                        
                        adjacent_colors = self.get_colors_around_city(state, opponent_city_pos)
                        unique_colors = set(adjacent_colors)
                        
                        if len(unique_colors) == 4 : 
                            score -= 200
                        elif len(unique_colors) == 3 and len(adjacent_colors) == 3:
                            score -= 100
                    
                elif not adjacent_cities:
                    score -= 20
                
        return score
    
    def evaluate_city_placement(self, state: GameStateDivercite) -> float:
        # Heuristic to boost the placement of cities near resources
        score = 0
        board = state.get_rep().get_env()
        
        for pos, piece in board.items():
            piece_type = piece.get_type()
            if piece_type[1] == 'C':
                adjacent_colors = self.get_colors_around_city(state, pos)
                adjacent_resource_count = len(adjacent_colors)
                unique_colors = set(adjacent_colors)
                city_color = piece_type[0]
                repeated_colors = [color for color in adjacent_colors if adjacent_colors.count(color) > 1]
                
                if len(unique_colors) == 4:
                    score += 200
                # elif len(unique_colors) == 3 and adjacent_resource_count == 3:
                #     missing_color = self.get_missing_colors_for_divercite(unique_colors)
                    
                #     color_ressources_left = state.players_pieces_left[self.get_id()]
                #     keys_to_remove = [key for key, value in color_ressources_left.items() if (value == 0 and key[1] == "R") or key[1] == 'C']
                #     for key in keys_to_remove:
                #         color_ressources_left.pop(key)
                        
                #     if missing_color in color_ressources_left:
                #         score += 100
                elif len(unique_colors) == 2 and adjacent_resource_count > 2 and city_color not in unique_colors:
                    score -= 50

                elif len(unique_colors) == 1 and city_color in unique_colors:
                    score += 20 * adjacent_resource_count
                    
                elif city_color in unique_colors: 
                    score += 30 * len(unique_colors)
                
                elif adjacent_resource_count == 0 and self.get_total_pieces_on_board(state) > 5:
                    score -= 100
                    
        return score
        
    def calculate_blocking_score(self, state: GameStateDivercite) -> float:
        # Heuristic to boost the blocking of cities with 3 different colors around them
        score = 0
        opponent_symbol = 'B' if self.piece_type == 'W' else 'W'
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
                
                ennemy_color_ressource_left = state.players_pieces_left[self.next_player.get_id()]
                keys_to_remove = [key for key, value in ennemy_color_ressource_left.items() if (value == 0 and key[1] == "R") or key[1] == 'C']
                for key in keys_to_remove:
                    ennemy_color_ressource_left.pop(key)
                
                missing_color = self.get_missing_colors_for_divercite(unique_colors)
                
                if missing_color + "R" in ennemy_color_ressource_left:
                    print("Missing color: ", missing_color)
                    score += 200
                
            elif len(unique_colors) == 2 and len(adjacent_colors) == 4 and adjacent_colors.count(blocking_color) == 1:
                # Check for blocking potential with only 2 resources
                # Ensure no color is repeated that prevents DiverCité
                color_counts = {color: adjacent_colors.count(color) for color in unique_colors}
                repeated_colors = [color for color, count in color_counts.items() if count > 1]
                
                if len(repeated_colors) == 0:
                    score += 100 
                    
            elif len(unique_colors) == 3 and len(adjacent_colors) == 3 and adjacent_colors.count(blocking_color) == 1:
                # Check for blocking potential with only 3 resources
                # Ensure no color is repeated that prevents DiverCité
                color_counts = {color: adjacent_colors.count(color) for color in unique_colors}
                repeated_colors = [color for color, count in color_counts.items() if count > 1]
                
                if len(repeated_colors) == 0:
                    # DiverCité is still theoretically possible, valid blocking
                    score -= 50
                    
        return score
         
    def calculate_divercite_score(self, state: GameStateDivercite) -> float:
        score = 0
        player_symbol = self.piece_type
        player_cities = self.get_placed_cities_by_player(state, player_symbol)

        for city_pos in player_cities.keys():
            adjacent_colors = self.get_colors_around_city(state, city_pos)
            unique_colors = set(adjacent_colors)
            city_color = player_cities[city_pos][0]
            
            # Case 1: Full divercité (4 unique colors + 4 resources => 5 points)
            if len(unique_colors) == 4 and len(adjacent_colors) == 4:
                score += 600  # Full divercité, highest score

            elif len(unique_colors) == 1 and city_color in unique_colors:
                if len(adjacent_colors) == 4:
                    score += 100
                else:
                    score += 15 * len(adjacent_colors) 
                
            elif len(unique_colors) == 3 and len(adjacent_colors) == 3 and city_color in unique_colors:
                missing_color = self.get_missing_colors_for_divercite(unique_colors)
                empty_positions = [
                            pos for _, (piece, pos) in state.get_neighbours(city_pos[0], city_pos[1]).items() if piece == 'EMPTY'
                        ]
                if empty_positions:
                    for empty_pos in empty_positions:
                        adjacent_cities = self.get_cities_affected_by_ressource(state, empty_pos)
                        opponent_cities = [city for city in adjacent_cities.values() if city[2] != self.piece_type]
                        opponent_cities_colors = [city[0] for city in opponent_cities]
                        
                        color_ressources_left = state.players_pieces_left[self.get_id()]
                        # Collect keys to remove in a separate list+
                        keys_to_remove = [key for key, value in color_ressources_left.items() if (value == 0 and key[1] == "R") or key[1] == 'C']

                        # Remove the keys after the iteration
                        for key in keys_to_remove:
                            color_ressources_left.pop(key)
                        
                        if missing_color in color_ressources_left and opponent_cities:
                            score -= 100 * len(opponent_cities)
                            
                        elif missing_color + "R" in color_ressources_left :
                            score += 60
                            
                        else:
                            score -= 100
                
            # # Case 4: 2 unique colors + 2 resources => early progress
            elif len(unique_colors) == 2 and len(adjacent_colors) == 2 and city_color in unique_colors:
                missing_color = self.get_missing_colors_for_divercite(unique_colors)
                color_ressources_left = state.players_pieces_left[self.get_id()]
                keys_to_remove = [key for key, value in color_ressources_left.items() if (value == 0 and key[1] == "R") or key[1] == 'C']
                for key in keys_to_remove:
                    color_ressources_left.pop(key)
                
                if missing_color in color_ressources_left:
                    score += 50
                
            # # Case 5: 1 unique color + 1 resource => initial placement
            # elif len(unique_colors) == 1 and len(adjacent_colors) == 1 and city_color in unique_colors:
            #     score += 10  # Minimal progress, lowest reward
                
            # Case 6: 3 unique colors + 4 resources => incomplete divercité
            elif len(unique_colors) == 3 and len(adjacent_colors) == 4 and adjacent_colors.count(city_color) < 3:
                score -= 50  # Penalize since divercité is not possible
    
            # Case 7: 2 unique colors + 4 resources (Locked city)
            elif len(unique_colors) == 2 and len(adjacent_colors) == 4 and city_color not in unique_colors:
                score -= 50  # Penalize lightly for inefficiency
            
            elif len(unique_colors) == 2 and len(adjacent_colors) == 3 and adjacent_colors.count(city_color) < 2:
                score -= 100
            
            elif city_color not in unique_colors and unique_colors:
                score -= 50  # Penalize for placing a city with no adjacent resources of the same color
                
            else :
                score -= 10
                
        return score
    
    def heuristic_evaluation(self, state: GameStateDivercite) -> float:
        score_divercite = self.calculate_divercite_score(state)
        score_bloquage = self.calculate_blocking_score(state)
        player_actual_score = state.scores[self.get_id()]
        score_resource_scarcity = self.evaluate_resource_scarcity(state)
        score_ressource_placement = self.evaluate_ressource_placement(state)
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
        
    def max_value(self, state : GameState, depth : int, max_depth : int, alpha : float, beta : float) -> float:
        if depth == max_depth:
            score = self.heuristic_evaluation(state)
            return score, None
        
        v_prime = float('-inf')
        m_prime = None
        
        for action in state.generate_possible_light_actions():
            next_state = state.apply_action(action)
            v, _ = self.min_value(next_state, depth + 1, max_depth, alpha, beta)
            
            if v > v_prime:
                v_prime = v
                m_prime = action
                alpha = max(alpha, v_prime)
            
            if v_prime >= beta:
                return v_prime, m_prime
        
        return v_prime, m_prime
    
    def min_value(self, state: GameState, depth: int, max_depth: int, alpha: float, beta: float) -> float:
        if depth == max_depth:
            score = self.heuristic_evaluation(state)
            return score, None
        
        v_prime = float('inf')
        m_prime = None
        
        for action in state.generate_possible_light_actions():
            next_state = state.apply_action(action)
            v, _ = self.max_value(next_state, depth + 1, max_depth, alpha, beta)
            
            if v < v_prime:
                v_prime = v
                m_prime = action
                beta = min(beta, v_prime)
            
            if v_prime <= alpha:
                return v_prime, m_prime
        
        return v_prime, m_prime
    def compute_action(self, current_state: GameState, remaining_time: int = 1e9, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameState): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """

        #TODO
        max_depth = 2
        
        _, best_action = self.max_value(current_state, 0, max_depth, float('-inf'), float('inf'))
        return best_action