from player_divercite import PlayerDivercite
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_divercite import GameStateDivercite
from seahorse.utils.custom_exceptions import MethodNotImplementedError

from src_2147174_2117902.definition import Algorithm, Strategy
from src_2147174_2117902.strategy_controller import *
from src_2147174_2117902.tools import Monitor


class MyPlayer(PlayerDivercite):
    """
    Player class for Divercite game that makes random moves.

    Attributes:
        piece_type (str): piece type of the player
    """

    def __init__(self, piece_type: str, name: str = "LCD_ZePequeno"):
        """
        Initialize the PlayerDivercite instance.

        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player (default is "bob")
            time_limit (float, optional): the time limit in (s)
        """
        super().__init__(piece_type, name)
        scoreHeuristic = ScoreHeuristic(loss_func='diff')
        piecesVarianceHeuristic = PiecesVarianceHeuristic('sigmoid')
        controlIndexHeuristic = ControlIndexHeuristic('sigmoid')
        diverciteHeuristic  =DiverciteHeuristic(loss_func='raw_eval')
        
        hybrid = scoreHeuristic*8 + controlIndexHeuristic + piecesVarianceHeuristic
        hybrid2 = scoreHeuristic*8+ diverciteHeuristic*5 + piecesVarianceHeuristic*4 + controlIndexHeuristic*2

        self._controller: StrategyController = StrategyController().add_strategy(
           MinimaxTypeASearch(scoreHeuristic,2,150))    

    @Monitor
    def compute_action(self, current_state: GameStateDivercite, remaining_time: int = 1e9, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameState): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """
        # TODO
        Strategy.set_current_state(current_state, remaining_time)
        return self._controller()
    

    def __del__(self):
        self._controller.dump_stats()
