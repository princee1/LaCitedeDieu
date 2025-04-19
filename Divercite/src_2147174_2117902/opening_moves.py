from .definition import Strategy,Heuristic
from seahorse.game.light_action import LightAction,Action
from random import choice
from .constant import *
from .helper import *
from seahorse.game.game_layout.board import Piece

POSITION_KEY = 'position'
PIECE_KEY = 'piece'

class RandomMoveHeuristic(Heuristic):
    def evaluate(self, current_state,**kwargs):
        return choice(list(current_state.get_possible_light_actions())) 

############################## Simple Strategy ##########################################
class SimpleMoveStrategy(Strategy):
    def __init__(self, heuristic):
        super().__init__(heuristic)

    def search(self):
        return self.main_heuristic(self.current_state,my_id=self.my_id, opponent_id=self.opponent_id,
                                        my_pieces=self.my_pieces, opponent_pieces=self.opponent_pieces,
                                        last_move=self.last_move, is_first_to_play=self.is_first_to_play, moves=self.moves,
                                        my_score=self.my_score, opponent_score=self.opponent_score,current_env=self.current_env)


############################## Opening Moves Strategy ##################################
class OpeningMoveStrategy(Strategy):
    # NOTE if theres multiple opening strategy, move the code to a heuristic
    def __init__(self,force_same_color:bool):
        super().__init__(None)
        self.force_same_color = force_same_color
        self.last_color_played = None
        self.center_city_position = center_city_position.copy()
        self.no_corner_city_position = no_corner_city_position.copy()


    def _search(self) -> LightAction:
       # NOTE forcing the same color or a different might not necessary be a better move
        if self.is_first_to_play and  self.current_state.step == 0 :
            city = choice(CityNames._member_names_)
            self.last_color_played,_ = city
            pos=choice(list(center_city_position))
            self.center_city_position.difference_update([pos])
            return LightAction({POSITION_KEY: pos, PIECE_KEY: city})
        
        # update computed moves
        self.center_city_position.difference_update(self.moves)
        self.no_corner_city_position.difference_update(self.moves)
        
        pieces:Piece = self.current_state.rep.env[self.last_move]
        c,t,_ = pieces.piece_type

        if  self.current_state.step >=2:
            if c == self.last_color_played:
                c= choice(list(COLORS.difference([c])))
            piece =  c+CITY_KEY 

        else:
            piece = c+CITY_KEY if self.force_same_color else choice(CityNames._member_names_)
            self.last_color_played,_= piece


        
        if t == CITY_KEY:
                
                if self.last_move in center_city_position:
                    index_compute = horizontal_vertical_compute if self.force_same_color else diagonal_compute # BUG Hyperparameter to test, based on statistic wether we play the same color or not 
                    return LightAction({POSITION_KEY: check_certain_position(self.last_move,index_compute,self.center_city_position), PIECE_KEY: piece })

                if self.last_move in corner_city_position:
                    # BUG Might not be ideal to play close to the player since it can rob our divercity or point
                    c_2 = self.last_color_played if self.last_color_played != None else c
                    c = choice(list(COLORS.difference([c,c_2])))
                    self.last_color_played = c

                    return LightAction({POSITION_KEY: check_certain_position(self.last_move,diagonal_compute,self.center_city_position), PIECE_KEY:c+CITY_KEY  })
                
                self.last_color_played = c
                return LightAction({POSITION_KEY: check_certain_position(self.last_move,horizontal_vertical_compute,self.center_city_position), PIECE_KEY: c+CITY_KEY  })
                
 
        if self.last_move in center_ressources_position:

            neighbors = self.current_state.get_neighbours(self.last_move[0],self.last_move[1])
            neighbors = [ v[1] for _,v in neighbors.items()]
            neighbors = list(self.center_city_position.intersection(neighbors))
            if len(neighbors)!=0:
                neighbors = choice(neighbors)
                self.last_color_played = c
                return LightAction({POSITION_KEY: neighbors, PIECE_KEY: c+CITY_KEY})
            pos = minimize_maximize_distance(self.last_move,self.no_corner_city_position)
            return LightAction({POSITION_KEY: pos, PIECE_KEY: piece})
            
        
        pos = minimize_maximize_distance(self.last_move,self.center_city_position)
        return LightAction({POSITION_KEY: pos, PIECE_KEY: piece})

