from enum import Enum
from board_divercite import BoardDivercite
import numpy as np


########################################     Board & Game Related   #####################################

BOARD_MASK = BoardDivercite.BOARD_MASK

MAX_MOVES = 20
MAX_STEP = 40

C_TEMP_NUM =10
C_TEMP_DEN =20 +MAX_STEP




N_DIMS= 9
BOARD_DIMS = N_DIMS,N_DIMS
########################################     City Position    #####################################
CENTER_CITY = 'center_city_key'
CORNER_CITY = 'corner_city_key'
AROUND_CITY = 'around_city_key'

city_position = set([(1, 4), (2, 3), (2, 5), (3, 2), (3, 4),  (3, 6), (4, 1), (4, 3), (4, 5),
                     (4, 7), (5, 2),  (5, 4),  (5, 6), (6, 3), (6, 5),  (7, 4)])

corner_city_position = set([
    (1, 4), (4, 1), (4, 7), (7, 4)
])

no_corner_city_position = set(city_position).difference(corner_city_position)

center_city_position = set([(4, 3), (4, 5), (3, 4), (5, 4)])

no_corner_no_center_city_position = no_corner_city_position.difference(
    center_city_position)

city_index_control = {AROUND_CITY: 5,
                      CENTER_CITY: 8,
                      CORNER_CITY: 3}

########################################  Ressource Position    #####################################

ressources_position = set([(0, 4), (1, 3), (1, 5), (2, 2), (2, 4), (2, 6), (3, 1),
                           (3, 3),  (3, 5), (3, 7),  (4,
                                                      0),  (4, 2), (4, 4), (4, 6),
                           (4, 8), (5, 1), (5, 3), (5, 5),  (5, 7),  (6, 2),  (6, 4), (6, 6),  (7, 3),   (7, 5), (8, 4)])

outside_ressources_position = set([
    (0, 4), (1, 3), (1, 5), (2, 2), (2, 6), (3, 1),
    (3, 7), (4, 0), (4, 8), (5, 1), (5, 7), (6, 2), (6, 6),
    (7, 3), (7, 5), (8, 4)
])

corner_ressource_position = set([
    (4, 0), (0, 4), (8, 4), (4, 8)
])

center_ressources_position = ressources_position.difference(
    outside_ressources_position)


########################################   Index Position Compute    #####################################

horizontal_vertical_compute = [(1, 1), (-1, -1), (1, -1), (-1, 1)]
diagonal_compute = [(2, 0), (0, 2), (-2, 0), (0, -2)]
other_type_around_compute = [(1, 0), (-1, 0), (0, 1), (0, -1)]

########################################    Key     #####################################

CITY_KEY = 'C'
RESSOURCE_KEY = 'R'
COLORS = set(['R', 'G', 'B', 'Y'])
W_PIECE ='W'
B_PIECE='B'
PIECE_TYPE =set([W_PIECE,B_PIECE])

########################################   Pieces     #####################################


class RessourcesNames(Enum):
    RR = 'RR'
    GR = 'GR'
    YR = 'YR'
    BR = 'BR'


class CityNames(Enum):
    RC = 'RC'
    GC = 'GC'
    YC = 'YC'
    BC = 'BC'
