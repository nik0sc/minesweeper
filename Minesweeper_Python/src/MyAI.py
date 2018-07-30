# ==============================CS-199==================================
# FILE:			MyAI.py
#
# AUTHOR: 		Justin Chung
#
# DESCRIPTION:	This file contains the MyAI class. You will implement your
#				agent in this file. You will write the 'getAction' function,
#				the constructor, and any additional helper functions.
#
# NOTES: 		- MyAI inherits from the abstract AI class in AI.py.
#
#				- DO NOT MAKE CHANGES TO THIS FILE.
# ==============================CS-199==================================

from AI import AI
from Action import Action

import numpy as np
from collections import OrderedDict, deque, defaultdict
from traceback import print_exc
import itertools as it
import random

DEBUG_PRINT = True


def dprint(*args, **kwargs):
    if DEBUG_PRINT:
        print(*args, **kwargs)


class MyAI(AI):
    """
    My AI class.
    """
    def __init__(self, rowDimension, colDimension, totalMines, startX, startY):
        # Minefield state
        self.mf = Minefield(rowDimension, colDimension, startX, startY, totalMines)

        # Add starting point to the expansion
        # Frontier is some kind of FIFO here. We just need constant-time
        # enqueue, dequeue, and membership testing, which OrderedDict has.
        # See https://stackoverflow.com/questions/8176513/ for more info
        self.frontier = Frontier()
        self.frontier.enqueue((startX, startY))

        # Fully exhausted on this run (won't come back again)
        self.explored = set()

        # Action queue
        self.action_queue = deque()
        # The agent took this action and now it is waiting for the percept.
        # This action is already taken by the game when it starts
        self.current_action = Action(AI.Action.UNCOVER, startX, startY)

    def goal_test(self):
        """
        Goal test.
        :return: True if goal state is reached
        """
        field_count = self.mf.get_report()
        if field_count[self.mf.UNFLAGGED] == 0:
            dprint('>>> Goal test passed! Global # Unflagged = 0')
            return True
        else:
            return False

    def tick(self):
        """
        Advance the AI.
        :return:
        """
        while True:
            if len(self.action_queue) != 0:
                dprint('<<< Things to do: leaving AI tick immediately')
                return

            if self.goal_test():
                self.action_queue.append(Action(AI.Action.LEAVE))
                return

            if len(self.frontier) == 0:
                # Ran out of things to do!
                dprint('!!! Frontier is empty!!')
                # Try looking for special patterns
                # dprint('>>> Looking for patterns')
                # edge_actions = self.search_edge_patterns()
                # if len(edge_actions) > 0:
                #     self.action_queue.extend(edge_actions)
                #     dprint('<<< Edge patterns found: things to do')
                #     return

                dprint('>>> Forcing rescan')
                self.rescan()
                if len(self.frontier) == 0 or len(self.action_queue) == 0:
                    dprint('Random uncover time')
                    self.uncover_random()
                    assert len(self.action_queue) != 0
                    return


                # if len(self.frontier) == 0:
                #     # Stalled?
                #     dprint('>>> Forcing rescan')
                #     self.rescan()
                #
                # # Don't try to dequeue nodes from an empty frontier
                # return

            node = Window(self.mf, *self.frontier.dequeue())
            dprint('Current node: {}'.format(node))

            if node.center_field in self.explored:
                dprint('>>> Skipping previously explored node')
                continue

            self.apply_opening_rule(node)

    def apply_opening_rule(self, node):
        """
        Apply the opening ruleset to the given tile, hopefully producing actions
        to add to the action queue and more nodes to add to the frontier.
        :param node: The <Window> object to apply the rule on.
        """
        if node.score < 0:
            dprint('Not clear: node {} score {}'.format(node, node.score))

        node_rem_unflagged, node_flagged, node_covered = node.remaining_at()
        dprint('Rem. unflagged: {} Flagged: {} Covered: {}'.format(
            node_rem_unflagged, node_flagged, node_covered
        ))

        if node_rem_unflagged == 0:
            dprint('>>> Rule: {} no more tiles to flag (uncovering rest)'.format(node))
            to_uncover = self.mf.filter_tiles(
                node.adjacents,
                lambda x: x == self.mf.UNFLAGGED
            )
            for tile in to_uncover:
                assert tile not in self.explored
                self.action_queue.append(Action(AI.Action.UNCOVER, *tile))
        elif node_rem_unflagged == node_covered:
            dprint('>>> Rule: {} tiles to flag = # covered (flagging rest)'.format(node))
            to_flag = self.mf.filter_tiles(
                node.adjacents,
                lambda x: x == self.mf.UNFLAGGED
            )
            for tile in to_flag:
                self.action_queue.append(Action(AI.Action.FLAG, *tile))
                self.mf.flag_at(*tile)
        else:
            dprint('>>> Nothing to do at {}'.format(node))
            # Dead end for now, hopefully agent can find a path back here?
            return

        for adjacent in node.adjacents:
            if adjacent not in self.explored:
                dprint('Frontier: Enqueueing {}'.format(adjacent))
                self.frontier.enqueue(adjacent)

        if node_covered == 0:
            dprint('Adding {} to explored'.format(node))
            self.explored.add(node.center_field)

    def uncover_random(self):
        """
        Uncover a random tile
        :return:
        """
        # Select the unflagged tiles
        where_result = np.where(self.mf.board==self.mf.UNFLAGGED)
        unflagged_tiles = np.dstack(np.array(list(where_result))).reshape((-1, 2))
        dprint('Unflagged tiles: {}'.format(unflagged_tiles))

        # Remove tiles next to an unflagged one
        unflagged_safe = []
        for coord in unflagged_tiles:
            node = Window(self.mf, *coord)
            # Don't want tiles with uncovered tiles around them
            if np.all(node.window<=0):
                unflagged_safe.append(coord)

        # If none are left, fall back to the unfiltered list
        if len(unflagged_safe) == 0:
            unflagged_safe = unflagged_tiles

        # Pick one and live with it
        coord_to_uncover = tuple(random.choice(unflagged_safe))
        self.action_queue.append(
            Action(AI.Action.UNCOVER, *coord_to_uncover)
        )
        self.frontier.enqueue(coord_to_uncover)

    def get_boundary(self):
        """
        Return the "boundary" of the cleared areas of the board
        :return: List of (x,y)-tuples
        """
        pass

    def search_edge_patterns(self):
        """
        Search the board for 1-1 and 1-2 patterns that indicate a mine or clear
        space nearby
        """
        actions = []
        # for window in self.mf.window_iter(field_edges=False):
        #     if window[1,1] == 1:
        #         if window[0,1] == 1 and window[]:
        #             actions.append(Action(AI.Action.UNCOVER))
        for x in pattern_info:
            pattern = Pattern(**x)
            actions.extend(pattern.match_board(self.mf))

        return actions


    def rescan(self):
        """
        Rescan the board and find new actions to take.
        :return:
        """
        for node in self.mf.window_iter():
            if node.score >= 0:
                self.apply_opening_rule(node)

    def getAction(self, number: int) -> "Action Object":
        # WARNING! ValueErrors and IndexErrors propagated from here are
        # swallowed up by the World class and an unhelpful error message results
        dprint('>>> Entering getAction')
        try:
            if number != -1:
                # Last action was an uncover action. Save the percept
                assert self.current_action.getMove() == AI.Action.UNCOVER
                update_info = (
                    self.current_action.getX(),
                    self.current_action.getY(),
                    number
                )
                dprint('>>> Update board at ({}, {}) with uncover percept {}'.format(
                    *update_info
                ))
                self.mf.update_board(*update_info)
            # Otherwise the last action was a flag or unflag. Ignore the percept
            # We received the percept resulting from the current action. Discard
            # the current action since we are done with it
            self.current_action = None

            if len(self.action_queue) == 0:
                # Run the AI.
                dprint('!!! Running AI tick')
                self.tick()
                dprint('??? AI thinks the world looks like this')
                self.mf.inspect()

                if len(self.action_queue) == 0:
                    dprint('!!! Agent couldn\'t produce action, leaving')
                    return Action(AI.Action.LEAVE)

            # Agent's next action
            self.current_action = self.action_queue.popleft()
            dprint('<<< Taking popped action {}'.format(self.current_action))
            dprint('Current action queue ({}): {}'.format(
                len(self.action_queue), self.action_queue))
            dprint('Current frontier ({}): {}'.format(
                len(self.frontier), self.frontier
            ))
            dprint('Current explored ({}): {}'.format(
                len(self.explored), self.explored
            ))
            return self.current_action
        except (ValueError, IndexError):
            dprint('!!! ValueError or IndexError in getAction')
            print_exc()
            raise


class Frontier(OrderedDict):
    def enqueue(self, key, value=None):
        # Does nothing if the key is already in the queue
        self.__setitem__(key, value)

    def dequeue(self, get_value=False):
        if get_value:
            return self.popitem(last=False)
        else:
            return self.popitem(last=False)[0]


class Minefield:
    """
    Class to let the AI keep track of game world state.
    """

    # 0-8: mine proximity
    # -1: unknown, unflagged
    # -2: unknown, flagged
    TILE_VALUES = frozenset(range(-2, 9))
    UNFLAGGED = -1
    FLAGGED = -2

    def __init__(self, dim_x, dim_y, start_x, start_y, total_mines):
        # Create an empty starting minefield
        self.board = np.full((dim_x, dim_y), self.UNFLAGGED, dtype=np.int8)
        # have a notion of "current tile"
        self._cursor = (0, 0)
        self.cursor = (start_x, start_y)
        self.total_mines = total_mines

    def __getitem__(self, key):
        # direct pass through to underlying ndarray
        return self.board[key]

    def __setitem__(self, key, value):
        self.board[key] = value

    @property
    def dim_x(self):
        return self.board.shape[0]

    @property
    def dim_y(self):
        return self.board.shape[1]

    def check_in_bounds(self, x, y):
        return (0 <= x < self.dim_x) and (0 <= y < self.dim_y)

    def get_cursor(self):
        return self._cursor

    def set_cursor(self, new):
        for a, b in zip(new, self.board.shape):
            assert 0 <= a < b, 'Cursor not in array bounds'

        self._cursor = new

    cursor = property(get_cursor, set_cursor)

    def get_report(self):
        return defaultdict(int, zip(*np.unique(self.board, return_counts=True)))

    def filter_tiles(self, coordinates, key):
        """
        Filter a list of coordinates on the field by a supplied function
        :param coordinates: The coordinates to filter on
        :param key: The function to filter with
        :return: Filtered list
        """
        return [coord for coord in coordinates
                if key(self.board[coord])]

    def update_board(self, x, y, value):
        if self.board[x, y] != -1:
            dprint('Warning: Updating a previously-set tile')

        assert value in self.TILE_VALUES, \
            '{} is not a valid tile value'.format(value)

        self.board[x, y] = value

    def flag_at(self, x, y):
        if self.board[x, y] != -1:
            raise ValueError('Can only flag an unknown and unflagged tile')

        self.board[x, y] = -2

    def unflag_at(self, x, y):
        if self.board[x, y] != -2:
            raise ValueError('Can only unflag an unknown and flagged tile')

        self.board[x, y] = -1

    def window_at(self, x=None, y=None):
        """
        Get a window into the game board.

        :param x:
        :param y:
        :return:
        """
        if x is None or y is None:
            return Window(self, *self.cursor)
        else:
            return Window(self, x, y)

    def window_iter(self, field_edges=True):
        """

        :return:
        """
        if field_edges:
            return (self.window_at(x, y)
                    for x in range(self.board.shape[0])
                    for y in range(self.board.shape[1]))
        else:
            return (self.window_at(x, y)
                    for x in range(1, self.board.shape[0] - 1)
                    for y in range(1, self.board.shape[1] - 1))

    def inspect(self):
        dprint('>>> Current board state\n')
        for row in self.board:
            for tile in row:
                if tile == self.UNFLAGGED:
                    dprint(' . ', end='')
                elif tile == self.FLAGGED:
                    dprint(' ? ', end='')
                else:
                    dprint(' {} '.format(tile), end='')
            dprint()
        counter = self.get_report()
        dprint('\n>>> Flagged: {} Remaining: {}'.format(
            counter[self.FLAGGED],
            self.total_mines - counter[self.FLAGGED]))


class Window:
    """
    A 2x2 to 3x3 window into the minefield.

    Due to the way slicing works on numpy arrays, the window updates when the
    main minefield updates.
    """
    def __init__(self, field: Minefield, x: int, y: int):
        self._field = field
        assert field.check_in_bounds(x, y)
        self._window = self._field[max(0, x-1):min(x+2, self._field.dim_x),
                                   max(0, y-1):min(y+2, self._field.dim_y)]
        self._center_field = (x, y)

    def __str__(self):
        return '<Window/Node {}, {}>'.format(*self.center_field)

    @property
    def field(self):
        """
        Get the parent minefield
        """
        return self._field

    @property
    def center_field(self):
        """
        Get the tile on the field that is being targeted by the window
        """
        return self._center_field

    @property
    def center_window(self):
        """
        Get the center of the window (which tile of the window maps to the
        center_field tile)
        """
        # This is easiest to see if you write out the center of window for each
        # tile on a minefield.
        return tuple(min(coord, 1) for coord in self.center_field)

    @property
    def adjacents(self):
        """
        Get a list of coordinates of adjacent tiles.
        """
        # Generate all 8 possible coordinates and throw out the ones that are
        # out of bounds
        coords = [(x + self.center_field[0], y + self.center_field[1])
                  for x in range(-1, 2) for y in range(-1, 2)
                  if (x, y) != (0, 0)]

        return [pair for pair in coords
                if self.field.check_in_bounds(*pair)]

    @property
    def window(self) -> np.ndarray:
        """
        Get the window into the minefield
        """
        return self._window

    @property
    def score(self):
        """
        Get the tile score (the percept value)
        """
        return self.field[self.center_field]

    def __getitem__(self, key):
        return self._window[key]

    def __setitem__(self, key, value):
        self._window[key] = value

    def remaining_at(self) -> 'Tuple(Rem_Unflagged, Flagged, Covered)':
        """
        Remaining unflagged, flagged, and covered tiles around this tile.
        """
        dprint('Window:')
        dprint(self._window)
        tile_value = self.score
        dprint('Score: {} at {}'.format(tile_value, self.center_window))

        if tile_value == Minefield.UNFLAGGED:
            raise ValueError('Still covered', {
                'window': self.window,
                'center_field': self.center_field
            })
        elif tile_value == Minefield.FLAGGED:
            dprint('! remaining_at on already-flagged tile')

        flagged_count = len(self._window[self._window == Minefield.FLAGGED])
        covered_count = len(self._window[self._window == Minefield.UNFLAGGED])

        if flagged_count > tile_value >= 0:
            raise ValueError('Too many flags around this tile')

        return tile_value - flagged_count, flagged_count, covered_count


class Pattern():
    """
    Pattern matching class.
    """
    UNFLAGGED = -1
    FLAGGED = -2
    DONTMATCH = -3
    UNFLAGGED_UNCOVER = -4
    UNFLAGGED_FLAG = -5
    SPECIALS = (DONTMATCH, UNFLAGGED_FLAG, UNFLAGGED_UNCOVER)

    ROT_NONE = 0
    ROT_4 = 1
    ROT_4_FLIP = 2

    _sym_name = {
        ROT_NONE: 'None',
        ROT_4: '4-way rotation',
        ROT_4_FLIP: '4-way rotation w/ flip'
    }

    def __init__(self, pattern: np.ndarray, symmetry=ROT_4_FLIP,
                 where=None, name=''):
        self.patterns = []
        self.name = name
        self.symmetry = symmetry

        if symmetry == self.ROT_NONE:
            self.patterns.append(pattern)
        elif symmetry == self.ROT_4:
            self.patterns.extend(np.rot90(pattern, k) for k in range(4))
            assert len(self.patterns) == 4
        elif symmetry == self.ROT_4_FLIP:
            self.patterns.extend(np.rot90(pattern, k) for k in range(4))
            self.patterns.extend(np.rot90(np.fliplr(pattern), k) for k in range(4))
            assert len(self.patterns) == 8
        else:
            raise ValueError

        self.get_tiles = where or self.tiles_all

    def __str__(self):
        return '<Pattern "{}", symmetry {}>'.format(
            self.name, self._sym_name.get(self.symmetry))

    __repr__ = __str__

    @classmethod
    def tiles_all(cls, field: Minefield):
        """
        Return all tiles in the field
        :param field: The minefield
        """
        return [(x, y) for x in range(field.dim_x) for y in range(field.dim_y)]

    @classmethod
    def tiles_at_innerborder(cls, field: Minefield):
        """
        Return a list of all tiles 1 tile away (straight or diagonally) from
        the border of the minefield.
        :param field: The minefield
        """
        return (
            [
                (x, y)
                for x in range(1, field.dim_x - 1)
                for y in (1, field.dim_y - 2)
            ]
            + [
                (x, y)
                for x in (1, field.dim_x - 2)
                for y in range(2, field.dim_y - 2)
            ])

    @classmethod
    def tiles_at_scoredtiles(cls, field: Minefield):
        """
        Return a list of tiles that have a nonzero score (are known to be
        touching a mine)
        :param field: The minefield
        """
        return field.filter_tiles(
            cls.tiles_all(field),
            lambda x: x > 0
        )

    def match_one(self, node: Window):
        """
        Match this pattern against a window and return actions to take
        :param node:
        :return:
        """
        if self.patterns[0].shape != node.window.shape:
            dprint('!!! Pattern shape and window shape mismatch at {}'
                   .format(node))
            return []

        for pattern in self.patterns:
            # dstack is like zip but for ndarrays
            # 'zip' and flatten
            zipped = np.dstack((pattern, node.window)).reshape((-1, 2))

            bad_pattern = False
            for pair in zipped:

                if pair[0] == self.DONTMATCH:
                    # Skip this pair
                    continue
                elif ((pair[0] == self.UNFLAGGED_UNCOVER or pair[0] == self.UNFLAGGED_FLAG)
                        and pair[1] != node.field.UNFLAGGED):
                    # Won't work either
                    dprint('Pattern failed: unflagged, node: {}, pair: {}'.format(node, pair))
                    dprint('Pattern: {}'.format(pattern))
                    bad_pattern = True
                    break
                elif pair[0] >= 0 and pair[0] != pair[1]:
                    # This sub-pattern won't work, move on to the next one
                    # assert pair[0] not in self.SPECIALS
                    dprint('Pattern failed: notequal, node: {}, pair: {}'.format(node, pair))
                    dprint('Pattern: {}'.format(pattern))
                    bad_pattern = True
                    break

            if bad_pattern:
                dprint('Moving on to next pattern in {}'.format(self))
                continue

            # Made it out of here, this pattern is good
            dprint('Good pattern: {}'.format(self))

            # Search this particular pattern for the action tiles
            # Then offset from centre_field
            flag_at = [tuple(c + node.center_field - node.center_window
                             for c in coord)
                       for coord in np.where(pattern==self.UNFLAGGED_FLAG)]
            uncover_at = [tuple(c + node.center_field - node.center_window
                                for c in coord)
                          for coord in np.where(pattern==self.UNFLAGGED_UNCOVER)]

            flag_actions = [Action(AI.Action.FLAG, *d)
                            for d in flag_at]
            uncover_actions = [Action(AI.Action.UNCOVER, *d)
                               for d in uncover_at]

            # if len(flag_actions) > 0:
            dprint('flag actions: {}'.format(flag_actions))
            # if len(uncover_actions) > 0:
            dprint('uncover actions: {}'.format(uncover_actions))

            return flag_actions + uncover_actions

    def match_board(self, mf: Minefield):
        tiles = self.get_tiles(mf)
        result = list(it.chain(self.match_one(Window(mf, *coord))
                               for coord in tiles))
        return result

pattern_info = [
    {
        'pattern': [
            [0, 0, 0],
            [1, 2, 1],
            [Pattern.UNFLAGGED_FLAG, Pattern.UNFLAGGED, Pattern.UNFLAGGED_FLAG]
        ],
        'symmetry': Pattern.ROT_4,
        'where': Pattern.tiles_at_scoredtiles,
        'name': '1-2-1'
    },
    {
        'pattern': [
            [0, 0, Pattern.DONTMATCH],
            [1, 1, Pattern.DONTMATCH],
            [Pattern.UNFLAGGED, Pattern.UNFLAGGED, Pattern.UNFLAGGED_UNCOVER]
        ],
        'symmetry': Pattern.ROT_4_FLIP,
        'where': Pattern.tiles_at_innerborder,
        'name': '1-1'
    },
    {
        'pattern': [
            [0, 0, Pattern.DONTMATCH],
            [1, 2, Pattern.DONTMATCH],
            [Pattern.UNFLAGGED, Pattern.UNFLAGGED, Pattern.UNFLAGGED_FLAG]
        ],
        'symmetry': Pattern.ROT_4_FLIP,
        'where': Pattern.tiles_at_innerborder,
        'name': '1-2'
    }
]


class Action(Action):
    """
    Improved action class (because the existing one is a pain to debug)
    """
    _move_tostr = {
        AI.Action.UNCOVER: 'UNCOVER',
        AI.Action.FLAG: 'FLAG',
        AI.Action.UNFLAG: 'UNFLAG',
        AI.Action.LEAVE: 'LEAVE'
    }

    def __str__(self):
        if self.getMove() == AI.Action.LEAVE:
            return '<Action: {}>'.format(self._move_tostr[self.getMove()])
        else:
            return '<Action: {} at ({},{})>'.format(
                self._move_tostr[self.getMove()],
                self.getX(),
                self.getY()
            )

    __repr__ = __str__