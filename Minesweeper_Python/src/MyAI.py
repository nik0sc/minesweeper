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

    def tick(self):
        """
        Advance the AI.
        :return:
        """
        while True:
            if len(self.action_queue) != 0:
                print('<<< Things to do: leaving AI tick immediately')
                return

            # Goal test
            field_count = self.mf.get_report()
            if field_count[self.mf.UNFLAGGED] == 0:
                print('>>> Goal test passed! Global # Unflagged = 0')
                self.action_queue.append(Action(AI.Action.LEAVE))
                return

            if len(self.frontier) == 0:
                # Ran out of things to do!
                # The smart thing to do here is to apply some other problem
                # solving strategy or even randomly uncover a tile.
                print('Frontier is empty!!')
                return

            node = Window(self.mf, *self.frontier.dequeue())
            print('Current node: {}'.format(node))

            if node.center_field in self.explored:
                print('>>> Skipping previously explored node')
                continue

            node_rem_unflagged, node_flagged, node_covered = node.remaining_at()
            print('Rem. unflagged: {} Flagged: {} Covered: {}'.format(
                node_rem_unflagged, node_flagged, node_covered
            ))

            if node_rem_unflagged == 0:
                print('>>> Rule: {} no more tiles to flag (uncovering rest)'.format(node))
                to_uncover = self.mf.filter_tiles(
                    node.adjacents,
                    lambda x: x == self.mf.UNFLAGGED
                )
                for tile in to_uncover:
                    assert tile not in self.explored
                    self.action_queue.append(Action(AI.Action.UNCOVER, *tile))
            elif node_rem_unflagged == node_covered:
                print('>>> Rule: {} tiles to flag = # covered (flagging rest)'.format(node))
                to_flag = self.mf.filter_tiles(
                    node.adjacents,
                    lambda x: x == self.mf.UNFLAGGED
                )
                for tile in to_flag:
                    self.action_queue.append(Action(AI.Action.FLAG, *tile))
                    self.mf.flag_at(*tile)
            else:
                print('>>> Nothing to do at {}'.format(node))
                # Dead end for now, hopefully agent can find a path back here?
                continue

            for adjacent in node.adjacents:
                if adjacent not in self.explored:
                    print('Frontier: Enqueueing {}'.format(adjacent))
                    self.frontier.enqueue(adjacent)

            if node_covered == 0:
                print('Adding {} to explored'.format(node))
                self.explored.add(node.center_field)

    def getAction(self, number: int) -> "Action Object":
        # WARNING! ValueErrors and IndexErrors propagated from here are
        # swallowed up by the World class and an unhelpful error message results
        print('>>> Entering getAction')
        try:
            if number != -1:
                # Last action was an uncover action. Save the percept
                assert self.current_action.getMove() == AI.Action.UNCOVER
                update_info = (
                    self.current_action.getX(),
                    self.current_action.getY(),
                    number
                )
                print('>>> Update board at ({}, {}) with uncover percept {}'.format(
                    *update_info
                ))
                self.mf.update_board(*update_info)
            # Otherwise the last action was a flag or unflag. Ignore the percept
            # We received the percept resulting from the current action. Discard
            # the current action since we are done with it
            self.current_action = None

            if len(self.action_queue) == 0:
                # Run the AI.
                print('!!! Running AI tick')
                self.tick()
                print('??? AI thinks the world looks like this')
                self.mf.inspect()
                if len(self.action_queue) == 0:
                    # Stalled?
                    print('<<< Agent couldn\'t produce action, leaving...')
                    return Action(AI.Action.LEAVE)

            # Agent's next action
            self.current_action = self.action_queue.popleft()
            print('<<< Taking popped action {}'.format(self.current_action))
            return self.current_action
        except (ValueError, IndexError):
            print('!!! ValueError or IndexError in getAction')
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
            print('Warning: Updating a previously-set tile')

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

    def window_iter(self):
        """

        :return:
        """
        return (self.window_at(x, y)
                for x in range(self.board.shape[0])
                for y in range(self.board.shape[1]))

    def inspect(self):
        print('>>> Current board state\n')
        for row in self.board:
            for tile in row:
                if tile == self.UNFLAGGED:
                    print(' . ', end='')
                elif tile == self.FLAGGED:
                    print(' ? ', end='')
                else:
                    print(' {} '.format(tile), end='')
            print()
        counter = self.get_report()
        print('\n>>> Flagged: {} Remaining: {}'.format(
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
        print('Window:')
        print(self._window)
        tile_value = self.score
        print('Score: {} at {}'.format(tile_value, self.center_window))

        if tile_value < 0:
            raise ValueError('Still covered')

        flagged_count = len(self._window[self._window == Minefield.FLAGGED])
        covered_count = len(self._window[self._window == Minefield.UNFLAGGED])

        if flagged_count > tile_value:
            raise ValueError('Too many flags around this tile')

        return tile_value - flagged_count, flagged_count, covered_count
