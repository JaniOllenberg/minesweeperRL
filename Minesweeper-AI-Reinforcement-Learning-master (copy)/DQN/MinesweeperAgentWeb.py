import os
import numpy as np
import pyautogui as pg
import GPU_disable

ROOT = os.getcwd()
IMGS = f'{ROOT}/pics_windows'

EPSILON = 0.8

CONFIDENCES = {
    'unsolved': 0.99,
    'zero': 0.99,
    'one': 0.95,
    'two': 0.95,
    'three': 0.85,
    'four': 0.95,
    'five': 0.95,
    'six': 0.95,
    'seven': 0.95,
    'eight': 0.95
}

TILES = {
    'U': 'unsolved',
    '0': 'zero',
    '1': 'one',
    '2': 'two',
    '3': 'three',
    '4': 'four'
}

TILES2 = {
    '5': 'five',
    '6': 'six',
    '7': 'seven',
    '8': 'eight',
}

class MinesweeperAgentWeb(object):
    def __init__(self, model):
        #pg.click((10,100)) # click on current tab so 'F2' resets the game
        game_window = pg.locateCenterOnScreen(f'{IMGS}/game_window.png', confidence=0.85)
        print(game_window)
        if(game_window == None):
        	game_window = pg.locateOnScreen(f'{IMGS}/game_window_beginner.png', confidence=0.80)
        print(game_window)
        pg.click(game_window)
        self.reset()

        self.mode, self.loc, self.dims = self.get_loc()
        self.nrows, self.ncols = self.dims[0], self.dims[1]
        self.ntiles = self.dims[2]
        self.board = self.get_board(self.loc)
        self.state = self.get_state(self.board)

        self.epsilon = EPSILON
        self.model = model

    def reset(self):
        pg.press('f2')

    def get_loc(self):
        '''
        obtain mode, screen coordinates and dimensions for Minesweeper board
        '''

        modes = {'custom':(3,3,9), 'beginner':(8,8,64), 'intermediate':(16,16,256), 'expert':(16,30,480)}
        boards = {mode: pg.locateOnScreen(f'{IMGS}/{mode}.png', confidence=0.92) for mode in modes.keys()}

        assert boards != {'beginner':None, 'intermediate':None, 'expert':None, 'custom':None},\
            'Minesweeper board not detected on screen'

        for mode in boards.keys():
            if boards[mode] != None:
                diff = mode
                loc = boards[mode]
                dims = modes[mode]

        print(f"diff:{diff}\nloc:{loc}\ndims:{dims}")
        return diff, loc, dims

    def get_tiles(self, tile, bbox):
        '''
        Gets all locations of a given tile.
        Different confidence values are needed to correctly find different tiles with grayscale=True
        '''
        conf = CONFIDENCES[tile]
        tiles = list(pg.locateAllOnScreen(f'{IMGS}/{tile}.png', region=bbox, grayscale=True, confidence=conf))

        return tiles

    def get_board(self, bbox):
        '''
        Gets the state of the board as a dictionary of coordinates and values,
        ordered from left to right, top to bottom
        '''

        all_tiles = [[t, self.get_tiles(TILES[t], self.loc)] for t in TILES]

        # for speedup; look for higher tiles only if n of lower tiles < total ----
        count=0
        for value, coords in all_tiles:
            count += len(coords)

        if count < self.ntiles:
            higher_tiles = [[t, self.get_tiles(TILES2[t], self.loc)] for t in TILES2]
            all_tiles += higher_tiles
        # ----

        tiles = []
        for value, coords in all_tiles:
            for coord in coords:
                tiles.append({'coord': (coord[0], coord[1]), 'value': value})

        tiles = sorted(tiles, key=lambda x: (x['coord'][1], x['coord'][0]))
        # print(tiles)

        i=0
        for row in range(self.nrows):
            for column in range(self.ncols):
                # print()
                # print(f"i:{i} (y,x):{(y,x)}")
                tiles[i]['index'] = (row, column)
                # print(f"{tiles[i]['index']=}")
                # print(f"{len(tiles)=}")
                # print(f"{tiles[i]['value']=}")
                i+=1

        return tiles

    def get_state(self, board):
        '''
        Gets the numeric image representation state of the board.
        This is what will be the input for the DQN.
        '''

        state_im = [t['value'] for t in board]
        state_im = np.reshape(state_im, (self.nrows, self.ncols, 1)).astype(object)

        state_im[state_im=='U'] = -1
        state_im[state_im=='B'] = -2

        state_im = state_im.astype(np.int8) / 8
        state_im = state_im.astype(np.float16)

        return state_im

    def get_action(self, state):
        board = self.state.reshape(1, self.ntiles)
        unsolved = [i for i, x in enumerate(board[0]) if x==-0.125]

        rand = np.random.random() # random value b/w 0 & 1

        #if rand < self.epsilon: # random move (explore)
        #    move = np.random.choice(unsolved)
        #    print("random move!")
        #else:
        moves = self.model.predict(np.reshape(self.state, (1, self.nrows, self.ncols, 1)))
        moves[board!=-0.125] = np.min(moves)
        move = np.argmax(moves)

        print("predicted moves:")
        print(moves)
        return move

    def get_neighbors(self, action_index):
        board_2d = [t['value'] for t in self.board]
        board_2d = np.reshape(board_2d, (self.nrows, self.ncols))

        tile = self.board[action_index]['index']
        x,y = tile[0], tile[1]

        neighbors = []
        for col in range(y-1, y+2):
            for row in range(x-1, x+2):
                if (-1 < x < self.nrows and
                    -1 < y < self.ncols and
                    (x != row or y != col) and
                    (0 <= col < self.ncols) and
                    (0 <= row < self.nrows)):
                    neighbors.append(board_2d[row, col])

        return neighbors

    def step(self, action_index):
        done = False
        game_won = False

        # number of solved tiles prior to move (initialized at 0)
        #self.n_solved = self.n_solved_

        # get neighbors before clicking
        # neighbors = self.get_neighbors(action_index)

        x, y = pg.position()
        pg.click(self.board[action_index]['coord'])
        pg.moveTo(x,y)

        if pg.locateOnScreen(f'{IMGS}/oof.png', region=self.loc) != None: # if lose
            print(f"Game lost!------------------------")
            done = True
            game_won = False 

        elif pg.locateOnScreen(f'{IMGS}/gg.png', region=self.loc) != None: # if win
            done = True
            game_won = True

        else: # if progress
            self.board = self.get_board(self.loc)
            self.state = self.get_state(self.board)

        return self.state, done, game_won
