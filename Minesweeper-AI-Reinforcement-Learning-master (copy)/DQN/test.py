import argparse
from tqdm import tqdm
from keras.models import load_model
from MinesweeperAgentWeb import *

def parse_args():
    parser = argparse.ArgumentParser(description='Play Minesweeper online using a DQN')
    #parser.add_argument('--model', type=str, default='conv128x4_dense512x2_y0.1_minlr0.001',

    parser.add_argument('--model', type=str, default='conv64x4_dense256x2_y0.1_minlr0.001',
                        help='name of model')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes to play')

    return parser.parse_args()

params = parse_args()

my_model = load_model(f'models/{params.model}.h5')

def main():
    pg.FAILSAFE = True
    agent = MinesweeperAgentWeb(my_model)

    for episode in tqdm(range(1, params.episodes+1)):
        agent.reset()

        done = False
        while not done:
            current_state = agent.state
            #print("current_state")
            #print(current_state)
            action = agent.get_action(current_state)
            print("action:")
            print(action)

            new_state, done = agent.step(action)
            #print("new_state")
            #print(new_state)

if __name__ == "__main__":
    main()
