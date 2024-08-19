from rl import Agent
import chess
import os
import random

TOTAL_GAMES = 1000000
CHKPT_STEPS = 1000
CHKPT_DIR = 'C:/Users/saraa/Desktop/chessai/models/'

def is_terminal(board: chess.Board):
    return board.is_checkmate() or board.can_claim_draw() or board.is_stalemate() or board.is_insufficient_material()

def get_reward(board: chess.Board):
    if board.can_claim_draw() or board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    else:
        return 100.0 if board.turn == chess.BLACK else -100.0

def str_board(board: chess.Board):
    out = ''
    board_str = str(board)
    rows = board_str.split('\n')
    rank_numbers = list(range(8, 0, -1)) # 1 to 8
    out += "+---+---+---+---+---+---+---+---+\n"
    for rank, row in zip(rank_numbers, rows):
        formatted_row = ' | '.join(row.split())
        out += f"| {formatted_row} | {rank}\n"
        out += "+---+---+---+---+---+---+---+---+\n"
    out += '  a   b   c   d   e   f   g   h'
    out = out.replace('.', ' ')
    return out

def train():
    agent = Agent(CHKPT_DIR, max_games=10)
    board = chess.Board()
    # stockfish = stockfish.Stockfish(path='C:/Users/saraa/Desktop/chessai/engine/sf16.exe')
    game_counter = 0
    # main network training loop
    print(agent.evaluate(board))
    print(agent.model.summary())
    for i in range(game_counter, TOTAL_GAMES):
        last_eval = 0.0
        # rate = 0.98 ** (i//1000+1) + 0.02
        while not is_terminal(board):
            agent.memory.append(agent.extract_features(board), (len(board.piece_map())-1)//4, last_eval, 0.0, False)
            # old epsilon greedy strategy
            # if random.random() <= rate:
            #     move = random.choice(list(board.legal_moves))
            #     last_eval = agent.evaluate(board)
            # else:
            #     (move, eval) = agent.search(board, 1)
            #     last_eval = eval
            # this will be considerably slower but will hopefully be more meaningful
            move = agent.choose_move(board)
            board.push(move)
            os.system('cls' if os.name == 'nt' else 'clear')
            print(str_board(board))
        game_counter += 1
        print(f'game {game_counter} completed')
        reward = get_reward(board)
        agent.memory.append(agent.extract_features(board), (len(board.piece_map())-1)//4, agent.evaluate(board), get_reward(board), True)
        board.reset()
        agent.learn()
        # agent.replay_memory()
        if (i+1) % 1000 == 0:
            agent.model.optimizer.learning_rate.assign(0.001 * 0.98 ** ((i+1) // 1000))
            agent.save(CHKPT_DIR, (i+1) // 1000 - 1)


def compare_versions(v1=0, v2=4, n_games=100):
    game = chess.Board()
    p1 = Agent(CHKPT_DIR, v1)
    p2 = Agent(CHKPT_DIR, v2)
    w = d = l = 0
    for i in range(n_games):
        p1_side = random.random() <= 0.5 # p1 is white if true
        while not is_terminal(game):
            if game.turn == p1_side:
                (move, eval) = p1.search(game, 1)
            else:
                (move, eval) = p2.search(game, 1)
            game.push(move)
            os.system('cls' if os.name == 'nt' else 'clear')
            print(game)
        if game.can_claim_draw() or game.is_stalemate() or game.is_insufficient_material():
            d += 1
        elif game.is_checkmate():
            if game.turn == p1_side:
                w += 1 # p2 wins!
            else:
                l += 1
        game.reset()

train()