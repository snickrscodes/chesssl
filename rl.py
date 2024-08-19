import chess
import tensorflow as tf
import keras
import numpy as np
import os
import math
from keras import ops

LOD = 2 # 0 = least features, 1 = medium, 2 = max
FEATURES = [384, 22528, 45056]
NUM_INPUTS = FEATURES[LOD]
piece_values = {chess.PAWN : 126, chess.KNIGHT : 781, chess.BISHOP : 825, chess.ROOK : 1276, chess.QUEEN : 2538}
KingBuckets = [
    -1, -1, -1, -1, 31, 30, 29, 28,
    -1, -1, -1, -1, 27, 26, 25, 24,
    -1, -1, -1, -1, 23, 22, 21, 20,
    -1, -1, -1, -1, 19, 18, 17, 16,
    -1, -1, -1, -1, 15, 14, 13, 12,
    -1, -1, -1, -1, 11, 10, 9, 8,
    -1, -1, -1, -1, 7, 6, 5, 4,
    -1, -1, -1, -1, 3, 2, 1, 0
]

class GLU(keras.layers.Layer):
    def __init__(self, units=32, activation=keras.activations.sigmoid, **kwargs):
        super().__init__()
        self.units = units
        self.activation = activation
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units*2), # we're going to split the doubled layer in half to retain dims
            initializer=keras.initializers.GlorotUniform(),
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units*2,), initializer=keras.initializers.Zeros(), trainable=True
        )
    def call(self, input):
        a1 = ops.matmul(input, self.w) + self.b
        # the * sign for hadamard product when combining the two halves
        return a1[:, :self.units] * self.activation(a1[:, self.units:])

@keras.saving.register_keras_serializable(package='activations', name='crelu')
def crelu(x):
    return ops.clip(x, 0.0, 1.0)
def sq_crelu(x):
    return ops.clip(x ** 2 * 0.9921875, 0.0, 1.0)
@keras.saving.register_keras_serializable(package='NNUEModel', name='nnue_model')
class KerasModel(keras.Model):
    def __init__(self, count=8, L1=256, L2=32):
        super().__init__()
        self.count = count # number of subnets
        self.L1 = L1
        self.L2 = L2
        # subnet 1
        self.w_sub1 = keras.layers.Dense(self.L1+self.count, name='w_sub1')
        self.b_sub1 = keras.layers.Dense(self.L1+self.count, name='b_sub1')
        # main subnet
        self.main_subnet1 = keras.layers.Dense(self.L2*self.count, activation=crelu, name='main_subnet1')
        self.main_subnet2 = keras.layers.Dense(self.L2*self.count, activation=crelu, name='main_subnet2')
        self.main_subnet3 = keras.layers.Dense(self.count, name='main_subnet3')

    def call(self, input: tuple, idx=[7]):
        x1 = self.w_sub1(input[0])
        x2 = self.b_sub1(input[1])
        h1 = (x1[:, self.L1:] - x2[:, self.L1:])/2.0
        x = ops.concatenate([x1[:, :self.L1], x2[:, :self.L1]], axis=-1)
        x = self.main_subnet1(x)
        x = self.main_subnet2(x)
        h2 = self.main_subnet3(x)
        x = h1+h2
        gather_idxs = [0] * len(idx)
        for i in range(len(idx)):
            gather_idxs[i] = [i, idx[i]]
        x = tf.gather_nd(x, gather_idxs)
        return x
    
    def build(self, input_shape):
        # Explicitly build the model to ensure all layers are initialized
        super().build(input_shape)

    def get_config(self):
        return {
            'count': self.count,
            'L1': self.L1,
            'L2': self.L2
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class Memory(object):
    def __init__(self, gamma=0.99, gae_lambda=0.95, max_games=10):
        self.gamma = gamma
        self.max_games = max_games
        self.gae_lambda = gae_lambda
        # store the xs and ys of previous games for replay memory
        self.input_memory_buffer = []
        self.input_piece_memory_buffer = []
        self.advantage_memory_buffer = []
        # store the results and states of the current game to calcualte advantage
        self.states, self.idxs, self.evals, self.rewards, self.overs = [], [], [], [], []
    def append(self, state, idx, eval, reward, over):
        self.states.append(state)
        self.idxs.append(idx)
        self.evals.append(eval)
        self.rewards.append(reward)
        self.overs.append(over)
    def remember(self, input, idxs, advantage):
        self.input_memory_buffer.append(input)
        self.input_piece_memory_buffer.append(idxs)
        self.advantage_memory_buffer.append(advantage)
        if len(self.input_memory_buffer) >= self.max_games:
            del self.input_memory_buffer[0]
            del self.input_piece_memory_buffer[0]
            del self.advantage_memory_buffer[0]
    def calc_adv(self):
        # q_values = []
        # for i in range(len(self.states)-1): # don't make a q value for the last position yet, we died there
        #     # version of the bellman equation
        #     new_q = self.rewards[i] + self.gamma*self.evals[i+1] # we already only store the max evaluation from the next state
        #     q_values.append(new_q)
        # q_values.append(self.rewards[-1])
        # return q_values
        # swapping this out for GAE because i realized chess is a game with sparse rewards :skull:
        advantage = [0] * len(self.rewards)
        for t in range(len(self.rewards)-1):
            discount = 1
            a_t = 0
            for k in range(t, len(self.rewards)-1):
                a_t += discount*(self.rewards[k] + self.gamma*self.evals[k+1]*(not self.overs[k]) - self.evals[k])
                discount *= self.gamma*self.gae_lambda
            advantage[t] = a_t
        return advantage

    def clear(self):
        self.states.clear()
        self.idxs.clear()
        self.evals.clear()
        self.rewards.clear()

class Agent(object):
    def __init__(self, dir, version=-1, max_games=10):
        self.model = self.load(dir, version)
        self.memory = Memory(max_games=max_games)
    
    def make_model(self, lr=0.001, beta_1=0.9, beta_2=0.999):
        model = KerasModel()
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2), loss=keras.losses.Huber(delta=1.0))
        return model

    def evaluate(self, board: chess.Board) -> float:
        return float(self.model(self.extract_features(board), idx=[(len(board.piece_map())-1)//4]))

    def choose_move(self, board: chess.Board):
        def evaluate_move(move):
            board.push(move)
            eval = 0
            if board.is_checkmate():
                eval = 100.0
            elif board.can_claim_draw() or board.is_stalemate() or board.is_insufficient_material():
                eval = 0.0
            else:
                eval = self.evaluate(board)
            board.pop()
            return eval
        moves = list(board.generate_legal_moves())
        x = np.array(list(map(evaluate_move, moves)))
        # apply softmax to logits
        if board.turn == chess.BLACK:
            x = -x
        range_x = np.max(x) - np.min(x)
        if range_x > 0:
            x_norm = (x - np.min(x)) / range_x
        else:
            x_norm = np.sign(x)
        exp_probs = np.exp(x_norm)
        probs = exp_probs/np.sum(exp_probs)
        # add dirichlet noise as in the alphazero paper (these are the same constants used in the paper)
        epsilon = 0.25
        alpha = 0.3
        dirichlet_noise = np.random.dirichlet([alpha] * len(probs)) # vector of values with the repeated alpha value
        noisy_probs = (1.0 - epsilon) * probs + epsilon * dirichlet_noise
        move = np.random.choice(moves, p=noisy_probs)
        return move

    def search(self, board: chess.Board, depth, alpha=-math.inf, beta=math.inf, return_best_move=True):
        if depth == 0:
            return self.evaluate(board) # we can assume batch size is 1 here
        if board.is_checkmate():
            return -100.0
        elif board.can_claim_draw() or board.is_stalemate() or board.is_insufficient_material():
            return 0.0
        def eval_move(move):
            board.push(move)
            eval = self.evaluate(board)
            board.pop()
            return eval
        # order the moves based on what the nn 'thinks' is promising (don't do this at the start when moves are random)
        moves = sorted(list(board.generate_legal_moves()), key = eval_move, reverse = True)
        # moves = list(board.generate_legal_moves())
        best_move = moves[0]
        for move in moves:
            board.push(move)
            eval = -self.search(board, depth-1, -beta, -alpha, False)
            board.pop()
            if eval >= beta:
                return beta
            if eval > alpha:
                best_move = move
                alpha = eval
        if return_best_move:
            return (best_move, alpha)
        else:
            return alpha

    # get the features from both perspectives
    def extract_features(self, board: chess.Board):
        match LOD:
            case 0:
                return self.ld_features(board)
            case 1:
                return self.md_features(board)
            case 2:
                return self.hd_features(board)
            case _:
                raise Exception('invalid lod')
    
    def hd_features(self, board: chess.Board):
        def orient(is_white_pov: bool, sq: int):
            return (56 * (not is_white_pov)) ^ sq
        def halfka_idx(is_white_pov: bool, king_sq: int, sq: int, p: chess.Piece):
            p_idx = (p.piece_type - 1) * 2 + (p.color != is_white_pov)
            if p_idx == 11:
                p_idx -= 1
            return orient(is_white_pov, sq) + p_idx * 64 + king_sq * 64 * 11
        w_indices = np.zeros(shape=(1, NUM_INPUTS))
        b_indices = np.zeros(shape=(1, NUM_INPUTS))
        for sq, p in board.piece_map().items():
            w_indices[0, halfka_idx(chess.WHITE, orient(chess.WHITE, board.king(chess.WHITE)), sq, p)] = 1.0
            b_indices[0, halfka_idx(chess.BLACK, orient(chess.BLACK, board.king(chess.BLACK)), sq, p)] = 1.0
        return (w_indices, b_indices)
    
    #TODO: bugged not working gotta fix
    def md_features(self, board: chess.Board):
        def orient(is_white_pov: bool, sq: int, ksq: int):
            # ksq must not be oriented
            kfile = (ksq % 8)
            return (7 * (kfile < 4)) ^ (56 * (not is_white_pov)) ^ sq
        def halfka_idx(is_white_pov: bool, king_sq: int, sq: int, p: chess.Piece):
            p_idx = (p.piece_type - 1) * 2 + (p.color != is_white_pov)
            o_ksq = orient(is_white_pov, king_sq, king_sq)
            if p_idx == 11:
                p_idx -= 1
            return orient(is_white_pov, sq, king_sq) + p_idx * 64 + KingBuckets[o_ksq] * 64 * 11
        values = [0] * NUM_INPUTS
        for ksq in range(64):
            for s in range(64):
                for pt, val in piece_values.items():
                    idxw = halfka_idx(True, ksq, s, chess.Piece(pt, chess.WHITE))
                    idxb = halfka_idx(True, ksq, s, chess.Piece(pt, chess.BLACK))
                    values[idxw] = val
                    values[idxb] = -val
        return np.array(values)

    def ld_features(self, board: chess.Board):
        w_pawn = (np.asarray(board.pieces(chess.PAWN, chess.WHITE).tolist())).astype(int)
        w_rook = (np.asarray(board.pieces(chess.ROOK, chess.WHITE).tolist())).astype(int)
        w_knight = (np.asarray(board.pieces(chess.KNIGHT, chess.WHITE).tolist())).astype(int)
        w_bishop = (np.asarray(board.pieces(chess.BISHOP, chess.WHITE).tolist())).astype(int)
        w_queen = (np.asarray(board.pieces(chess.QUEEN, chess.WHITE).tolist())).astype(int)
        w_king = (np.asarray(board.pieces(chess.KING, chess.WHITE).tolist())).astype(int)
        b_pawn = (np.asarray(board.pieces(chess.PAWN, chess.BLACK).tolist())).astype(int)
        b_rook = (np.asarray(board.pieces(chess.ROOK, chess.BLACK).tolist())).astype(int)
        b_knight = (np.asarray(board.pieces(chess.KNIGHT, chess.BLACK).tolist())).astype(int)
        b_bishop = (np.asarray(board.pieces(chess.BISHOP, chess.BLACK).tolist())).astype(int)
        b_queen = (np.asarray(board.pieces(chess.QUEEN, chess.BLACK).tolist())).astype(int)
        b_king = (np.asarray(board.pieces(chess.KING, chess.BLACK).tolist())).astype(int)
        return (np.expand_dims(np.concatenate((w_pawn, w_rook, w_knight, w_bishop, w_queen, w_king)), 0), np.expand_dims(np.concatenate((b_pawn, b_rook, b_knight, b_bishop, b_queen, b_king)), 0))

    def learn(self):
        with tf.GradientTape() as tape:
            y_true = tf.convert_to_tensor(self.memory.calc_adv())
            input = tf.convert_to_tensor(self.memory.states) # array of states, automatically adds a batch dim
            input = tf.squeeze(tf.transpose(input, perm=[1, 0, 3, 2]), axis=-1)
            y_pred = self.model(input, idx=self.memory.idxs)
            loss = keras.losses.huber(y_true, y_pred, delta=1.0)
            params = self.model.trainable_variables
            grads = tape.gradient(loss, params)
            self.model.optimizer.apply_gradients(zip(grads, params))
            # self.memory.remember(input, self.memory.idxs.copy(), y_true)
            self.memory.clear()
    def replay_memory(self):
        with tf.GradientTape(persistent=True) as tape:
            for i in range(len(self.memory.input_memory_buffer)):
                y_true = self.memory.advantage_memory_buffer[i]
                input = self.memory.input_memory_buffer[i]
                y_pred = self.model(input, idx=self.memory.input_piece_memory_buffer[i])
                loss = keras.losses.huber(y_true, y_pred, delta=1.0)
                params = self.model.trainable_variables
                grads = tape.gradient(loss, params)
                self.model.optimizer.apply_gradients(zip(grads, params))
    
    def save(self, dir, index=0):
        self.model.save(dir+'modelv'+str(index)+'.keras')
        # self.model.save_weights(CHKPT_DIR+'modelv'+str(index)+'.weights.h5')
    def load(self, dir, index=-1) -> keras.Model:
        files = os.listdir(dir)
        if index == -1: 
            index = len(files)-1
        if len(files) > 0:
            model = keras.saving.load_model(dir+'modelv'+str(index)+'.keras')
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999), loss=keras.losses.Huber(delta=1.0))
            # model.load_weights(CHKPT_DIR+'modelv'+str(index)+'.weights.h5')
            return model
        else:
            return self.make_model()

def is_terminal(board: chess.Board):
    return board.is_checkmate() or board.can_claim_draw() or board.is_stalemate() or board.is_insufficient_material()

def get_reward(board: chess.Board):
    if board.can_claim_draw() or board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    else:
        return 100.0 if board.turn == chess.BLACK else -100.0

def perft(depth: int, board: chess.Board) -> int:
    if depth == 1:
        return board.legal_moves.count()
    elif depth > 1:
        count = 0

        for move in board.legal_moves:
            board.push(move)
            count += perft(depth - 1, board)
            board.pop()

        return count
    else:
        return 1