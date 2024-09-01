import tensorflow as tf
import numpy as np
import time
import stockfish
import math
import chess
import datetime
import chess.pgn
import random
import sys
import keras
import os


os.environ['KERAS_BACKEND'] = 'tensorflow'
keras.utils.set_random_seed(42)
NUM_INPUTS = 41024 # 64*(64*10+1)
CHKPT_DIR = 'C:/Users/saraa/Desktop/chessai/checkpoints/'
TRAINING_DIR = 'C:/Users/saraa/Desktop/chessai/training_data/'
def crelu(x):
    return tf.clip_by_value(x, 0.0, 1.0)

# sparse layers make computation easier!!

class Linear(object):
    def __init__(self, units: int, name: str, activation=None, input_shape=None, sparse=False):
        self.units = units
        self.name = name
        self.sparse = sparse
        self.activation = activation if activation is not None else lambda x: x
        self.weight = None
        self.bias = None
        if input_shape is not None:
            self.create_dense_vars(input_shape)

    def __call__(self, input: tf.SparseTensor | tf.Tensor):
        if self.weight is None or self.bias is None:
            self.create_dense_vars(input.shape.as_list())
        # the matmul converts the sparse tensor to a dense tensor
        if self.sparse:
            # embedding won't work here because it discards completely inactive inputs (all zeros)
            # x = tf.nn.embedding_lookup_sparse(self.weight, input, None, combiner="sum")
            # input MUST be of fp32 dtype otherwise this fails
            x = tf.sparse.sparse_dense_matmul(tf.cast(input, dtype=self.weight.dtype), self.weight)
        else:
            x = tf.matmul(input, self.weight)
        return self.activation(x + self.bias)
    
    def create_dense_vars(self, input_shape: list | tuple, weight_initializer=keras.initializers.RandomNormal(0.0, 1.0), bias_initializer=keras.initializers.Zeros(), variance=2.0):
        fan_in = np.prod(input_shape[1:])
        std = math.sqrt(variance / fan_in)
        weight_initializer.stddev = std
        self.weight = tf.Variable(weight_initializer(shape=(input_shape[-1], self.units)), name=self.name+"_weight", trainable=True)
        self.bias = tf.Variable(bias_initializer(shape=(1, self.units)), name=self.name+"_bias", trainable=True)

    def get_variables(self) -> list:
        return [self.weight, self.bias]
    
    # load variables from the dictionary we generate from a tf.train.Checkpoint

    def load_variables(self, vars: dict):
        for name, var in vars.items():
            if self.name+"_weight" in name:
                self.weight = var
            elif self.name+"_bias" in name:
                self.bias = var
        
class Adam(object):
    def __init__(self, lr=8.75e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-7, gamma=0.992):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.gamma = gamma
        self.epsilon = epsilon
        self.trainable_variables, self.moment1, self.moment2, self.updates = {}, {}, {}, {}

    # we might have sparse gradients

    def sparse_to_dense(self, input: tf.IndexedSlices):
        values = input.values
        indices = input.indices
        dense_shape = input.dense_shape
        dense_tensor = tf.scatter_nd(tf.expand_dims(indices, axis=1), values, dense_shape)
        return dense_tensor

    # a quick simple implementation of the adam algorithm, don't need any advanced features from the keras optimizers
    def update_variable(self, gradient: tf.Tensor, variable: tf.Variable):
        # no need for this, no embedding = no sparse gradients
        # if isinstance(gradient, tf.IndexedSlices):
        #     gradient = self.sparse_to_dense(gradient)
        if variable.name not in self.trainable_variables:
            self.trainable_variables[variable.name] = variable
            self.moment1[variable.name] = tf.zeros_like(variable)
            self.moment2[variable.name] = tf.zeros_like(variable)
            self.updates[variable.name] = 0
        self.moment1[variable.name] = self.beta_1 * self.moment1[variable.name] + gradient * (1.0 - self.beta_1)
        self.moment2[variable.name] = self.beta_2 * self.moment2[variable.name] + tf.square(gradient) * (1.0 - self.beta_2)    
        corrected_moment1 = self.moment1[variable.name] / (1.0 - self.beta_1 ** (self.updates[variable.name] + 1))
        corrected_moment2 = self.moment2[variable.name] / (1.0 - self.beta_2 ** (self.updates[variable.name] + 1))
        variable.assign_sub(self.lr * corrected_moment1 / (tf.sqrt(corrected_moment2) + self.epsilon))
        self.updates[variable.name] += 1

    def apply_gradients(self, grads_and_vars: zip):
        for grad, var in grads_and_vars:
            if grad is not None:
                self.update_variable(grad, var)

class NNUEModel(object):
    def __init__(self, count=8, L1=256, L2=32):
        self.count = count # number of subnets
        self.L1 = L1
        self.L2 = L2
        # subnet 1
        self.w_sub1 = Linear(self.L1+self.count, name='w_sub1', sparse=True)
        self.b_sub1 = Linear(self.L1+self.count, name='b_sub1', sparse=True)
        # main subnet
        self.main_subnet1 = Linear(self.L2*self.count, activation=crelu, name='main_subnet1')
        self.main_subnet2 = Linear(self.L2*self.count, activation=crelu, name='main_subnet2')
        self.main_subnet3 = Linear(self.count, name='main_subnet3')
        self.layers = [self.w_sub1, self.b_sub1, self.main_subnet1, self.main_subnet2, self.main_subnet3]

    def __call__(self, input: tuple[tf.SparseTensor]):
        # since all nonzero entries are 1, this will count the number of pieces
        # gradients shouldn't pass through the index computation, no weights are used
        num_pieces = tf.sparse.reduce_sum(input[0], axis=-1, keepdims=False)
        idxs = (num_pieces+1)//4 # -1+2, the +2 comes from accounting for 2 kings bc the feature set doesn't
        gather_idxs = tf.transpose(tf.stack([tf.range(0, tf.size(idxs)), idxs], axis=0), perm=[1, 0])
        # now evaluate the rest of the network before deciding on the output to use
        x1 = self.w_sub1(input[0]) # evaluates the first side to move
        x2 = self.b_sub1(input[1]) # evaluates the second side to move
        h1 = (x1[:, self.L1:] - x2[:, self.L1:])/2.0
        x = tf.concat([x1[:, :self.L1], x2[:, :self.L1]], axis=-1)
        x = self.main_subnet1(x)
        x = self.main_subnet2(x)
        h2 = self.main_subnet3(x)
        x = h1+h2
        # takes the indices and pairs them with their corresponding entries in each state in the batch
        # then transposes it to generate indices to take from x
        x = tf.gather_nd(x, gather_idxs)
        return x
    
    def save_checkpoint(self, dir=CHKPT_DIR):
        checkpoint = tf.train.Checkpoint(**self.get_vars_dict())
        checkpoint.save(dir+'nnue_checkpoint')

    def get_vars_dict(self) -> dict:
        # gets all the variables in the model
        vars = {}
        for layer in self.layers:
            for variable in layer.get_variables():
                vars[variable.name] = variable
        return vars
    
    def get_trainable_variables(self) -> list:
        # gets all the variables in the model
        vars = []
        for layer in self.layers:
            for variable in layer.get_variables():
                vars.append(variable)
        return vars

    def load_checkpoint(self, dir=CHKPT_DIR):
        vars = {}
        checkpoint_reader = tf.train.load_checkpoint(dir)
        var_to_shape_map = checkpoint_reader.get_variable_to_shape_map()
        for var_name in var_to_shape_map:
            name = var_name[:str.find(var_name, '/')]
            variable = tf.Variable(initial_value=checkpoint_reader.get_tensor(var_name), trainable=True, name=name)
            vars[name] = variable
        for layer in self.layers:
            layer.load_variables(vars)

class Memory(object):
    def __init__(self, gamma=0.99, gae_lambda=0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        # store the results of the current game to calcualte advantage
        self.evals, self.rewards, self.overs = [], [], []
    def append(self, eval, reward, over):
        self.evals.append(eval)
        self.rewards.append(reward)
        self.overs.append(over)
    def calc_adv(self):
        advantage = [0] * len(self.rewards)
        for t in range(len(self.rewards)-1):
            discount = 1
            a_t = 0
            for k in range(t, len(self.rewards)-1):
                a_t += discount*(self.rewards[k] + self.gamma*self.evals[k+1]*(not self.overs[k]) - self.evals[k])
                discount *= self.gamma*self.gae_lambda
            advantage[t] = a_t
        return advantage
    def calc_q(self):
        q_values = [0]*len(self.rewards)
        for i in range(len(self.rewards)-1):
            # version of the bellman equation
            new_q = self.rewards[i] + self.gamma*self.evals[i+1] # we already only store the max evaluation from the next state
            q_values[i] = new_q
        q_values[-1] = self.rewards[-1] # the last state is always terminal
        return q_values

    def clear(self):
        self.overs.clear()
        self.evals.clear()
        self.rewards.clear()

class Bot(object):
    def __init__(self):
        self.model = NNUEModel()
        self.optimizer = Adam(lr=8.75e-4, beta_1=0.9, beta_2=0.999, epsilon=1.0e-7, gamma=0.992)
        self.scaling = 340
        self.loss_lambda = 1.0
        self.losses = np.zeros(shape=(10000,), dtype=np.float32)
        self.num_epochs = 800
        self.steps = 0
    
    def save_checkpoint(self, dir=CHKPT_DIR):
        self.model.save_checkpoint(dir)
    
    def load_checkpoint(self, dir=CHKPT_DIR):
        self.model.load_checkpoint(dir)

    def get_features(self, boards: list[chess.Board]) -> tuple[tf.SparseTensor]:
        def orient(is_white_pov: bool, sq: int):
            return (63 * (not is_white_pov)) ^ sq

        def halfkp_idx(is_white_pov: bool, king_sq: int, sq: int, p: chess.Piece):
            p_idx = (p.piece_type - 1) * 2 + (p.color != is_white_pov)
            return 1 + orient(is_white_pov, sq) + p_idx * 64 + king_sq * 641

        def has_only_kings(board: chess.Board):
            for sq, p in boards[i].piece_map().items():
                if p.piece_type != chess.KING:
                    return False
            return True

        cur_indices = []
        opp_indices = []
        for i in range(len(boards)):
            cur_side = boards[i].turn
            for sq, p in boards[i].piece_map().items():
                if p.piece_type == chess.KING:
                    continue
                cur_indices.append([i, halfkp_idx(cur_side, orient(cur_side, boards[i].king(cur_side)), sq, p)])
                opp_indices.append([i, halfkp_idx(not cur_side, orient(not cur_side, boards[i].king(not cur_side)), sq, p)])
        cur_features = tf.SparseTensor(cur_indices, [1]*len(cur_indices), [len(boards), NUM_INPUTS])
        opp_features = tf.SparseTensor(opp_indices, [1]*len(opp_indices), [len(boards), NUM_INPUTS])
        return (cur_features, opp_features)
    
    def evaluate(self, board):
        if isinstance(board, chess.Board):
            return float(self.model(self.get_features([board])))
        elif isinstance(board, list):
            return list(self.model(self.get_features(board)))
        else:
            raise Exception('the board needs to be either a single board or list of boards to evaluate')

    # this was intended for stockfish training data

    # def train(self, input, labels, results): # results and labels need to have same dimensionality
    #     with tf.GradientTape() as tape:
    #         logits = self.model(input)
    #         wdl_eval_model = tf.nn.sigmoid(logits / self.scaling)
    #         wdl_eval_target = tf.nn.sigmoid(labels / self.scaling)
    #         # wdl_value_target = self.loss_lambda * wdl_eval_target + (1.0 - self.loss_lambda) * results
    #         loss = tf.reduce_mean(tf.abs(wdl_eval_model - wdl_eval_target) ** 2.5)
    #     variables = self.model.get_trainable_variables()
    #     gradients = tape.gradient(loss, variables)
    #     self.optimizer.apply_gradients(zip(gradients, variables))

    def learn(self, file: str, num_games=1000000):
        with open(file) as pgn:
            boards, score = self.parse_file(pgn)
            count = 0
            while boards is not None and count <= num_games:
                try:
                    input = self.get_features(boards)
                    evals = self.model(input).numpy().tolist() # take advantage of batch processing
                    memory = Memory()
                    memory.evals = evals; memory.overs = [False]*len(evals)
                    memory.overs[-1] = True
                    memory.rewards = [0]*len(evals) # the only reward is at the end of the game! *very* sparse
                    memory.rewards[-1] = score
                    labels = tf.convert_to_tensor(memory.calc_q())
                    self.train(input, labels)
                    count += 1
                    boards, score = self.parse_file(pgn)
                except:
                    print(f'exception after {self.steps} steps')

    def train(self, input, labels):
        with tf.GradientTape() as tape:
            logits = self.model(input)
            loss = tf.reduce_mean(tf.abs(logits - labels) ** 2.5)
            self.losses[self.steps%10000] = loss.numpy()
        variables = self.model.get_trainable_variables()
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        self.steps += 1
        if self.steps % 10000 == 0:
            bot.save_checkpoint()
            print(f'{datetime.datetime.now()} completed {self.steps} training steps, average loss is {np.mean(self.losses)}')

    def parse_file(self, pgn):
        game = chess.pgn.read_game(pgn)
        if game is None:
            return None, 0.0
        boards = []
        board = game.board()
        for move in game.mainline_moves():
            board.push(move)
            boards.append(board.copy(stack=False))
        result = game.headers.get("Result")
        score = 0.0
        if result == "1-0":
            score = 100.0
        elif result == "0-1":
            score = -100.0
        return boards, score
        
    def train_loop(self):
        files = os.listdir(TRAINING_DIR)
        for epoch in range(self.num_epochs):
            print(f'{datetime.datetime.now()} begin epoch {epoch+1}')
            for file in files:
                self.learn(TRAINING_DIR+file)
            self.optimizer.lr *= self.optimizer.gamma

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
        
def is_terminal(board: chess.Board):
    return board.is_checkmate() or board.can_claim_draw() or board.is_stalemate() or board.is_insufficient_material()

def print_bar(wins, draws, losses, bar_length=30):
    total = wins + draws + losses
    win_ratio = wins / total
    draw_ratio = draws / total
    loss_ratio = losses / total
    win_length = int(bar_length * win_ratio)
    draw_length = int(bar_length * draw_ratio)
    loss_length = int(bar_length * loss_ratio)
    bar = f"\033[92m{'█' * win_length}\033[0m\033[90m{'█' * draw_length}\033[0m\033[91m{'█' * loss_length}\033[0m"
    if len(bar) < bar_length:
        bar += ' ' * (bar_length - len(bar))
    print(f"[{bar}] W:{wins}, D:{draws}, L:{losses}")

def play(bot1: Bot, bot2: Bot, depth=1, n_games=100):
    game = chess.Board()
    w = d = l = 0
    for i in range(n_games):
        p1_side = random.random() <= 0.5 # p1 is white if true
        while not is_terminal(game):
            if game.turn == p1_side:
                (move, eval) = bot1.search(game, depth)
            else:
                (move, eval) = bot2.search(game, depth)
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
    print_bar(w, d, l)
