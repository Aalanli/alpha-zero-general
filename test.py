# %%
from dataclasses import dataclass
import logging

from tqdm import tqdm
import jax
from jax import Array
import jax.numpy as jnp

import equinox as eqx
from equinox import nn

import optax

from NeuralNet import NeuralNet
from connect_n import ConnectNGame

from utils import AverageMeter


class ConnectNNet(eqx.Module):
    layers: list
    fc3: eqx.Module
    fc4: eqx.Module

    def __init__(self, game: ConnectNGame):
        config = game.net_config
        keys = jax.random.split(config.key, 8)
        flat_shape = config.num_channels * (game.config.board_n - 4) ** 2
        self.layers = nn.Sequential([
            nn.Conv2d(1, config.num_channels, 3, 1, 1, key=keys[0]),
            nn.BatchNorm(config.num_channels, axis_name="batch"),
            nn.Lambda(jax.nn.relu),
            nn.Conv2d(config.num_channels, config.num_channels, 3, 1, 1, key=keys[1]),
            nn.BatchNorm(config.num_channels, axis_name="batch"),
            nn.Lambda(jax.nn.relu),
            nn.Conv2d(config.num_channels, config.num_channels, 3, 1, key=keys[2]),
            nn.BatchNorm(config.num_channels, axis_name="batch"),
            nn.Lambda(jax.nn.relu),
            nn.Conv2d(config.num_channels, config.num_channels, 3, 1, key=keys[3]),
            nn.BatchNorm(config.num_channels, axis_name="batch"),
            nn.Lambda(jax.nn.relu),
            nn.Lambda(lambda x: x.reshape(-1)),
            nn.Linear(flat_shape, 1024, key=keys[4]),
            nn.BatchNorm(1024, axis_name="batch"),
            nn.Lambda(jax.nn.relu),
            nn.Dropout(p=config.p_drop),
            nn.Linear(1024, 512, key=keys[4]),
            nn.BatchNorm(512, axis_name="batch"),
            nn.Lambda(jax.nn.relu),
            nn.Dropout(p=config.p_drop),
        ])

        self.fc3 = nn.Linear(512, game.getActionSize(), key=keys[6])
        self.fc4 = nn.Linear(512, 1, key=keys[7])
    

    def __call__(self, board: Array, state, key):
        board = board.astype(self.fc4.weight.dtype)
        board, state = self.layers(board, state, key=key)
        pi = self.fc3(board)
        v = self.fc4(board)
        return pi, jax.nn.tanh(v), state
    
    def run_batch(self, boards, state, key):
        return jax.vmap(self, axis_name="batch", in_axes=(0, None, None), out_axes=(0, 1, None))(boards[:, None], state, key)


class ConnectNNetWrapper(NeuralNet):
    fake_checkpoint: dict = {}

    def __init__(self, game: ConnectNGame):
        self.net, self.state = eqx.nn.make_with_state(ConnectNNet)(game)
        self.game = game
        self.config = game.net_config
        self.key = jax.random.split(self.config.key)[1]

    def train(self, examples: list):
        log = logging.getLogger('train')

        boards = jnp.stack([e[0] for e in examples])[:, None]
        pis = jnp.stack([e[1] for e in examples])
        vs = jnp.stack([e[2] for e in examples])
        optim = optax.adamw(self.config.lr)
        opt_state = optim.init(eqx.filter(self.net, eqx.is_array))

        def loss(model, state, boards, pis, vs, key):
            pis_pred, vs_pred, state = jax.vmap(model, axis_name="batch", in_axes=(0, None, None), out_axes=(0, 1, None))(boards, state, key)
            ls_pi = -jnp.sum(jax.nn.log_softmax(pis_pred) * pis) / pis.shape[0]
            ls_v = jnp.sum((vs - vs_pred) ** 2) / vs.shape[0]
            return ls_pi + ls_v, (ls_pi, ls_v, state)
        
        @eqx.filter_jit
        def step(model, state, opt_state, boards, pis, vs, key):
            key, sample, new_key = jax.random.split(key, 3)
            samples = jax.random.randint(sample, [self.config.batch_size], 0, pis.shape[0])
            
            (_, (ls_p, ls_v, state)), grad = eqx.filter_value_and_grad(loss, has_aux=True)(model, state, boards[samples], pis[samples], vs[samples], key)
            updates, opt_state = optim.update(grad, opt_state, model)
            model = eqx.apply_updates(model, updates)
            return model, state, opt_state, ls_p, ls_v, new_key

        pi_losses = AverageMeter()
        v_losses = AverageMeter()
        for epoch in tqdm(range(self.config.epochs), desc="epochs"):
            self.net, self.state, opt_state, ls_p, ls_v, self.key = step(self.net, self.state, opt_state, boards, pis, vs, self.key)
            
            pi_losses.update(ls_p.item())
            v_losses.update(ls_v.item())
            
            if (epoch + 1) % self.config.log_loss_iter == 0:
                log.info(f"pi loss: {pi_losses.avg}")
                log.info(f"v loss: {v_losses.avg}")

                pi_losses = AverageMeter()
                v_losses = AverageMeter()
        
    def predict(self, board: Array):
        @eqx.filter_jit
        def pred(net, board: Array, state, key):
            board = board[None, None]
            pis_pred, vs_pred, _ = jax.vmap(net, axis_name="batch", in_axes=(0, None, None), out_axes=(0, 1, None))(board, state, key)
            probs = jax.nn.softmax(pis_pred)
            return probs[0], vs_pred[0]

        net = self.net #eqx.nn.inference_mode(self.net)        
        return pred(net, board, self.state, self.key)

    def save_checkpoint(self, folder, filename):
        pa = folder + filename
        self.fake_checkpoint[pa] = (self.net, self.state, self.key)
    
    def load_checkpoint(self, folder, filename):
        pa = folder + filename
        self.net, self.state, self.key = self.fake_checkpoint[pa]

from utils import dotdict
from Coach import Coach

import coloredlogs
coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 1000,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})


game = ConnectNGame()
# net, state = eqx.nn.make_with_state(ConnectNNet)(game)
# key = jax.random.PRNGKey(13)
# boards = game.getInitBoard()[None]
# pi, v, state = net.run_batch(boards, state, key)

wrap = ConnectNNetWrapper(game)
coach = Coach(game, wrap, args)
coach.learn()

