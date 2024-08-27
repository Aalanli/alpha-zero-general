# %%
from dataclasses import dataclass
import logging
import os

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

@dataclass
class ConnectNNetConfig:
    batch_size: int = 64
    epochs: int = 10
    num_channels: int = 512
    p_drop: float = 0.3
    lr: float = 0.1
    log_loss_iter = 5


class ConnectNNet(eqx.Module):
    layers: list
    fc3: eqx.Module
    fc4: eqx.Module

    def __init__(self, game: ConnectNGame, config: ConnectNNetConfig, key):
        keys = jax.random.split(key, 8)
        flat_shape = config.num_channels * (game.config.board_n) ** 2
        self.layers = nn.Sequential([
            nn.Conv2d(1, config.num_channels, 3, 1, 1, key=keys[0]),
            nn.BatchNorm(config.num_channels, axis_name="batch"),
            nn.Lambda(jax.nn.relu),
            # nn.Conv2d(config.num_channels, config.num_channels, 3, 1, 1, key=keys[1]),
            # nn.BatchNorm(config.num_channels, axis_name="batch"),
            # nn.Lambda(jax.nn.relu),
            # nn.Conv2d(config.num_channels, config.num_channels, 3, 1, 1, key=keys[2]),
            # nn.BatchNorm(config.num_channels, axis_name="batch"),
            # nn.Lambda(jax.nn.relu),
            # nn.Conv2d(config.num_channels, config.num_channels, 3, 1, 1, key=keys[3]),
            # nn.BatchNorm(config.num_channels, axis_name="batch"),
            # nn.Lambda(jax.nn.relu),
            nn.Lambda(lambda x: x.reshape(-1)),
            # nn.Linear(flat_shape, 1024, key=keys[4]),
            # nn.BatchNorm(1024, axis_name="batch"),
            # nn.Lambda(jax.nn.relu),
            # nn.Dropout(p=config.p_drop),
            # nn.Linear(1024, 512, key=keys[4]),
            # nn.BatchNorm(512, axis_name="batch"),
            # nn.Lambda(jax.nn.relu),
            # nn.Dropout(p=config.p_drop),
        ])

        # self.fc3 = nn.Linear(512, game.getActionSize(), key=keys[6])
        # self.fc4 = nn.Linear(512, 1, key=keys[7])

        self.fc3 = nn.Linear(flat_shape, game.getActionSize(), key=keys[6])
        self.fc4 = nn.Linear(flat_shape, 1, key=keys[7])
    

    def __call__(self, board: Array, state, key):
        board = board.astype(self.fc4.weight.dtype)
        board, state = self.layers(board, state, key=key)
        pi = self.fc3(board)
        v = self.fc4(board)
        return pi, jax.nn.tanh(v), state
    
    def run_batch(self, boards, state, key):
        return jax.vmap(self, axis_name="batch", in_axes=(0, None, None), out_axes=(0, 1, None))(boards[:, None], state, key)

@eqx.filter_jit
def net_pred(net, board: Array, state, key):
    board = board[None, None]
    pis_pred, vs_pred, _ = jax.vmap(net, axis_name="batch", in_axes=(0, None, None), out_axes=(0, 1, None))(board, state, key)
    probs = jax.nn.softmax(pis_pred)
    return probs[0], vs_pred[0]

class ConnectNNetWrapper(NeuralNet):
    def __init__(self, game: ConnectNGame, config: ConnectNNetConfig, key):
        self.key, net_key = jax.random.split(key)
        self.net, self.state = eqx.nn.make_with_state(ConnectNNet)(game, config, net_key)
        self.game = game
        self.config = config

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
        net = self.net #eqx.nn.inference_mode(self.net)        
        return net_pred(net, board, self.state, self.key)

    def save_checkpoint(self, folder, filename):
        eqx.tree_serialise_leaves(os.path.join(folder, filename), [self.net, self.state, self.key])
    
    def load_checkpoint(self, folder, filename):
        self.net, self.state, self.key = eqx.tree_deserialise_leaves(os.path.join(folder, filename), [self.net, self.state, self.key])

from Coach import Coach
from Args import Args
import coloredlogs
coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = Args()

game = ConnectNGame()
# net, state = eqx.nn.make_with_state(ConnectNNet)(game)
# key = jax.random.PRNGKey(13)
# boards = game.getInitBoard()[None]
# pi, v, state = net.run_batch(boards, state, key)
net_config = ConnectNNetConfig()
key = jax.random.PRNGKey(42)
wrap = ConnectNNetWrapper(game, net_config, key)
coach = Coach(game, wrap, args)
coach.learn()

