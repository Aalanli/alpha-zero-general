# %%
from dataclasses import dataclass
import jax
from jax import Array
import jax.numpy as jnp
from Game import Game

@dataclass
class ConnectNConfig:
    connect_n: int = 5
    board_n: int = 7

@dataclass
class ConnectNNetConfig:
    batch_size: int = 64
    epochs: int = 10
    num_channels: int = 512
    p_drop: float = 0.3
    lr: float = 0.1
    key: Array = jax.random.PRNGKey(42)
    log_loss_iter = 5


class ConnectNGame(Game):
    def __init__(self, config: ConnectNConfig = ConnectNConfig(), net_config: ConnectNNetConfig = ConnectNNetConfig()):
        self.config = config
        self.net_config = net_config
        self.n = config.board_n
        self.cn = config.connect_n
        self.win_diag1 = jnp.eye(self.cn, self.cn, dtype=jnp.int8)
        self.win_diag2 = self.win_diag1[::-1]
        self.win_col = jnp.ones([self.cn], dtype=jnp.int8).reshape(1, self.cn)
        self.win_row = self.win_col.reshape(self.cn, 1)
    
    def getInitBoard(self):
        return jnp.zeros([self.n, self.n], jnp.int8)

    def getBoardSize(self):
        return (self.n, self.n)
    
    def getActionSize(self):
        return self.n ** 2

    def getNextState(self, board: Array, player: int, action: int):
        board = board.at[action // self.n, action % self.n].set(player)
        return (board, -player)
    
    def getValidMoves(self, board: Array, player: int):
        return jnp.concat([(board == 0).flatten()]).astype(jnp.float32)
    
    def getGameEnded(self, board: Array, player: int):
        fmin = jnp.array(0)
        fmax = jnp.array(0)
        for f in [self.win_diag1, self.win_diag2, self.win_col, self.win_row]:
            b1 = jax.scipy.signal.convolve2d(board, f, mode='valid').astype(jnp.int8)
            fmin = jnp.minimum(fmin, b1.min())
            fmax = jnp.maximum(fmax, b1.max())

        cmp = player * self.cn
        win = fmin == cmp | fmax == cmp
        loss = fmin == -cmp | fmax == -cmp
        return jnp.where(win, 1, jnp.where(loss, -1, 0))
    
    def getCanonicalForm(self, board, player):
        return board * player
    
    def getSymmetries(self, board: Array, pi: Array):
        pi_board = pi.reshape(self.n, self.n)
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = jnp.rot90(board, i)
                newPi = jnp.rot90(pi_board, i)
                if j:
                    newB = jnp.fliplr(newB)
                    newPi = jnp.fliplr(newPi)
                l += [(newB, newPi.flatten())]
        return l

    def stringRepresentation(self, board: Array):
        return board.tobytes()
    
    

if __name__ == '__main__':
    game = ConnectNGame(ConnectNConfig(board_n=5))
    board = game.getInitBoard()
    board, _ = game.getNextState(board, 1, 1)
    board = jnp.array([
        [-1, -1, -1, -1, 1],
        [0, -1, 0, 0, 0],
        [0, 0, -1, 0, 0],
        [0, 0, 0, -1, 0],
        [0, 0, 0, 0, 0],
    ])
    game.getGameEnded(board, 1)

