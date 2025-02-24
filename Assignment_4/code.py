import numpy as np
from collections import defaultdict
import random
import matplotlib.pyplot as plt

class TicTacToeEnv:
    """
    Tic Tac Toe environment that follows a gym-like interface.
    The board is represented as a 3x3 numpy array with:
    0 for empty squares
    1 for player X (Q-learning agent)
    -1 for player O (opponent)
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset the game board to initial empty state"""
        self.board = np.zeros((3, 3))
        self.current_player = 1  # X starts
        return self._get_state()
    
    def _get_state(self):
        """
        Convert board to a tuple for Q-table key.
        We use a tuple because it's immutable and can be used as a dictionary key.
        """
        return tuple(self.board.flatten())
    
    def _check_winner(self):
        """
        Check if there's a winner or draw.
        Returns:
            1 for X win
            -1 for O win
            0 for draw
            None if game is still ongoing
        """
        # Check rows, columns and diagonals
        for player in [1, -1]:
            # Rows and columns
            for i in range(3):
                if all(self.board[i, :] == player) or all(self.board[:, i] == player):
                    return player
            # Diagonals
            if all(np.diag(self.board) == player) or all(np.diag(np.fliplr(self.board)) == player):
                return player
        # Check for draw - occurs when all squares are filled
        if np.all(self.board != 0):
            return 0
        return None
    
    def get_valid_moves(self):
        """Return list of empty squares as (row, col) tuples"""
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]
    
    def step(self, action):
        """
        Make a move and return (next_state, reward, done).
        
        Args:
            action: tuple (row, col) indicating where to place the mark
            
        Returns:
            next_state: tuple representing the new board state
            reward: float indicating the reward for this action
            done: boolean indicating if the game is over
        """
        row, col = action
        if self.board[row, col] != 0:
            return self._get_state(), -10, True  # Invalid move penalty
        
        self.board[row, col] = self.current_player
        
        winner = self._check_winner()
        done = winner is not None
        
        # Reward structure:
        # Win: +1
        # Draw: +0.5
        # Loss: -1
        # Invalid move: -10
        # Ongoing game: 0
        if winner == 1:  # Agent wins
            reward = 1
        elif winner == -1:  # Opponent wins
            reward = -1
        elif winner == 0:  # Draw
            reward = 0.5
        else:  # Game continues
            reward = 0
            
        self.current_player *= -1  # Switch players
        return self._get_state(), reward, done

class QLearningAgent:
    """
    Q-learning agent that learns to play Tic Tac Toe through experience.
    
    The agent maintains a Q-table mapping state-action pairs to expected rewards.
    It learns by updating these Q-values based on the rewards it receives and
    its estimates of future rewards.
    """
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        # Initialize Q-table as a nested defaultdict
        # This allows us to access any state-action pair without explicitly creating it
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.lr = learning_rate  # How quickly the agent updates its Q-values
        self.gamma = discount_factor  # How much the agent values future rewards
        self.epsilon = epsilon  # Probability of choosing a random action (exploration)
        
        # Performance tracking
        self.games_played = 0
        self.wins = 0
        self.draws = 0
        self.losses = 0
    
    def get_action(self, state, valid_moves):
        """
        Choose action using epsilon-greedy policy.
        
        With probability epsilon, choose a random action (exploration).
        Otherwise, choose the action with the highest Q-value (exploitation).
        """
        if random.random() < self.epsilon:  # Exploration
            return random.choice(valid_moves)
        
        # Exploitation: choose best known action
        return max(valid_moves, key=lambda a: self.q_table[state][a])
    
    def learn(self, state, action, reward, next_state, next_valid_moves):
        """
        Update Q-value using the Q-learning update rule:
        Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        
        Where:
        - α (learning rate) determines how much we update our Q-value
        - r is the immediate reward
        - γ (discount factor) determines the importance of future rewards
        - max Q(s',a') is our estimate of optimal future value
        """
        # Calculate max future Q-value (0 if game is over)
        if next_valid_moves:
            next_max_q = max(self.q_table[next_state][a] for a in next_valid_moves)
        else:
            next_max_q = 0
            
        # Current Q-value for this state-action pair
        current_q = self.q_table[state][action]
        
        # Update Q-value using Q-learning formula
        self.q_table[state][action] = current_q + self.lr * (
            reward + self.gamma * next_max_q - current_q
        )
    
    def update_stats(self, reward):
        """Update performance statistics based on game outcome"""
        self.games_played += 1
        if reward == 1:
            self.wins += 1
        elif reward == 0.5:
            self.draws += 1
        elif reward == -1:
            self.losses += 1

def train_agent(episodes=10000, plot_progress=True):
    """
    Train the Q-learning agent through self-play and track its progress.
    
    Args:
        episodes: Number of training games to play
        plot_progress: Whether to display a learning curve
    """
    env = TicTacToeEnv()
    agent = QLearningAgent()
    
    # Track performance over time
    window_size = 100
    moving_avg_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Agent's turn (X)
            valid_moves = env.get_valid_moves()
            action = agent.get_action(state, valid_moves)
            next_state, reward, done = env.step(action)
            episode_reward += reward
            
            # If game is not over, simulate opponent's move (O)
            if not done:
                opponent_moves = env.get_valid_moves()
                opponent_action = random.choice(opponent_moves)
                next_state, reward, done = env.step(opponent_action)
                # Invert reward since opponent's success is agent's failure
                reward = -reward
                episode_reward += reward
            
            # Learn from the transition
            agent.learn(state, action, reward, next_state, env.get_valid_moves())
            state = next_state
        
        # Update agent statistics
        agent.update_stats(episode_reward)
        
        # Track moving average of rewards
        if episode >= window_size:
            avg_reward = (agent.wins - agent.losses) / window_size
            moving_avg_rewards.append(avg_reward)
            agent.wins = agent.draws = agent.losses = 0
        
        # Decay epsilon for less exploration over time
        agent.epsilon = max(0.01, agent.epsilon * 0.995)
    
    if plot_progress and moving_avg_rewards:
        plt.figure(figsize=(10, 6))
        plt.plot(moving_avg_rewards)
        plt.title('Agent Learning Progress')
        plt.xlabel('Episodes (hundreds)')
        plt.ylabel('Average Reward')
        plt.grid(True)
        plt.show()
    
    return agent

def play_game(agent, human_player=-1):
    """
    Play an interactive game against the trained agent.
    
    Args:
        agent: Trained QLearningAgent instance
        human_player: 1 for X, -1 for O
    """
    env = TicTacToeEnv()
    state = env.reset()
    done = False
    
    def print_board():
        """Display the current game board"""
        symbols = {0: ' ', 1: 'X', -1: 'O'}
        print('\n')
        for i in range(3):
            print('-------------')
            row = '|'
            for j in range(3):
                row += f' {symbols[env.board[i,j]]} |'
            print(row)
        print('-------------')
    
    while not done:
        print_board()
        current_player = env.current_player
        
        if current_player == human_player:
            valid_moves = env.get_valid_moves()
            print("\nValid moves (row, col):", valid_moves)
            while True:
                try:
                    row = int(input("Enter row (0-2): "))
                    col = int(input("Enter col (0-2): "))
                    if (row, col) in valid_moves:
                        break
                    print("Invalid move, try again")
                except ValueError:
                    print("Invalid input, try again")
            action = (row, col)
        else:
            valid_moves = env.get_valid_moves()
            action = agent.get_action(state, valid_moves)
            print(f"\nAgent plays: {action}")
        
        state, reward, done = env.step(action)
        
        if done:
            print_board()
            winner = env.current_player * -1  # Winner is the last player who moved
            if winner == 1:
                print("\nX wins!")
            elif winner == -1:
                print("\nO wins!")
            else:
                print("\nIt's a draw!")

# Example usage
if __name__ == "__main__":
    # Train the agent with progress tracking
    print("Training agent...")
    agent = train_agent(episodes=10000, plot_progress=True)
    print("Training complete!")
    
    # Display final statistics
    print(f"\nFinal training statistics:")
    print(f"Games played: {agent.games_played}")
    print(f"Q-table size: {len(agent.q_table)} states")
    
    # Play against the trained agent
    play_game(agent, human_player=-1)  # Human plays as O