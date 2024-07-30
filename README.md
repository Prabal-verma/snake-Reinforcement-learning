
# Snake Game with Reinforcement Learning

This project implements a classic Snake game where the snake learns to play the game using reinforcement learning. The implementation leverages PyTorch for the RL model, and various Python libraries for game mechanics, plotting, and display.

## Features

- **Reinforcement Learning**: The snake learns to play the game using deep Q-learning.
- **Real-time Visualization**: Scores and mean scores are plotted in real-time during training.
- **Interactive Gameplay**: The game can be watched as the snake learns to improve its performance over time.

## Dependencies

The project uses the following libraries:

- [PyTorch](https://pytorch.org/) - For implementing the reinforcement learning model.
- [Matplotlib](https://matplotlib.org/) - For plotting scores and mean scores.
- [IPython](https://ipython.org/) - For displaying plots in Jupyter Notebooks.
- [NumPy](https://numpy.org/) - For numerical operations.
- [Pygame](https://www.pygame.org/) - For implementing the game mechanics.

## Installation

To run this project, you need to install the required libraries. You can do this using `pip`:

```bash
pip install torch matplotlib ipython numpy pygame
```

## Usage

1. **Clone the repository**:

    ```bash
    git clone https://github.com/Prabal-verma/snake-Reinforcement-learning.git
    cd snake-rl
    ```

2. **Run the training script**:

    ```bash
    python agent.py
    ```

3. **Watch the training process**:

    The training script will display the game window where the snake is learning to play the game. Scores and mean scores will be plotted in real-time.

## Code Overview

### `agent.py`

This script initializes the game and the reinforcement learning agent. It handles the training loop where the agent plays the game, learns from its actions, and updates its policy based on the rewards received.

### `game.py`

This module contains the implementation of the Snake game using Pygame. It defines the game mechanics, including how the snake moves, eats food, and checks for collisions.

### `model.py`

This module defines the neural network model used for the reinforcement learning agent. It uses PyTorch to build and train the model.

### `helper.py`

This module contains helper functions for plotting scores and mean scores during training.

## Example

Here's a snippet of how the training loop might look:

```python
# Main game loop
if __name__ == '__main__':
    game = SnakeGameAI()
    agent = DQNAgent()

    scores = []
    mean_scores = []
    total_score = 0
    record = 0

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            mean_scores.append(mean_score)
            plot(scores, mean_scores)
```

## Contributing

If you find any issues or have suggestions for improvements, feel free to open an issue or create a pull request. Contributions are welcome!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
