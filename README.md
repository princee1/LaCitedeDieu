# LaCitedeDieu

## Overview
**LaCiteDeDieu** is a project that showcases the final product of a collaborative effort. This repository serves as a presentation of the completed work, highlighting the culmination of development efforts and the final state of the project.

The original repository, which contains the full history of commits, development progress, and all intermediate stages of the project, is not included here. Instead, this repository is intended solely for showcasing the final product.

## Purpose
The purpose of this repository is to:
- Provide a clean and concise view of the final product.
- Serve as a demonstration of the work completed in collaboration with my compadre.
- Allow others to explore the final state of the project without the clutter of development history.

## Original Repository
The original repository contains:
- The complete commit history.
- All stages of development and progress.
- Detailed insights into the evolution of the project.
- **Collaborative Effort**: The project was developed in collaboration with my compadre @https://github.com/Mbaka11, showcasing teamwork and shared vision.

If you are interested in the development process, commit history, or intermediate stages, please refer to the original repository. https://github.com/Mbaka11/INF8175-AI_AGENT


## How It Works

### Game Overview
**LaCiteDeDieu** is a strategic board game where players compete to build cities and manage resources on a grid-based board. The game emphasizes strategic placement, resource management, and tactical decision-making to outmaneuver opponents and achieve victory.

### Gameplay Mechanics
1. **Board Layout**: The game board is a 9x9 grid with predefined positions for resources and cities.
2. **Players**: Two players compete, each controlling pieces of their respective colors (White and Black).
3. **Objective**: Players aim to maximize their score by strategically placing resources and cities while blocking their opponent's moves.
4. **Actions**: Players can perform light actions (e.g., placing resources) or heavy actions (e.g., building cities).
5. **Scoring**: Points are awarded based on resource placement, city placement, and strategic control of the board.

### How to Play
1. **Setup**: 
    - Launch the game using the provided GUI or command-line interface.
    - Each player is assigned a color (White or Black).
2. **Turns**:
    - Players take turns performing actions such as placing resources or building cities.
    - The game alternates between light and heavy actions.
3. **Winning**:
    - The game ends when all possible actions are exhausted or a predefined number of turns are completed.
    - The player with the highest score wins.

## AI Agent Development

### Overview
The AI agent for **LaCiteDeDieu** was designed to simulate intelligent gameplay, leveraging advanced algorithms and heuristics to make strategic decisions. The AI can play against human players or other AI agents.

### Key Components
1. **Game State Representation**:
    - The `GameStateDivercite` class models the current state of the board, including player scores, available actions, and board layout.
    - It provides methods to generate possible actions, apply actions, and compute scores.

2. **Heuristics**:
    - Custom heuristics were implemented to evaluate board states and guide the AI's decision-making process.
    - Examples include:
      - **ScoreHeuristic**: Evaluates the potential score of a move.
      - **ControlIndexHeuristic**: Measures control over key areas of the board.
      - **DiverciteHeuristic**: Balances resource placement and city-building strategies.

3. **Search Algorithms**:
    - The AI uses search algorithms like Minimax and Monte Carlo Tree Search (MCTS) to explore possible moves and predict outcomes.
    - Alpha-beta pruning optimizes the Minimax algorithm by reducing the number of nodes evaluated.

4. **Caching and Optimization**:
    - An LRUCache is used to store previously evaluated states, improving efficiency.
    - Symmetry detection reduces redundant calculations by identifying equivalent board states.

### Challenges and Solutions
- **Challenge**: Balancing AI difficulty to provide a challenging yet fair experience.
  - **Solution**: Fine-tuned heuristics and search depth to adjust the AI's skill level.
- **Challenge**: Optimizing performance for large search spaces.
  - **Solution**: Implemented caching and symmetry detection to reduce redundant calculations.

## How to Run the Project

### Prerequisites
- Python 3.10 or higher.
- Required Python libraries (install via `pip install -r requirements.txt`).
- A modern web browser for the GUI.

### Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/Mbaka11/INF8175-AI_AGENT.git
    cd LaCiteDeDieu
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Launch the game:
    ```bash
    python main_divercite.py
    ```
4. Open the GUI:
    - Navigate to `GUI/index.html` in your browser.
    - Follow the on-screen instructions to start a match.

## Credits
This project was developed collaboratively by:
- [Princee1](https://github.com/princee1)
- [Mbaka11](https://github.com/Mbaka11)

Special thanks to the INF8175 course for inspiring this project and providing the foundation for its development.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
