---
title: ChessGPT
emoji: ♟️
colorFrom: gray
colorTo: indigo
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
---

# ChessGPT

A 6.3M-parameter GPT that learned chess only by reading 370,000 Lichess games as lists of moves: it was never told the rules and never shown a board.

- Play against it, letting it choose **only among legal moves** or **freely** (then it sometimes tries an illegal move).
- See the 8 moves it liked most, with their probabilities.
- See **the board it "sees"**: a linear probe reads the position from its activations; red squares are where its picture is wrong.

Code and training notebooks: see the GitHub repository.
