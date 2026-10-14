"""ChessGPT: play chess against a GPT that learned the game only by reading move lists.

Run locally:            python app/app.py            (then open http://127.0.0.1:7860)
Temporary public link:  python app/app.py --share    (about 72 hours, to send to friends)
Play by clicking a piece and then its destination square.
"""

# Command-line arguments (--share).
import sys

# Check which arguments the installed Gradio version accepts.
import inspect

# Square root for the attention scale.
import math

# File paths relative to this file.
from pathlib import Path

# Numerical computing.
import numpy as np

# PyTorch: tensors and neural networks.
import torch

# Neural network layers.
import torch.nn as nn

# Softmax.
import torch.nn.functional as F

# Chess rules and SAN.
import chess

# SVG drawings of chess boards.
import chess.svg

# The web interface.
import gradio as gr

# Folder of this file (works locally and on Hugging Face Spaces).
HERE = Path(__file__).resolve().parent


# Find a file next to app.py (Hugging Face Space) or where the notebooks save it (local use).
def find_file(name):
    """Return the path of the first existing candidate, or None."""

    # Candidate locations, in order: next to app.py, notebooks/data/lichess/, data/lichess/.
    for folder in [HERE, HERE.parent / "notebooks" / "data" / "lichess", HERE.parent / "data" / "lichess"]:

        # Path in this folder.
        path = folder / name

        # Found.
        if path.exists():

            # Use it.
            return path

    # Not found anywhere.
    return None

# =====================================================================
# MODEL (identical to chessgpt_v5.ipynb)
# =====================================================================


# Multi-head causal self-attention.
class CausalSelfAttention(nn.Module):

    # Declare the weights.
    def __init__(self, n_embd, n_head):

        # Initialise the parent class.
        super().__init__()

        # The embedding size must split evenly into the heads.
        assert n_embd % n_head == 0

        # Number of heads.
        self.n_head = n_head

        # Dimensions per head (d_k).
        self.head_dim = n_embd // n_head

        # W_Q, W_K and W_V for all heads in one matrix: d -> 3d.
        self.qkv = nn.Linear(n_embd, 3 * n_embd)

        # Output projection W_O: d -> d.
        self.proj = nn.Linear(n_embd, n_embd)

    # Attention over a sequence (optionally continuing from cached keys and values).
    def forward(self, x, cache=None, return_attn=False):
        """x: (B, T, d). Returns output (B, T, d), the (keys, values) for a cache, and the attention map if asked."""

        # Batch size, new positions, embedding size.
        B, T, C = x.shape

        # Queries, keys and values of the new positions, each (B, T, d).
        q, k, v = self.qkv(x).split(C, dim=2)

        # Split d into heads and put the heads first: (B, T, d) -> (B, T, H, d_k) -> (B, H, T, d_k).
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Same for the keys.
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Same for the values.
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # With a cache: prepend the keys and values of the positions already processed.
        if cache is not None:

            # Old keys + new keys along the time axis.
            k = torch.cat([cache[0], k], dim=2)

            # Old values + new values.
            v = torch.cat([cache[1], v], dim=2)

        # Total number of positions that can be looked at (old + new).
        T_k = k.shape[2]

        # Scores q . k / sqrt(d_k) for every pair (new position, any position): (B, H, T, T_k).
        # Dividing q (small) instead of the score matrix (large) gives the same result and saves one big T x T tensor.
        att = (q / math.sqrt(self.head_dim)) @ k.transpose(-2, -1)

        # Absolute position of every query: the new positions are the last T of the T_k.
        query_pos = torch.arange(T, device=x.device)[:, None] + (T_k - T)

        # Absolute position of every key.
        key_pos = torch.arange(T_k, device=x.device)[None, :]

        # Causal mask: a query may look at keys at its own position or before (True = allowed).
        allowed = key_pos <= query_pos

        # Forbidden scores become -infinity, i.e. weight 0 after the softmax ("_": in place, again to save memory).
        att.masked_fill_(~allowed, float("-inf"))

        # Attention weights alpha: each row sums to 1.
        att = F.softmax(att, dim=-1)

        # Weighted average of the values: (B, H, T, T_k) @ (B, H, T_k, d_k) -> (B, H, T, d_k).
        y = att @ v

        # Put the heads back side by side: (B, H, T, d_k) -> (B, T, H, d_k) -> (B, T, d).
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection, keys and values (for the cache), attention map (only if asked).
        return self.proj(y), (k, v), (att if return_attn else None)


# One Transformer block: communication (attention) + computation (MLP), each with a residual connection.
class Block(nn.Module):

    # Declare the sub-layers.
    def __init__(self, n_embd, n_head):

        # Initialise the parent class.
        super().__init__()

        # LayerNorm before attention.
        self.ln1 = nn.LayerNorm(n_embd)

        # Multi-head causal self-attention.
        self.attn = CausalSelfAttention(n_embd, n_head)

        # LayerNorm before the MLP.
        self.ln2 = nn.LayerNorm(n_embd)

        # MLP: d -> 4d -> GELU -> d.
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd))

    # Forward pass of the block.
    def forward(self, x, cache=None, return_attn=False):

        # Attention on the normalised input.
        a, kv, att = self.attn(self.ln1(x), cache, return_attn)

        # Residual connection around attention.
        x = x + a

        # Residual connection around the MLP.
        x = x + self.mlp(self.ln2(x))

        # New representation, keys/values, attention map.
        return x, kv, att


# The full move-level GPT.
class GPT(nn.Module):

    # Declare the layers.
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer):

        # Initialise the parent class.
        super().__init__()

        # Maximum context length.
        self.block_size = block_size

        # Token embedding E: one vector per move.
        self.tok_emb = nn.Embedding(vocab_size, n_embd)

        # Position embedding P: one vector per position 0..block_size-1.
        self.pos_emb = nn.Embedding(block_size, n_embd)

        # The stack of Transformer blocks.
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])

        # Final LayerNorm.
        self.ln_f = nn.LayerNorm(n_embd)

        # Output layer: one logit per possible next move.
        self.head = nn.Linear(n_embd, vocab_size)

        # GPT-2 initialisation of every Linear and Embedding layer.
        self.apply(self._init_weights)

        # Weight tying: the output layer uses the embedding matrix itself (same tensor, kept in sync automatically).
        self.head.weight = self.tok_emb.weight

        # Smaller initialisation for the projections that add into the residual path.
        for name, p in self.named_parameters():

            # W_O of attention and the second linear layer of the MLP.
            if name.endswith("attn.proj.weight") or name.endswith("mlp.2.weight"):

                # Standard deviation 0.02 / sqrt(2L).
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    # Initialisation rule for one module.
    def _init_weights(self, module):

        # Linear layers.
        if isinstance(module, nn.Linear):

            # Weights from N(0, 0.02^2).
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            # Biases at zero.
            if module.bias is not None:

                # Zero.
                nn.init.zeros_(module.bias)

        # Embedding tables.
        elif isinstance(module, nn.Embedding):

            # Vectors from N(0, 0.02^2).
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # Residual stream after the first n_blocks blocks (0 = embeddings only), used by the linear probes.
    def hidden(self, idx, n_blocks):
        """idx: (B, T) token indices. Returns the hidden states (B, T, d)."""

        # Absolute positions 0..T-1.
        pos = torch.arange(idx.shape[1], device=idx.device)

        # Token embedding + position embedding.
        x = self.tok_emb(idx) + self.pos_emb(pos)

        # Only the first n_blocks blocks.
        for block in self.blocks[:n_blocks]:

            # Block output (keys, values and attention maps are not needed).
            x, _, _ = block(x)

        # The residual stream at that depth.
        return x

    # Forward pass.
    def forward(self, idx, cache=None, use_cache=False, return_attn=False):
        """idx: (B, T) token indices. Returns logits (B, T, V), the new cache (if use_cache), the attention maps (if return_attn)."""

        # Batch size and number of new positions.
        B, T = idx.shape

        # Positions already in the cache (0 without a cache).
        offset = 0 if cache is None else cache[0][0].shape[2]

        # The context must fit in the position table.
        assert offset + T <= self.block_size, "sequence longer than the context of the model"

        # Absolute positions of the new characters.
        pos = torch.arange(offset, offset + T, device=idx.device)

        # Token embedding + position embedding: (B, T, d) + (T, d) broadcast over the batch.
        x = self.tok_emb(idx) + self.pos_emb(pos)

        # Keys and values of every block (for the cache).
        new_cache = []

        # Attention map of every block.
        attn_maps = []

        # Through the blocks.
        for i, block in enumerate(self.blocks):

            # Block i, with its part of the cache if there is one.
            x, kv, att = block(x, None if cache is None else cache[i], return_attn)

            # Keep its keys and values.
            new_cache.append(kv)

            # Keep its attention map (None if not asked).
            attn_maps.append(att)

        # Final LayerNorm and output layer.
        logits = self.head(self.ln_f(x))

        # Logits, cache and attention maps (None when not asked for).
        return logits, (new_cache if use_cache else None), (attn_maps if return_attn else None)

# Checkpoint written by chessgpt_v5.ipynb.
MODEL_PATH = find_file("chessgpt_v5.pt")

# Stop with a clear message if it is missing.
assert MODEL_PATH is not None, "chessgpt_v5.pt not found: run notebooks/chessgpt_v5.ipynb, or copy the file next to app.py."

# Load it on the CPU.
checkpoint = torch.load(MODEL_PATH, map_location="cpu")

# Vocabulary of moves, in index order.
vocab = checkpoint["vocab"]

# Model configuration.
config = checkpoint["config"]

# Move -> index.
stoi = {m: i for i, m in enumerate(vocab)}

# Special tokens (as in v5).
PAD_ID, UNK_ID, SEP_ID = 0, 1, 2

# Index of the first real move.
N_SPECIAL = 3

# Maximum context in plies.
BLOCK = config["block"]

# Rebuild the model.
model = GPT(len(vocab), BLOCK, config["n_embd"], config["n_head"], config["n_layer"])

# Load the trained weights.
model.load_state_dict(checkpoint["model_state"])

# Evaluation mode.
model.eval()

# Inference only: no gradients for the weights (this also holds in the threads Gradio uses for each request).
model.requires_grad_(False)

# =====================================================================
# THE PROBE FOR THE "MIND'S EYE" (saved by chessgpt_v5.ipynb)
# =====================================================================

# Probe file (None if missing: the app still works, without the mind's eye).
PROBE_PATH = find_file("chessgpt_v5_probe.pt")

# None until a probe file is loaded.
probe = None

# Load it if present.
if PROBE_PATH is not None:

    # Saved weights and standardisation.
    probe_data = torch.load(PROBE_PATH, map_location="cpu")

    # One linear map d -> 64 squares x 13 classes.
    probe = nn.Linear(config["n_embd"], 64 * 13)

    # Trained weights.
    probe.load_state_dict(probe_data["state"])

    # Inference only.
    probe.requires_grad_(False)

    # Mean of the hidden states.
    probe_mean = probe_data["mean"].float()

    # Standard deviation of the hidden states.
    probe_std = probe_data["std"].float()

    # Depth of the hidden states (number of blocks).
    probe_depth = probe_data["depth"]

# =====================================================================
# MODEL HELPERS
# =====================================================================


# Token indices of a game prefix: [;, m1, ..., mk].
def context_ids(moves):
    """List of SAN moves -> list of token indices (unknown moves become <unk>)."""

    # Start token and the moves.
    return [SEP_ID] + [stoi.get(m, UNK_ID) for m in moves]


# Probabilities of the next move.
def next_move_probs(moves):
    """Return a tensor of V probabilities after the given moves."""

    # Token indices, at most the last BLOCK (only in games longer than 255 plies).
    ids = context_ids(moves)[-BLOCK:]

    # One forward pass.
    logits, _, _ = model(torch.tensor([ids]))

    # Softmax of the last position.
    return F.softmax(logits[0, -1], dim=0)


# Draw one item from a list of (item, probability), with temperature (0 = the most likely).
def draw(items, probs, temperature):
    """items: list; probs: 1-D tensor of the same length."""

    # Greedy choice.
    if temperature == 0:

        # Most likely item.
        return items[int(torch.argmax(probs))]

    # p^(1/T), renormalised.
    p = probs ** (1.0 / temperature)

    # One draw.
    return items[int(torch.multinomial(p / p.sum(), 1))]


# Choose ChessGPT's move.
def choose_move(board, moves, temperature, legal_only):
    """Return (SAN of the move to play, message, candidates for the display, probability on legal moves)."""

    # Distribution over the whole vocabulary.
    probs = next_move_probs(moves)

    # Canonical SAN of every legal move.
    legal_sans = [board.san(m) for m in board.legal_moves]

    # Legal moves that have a token.
    known = [s for s in legal_sans if s in stoi]

    # Total probability on legal moves.
    legal_mass = float(sum(probs[stoi[s]] for s in known))

    # The 8 most likely tokens, for the display.
    top_p, top_i = torch.topk(probs, 8)

    # Labels for the display: illegal moves are marked.
    candidates = {}

    # Each candidate.
    for p, i in zip(top_p.tolist(), top_i.tolist()):

        # Token text (special tokens shown as they are).
        name = vocab[i]

        # Mark it if it is not a legal move here.
        label = name if name in legal_sans else f"{name} (illegal)"

        # Probability.
        candidates[label] = p

    # Unrestricted mode: sample from the whole vocabulary first.
    if not legal_only:

        # One token.
        token = draw(list(range(len(vocab))), probs, temperature)

        # Its text.
        san = vocab[token]

        # A correct legal move: play it.
        if san in legal_sans:

            # Done.
            return san, f"ChessGPT played **{san}**.", candidates, legal_mass

        # Try a lenient parse (e.g. "Bxe4" without a capture).
        try:

            # Parsed move (raises ValueError if illegal or not a move).
            move = board.parse_san(san) if token >= N_SPECIAL else None

        # Not a legal move at all.
        except ValueError:

            # Nothing.
            move = None

        # Legal but written wrongly.
        if move:

            # Play the canonical version.
            return board.san(move), f"ChessGPT wrote **{san}** (wrong notation) and played **{board.san(move)}**.", candidates, legal_mass

        # Illegal: remember it and fall back to a legal move below.
        illegal_note = f"ChessGPT first tried **{san}**, which is illegal here; it then picked among the legal moves. "

    # Legal-only mode (or after an illegal attempt).
    else:

        # No illegal attempt.
        illegal_note = ""

    # Legal moves without a token (very rare): uniform choice among all legal moves.
    if not known:

        # Random legal move.
        san = legal_sans[int(torch.randint(len(legal_sans), (1,)))]

        # Report.
        return san, illegal_note + f"ChessGPT played **{san}** (random: no legal move in its vocabulary).", candidates, legal_mass

    # Probabilities of the legal moves only (masking): the model's preferences among the possible moves.
    legal_probs = torch.stack([probs[stoi[s]] for s in known])

    # Draw one of them.
    san = draw(known, legal_probs, temperature)

    # Report.
    return san, illegal_note + f"ChessGPT played **{san}**.", candidates, legal_mass


# Turn 64 relative labels (1-6 = player to move, 7-12 = opponent) into a board.
def labels_to_board(labels, turn):
    """Return a chess.Board with only the pieces."""

    # Empty board.
    b = chess.Board(None)

    # Every square.
    for square in range(64):

        # Its class.
        c = int(labels[square])

        # Not empty.
        if c:

            # Piece type 1-6; colour of the player to move for classes 1-6.
            b.set_piece_at(square, chess.Piece((c - 1) % 6 + 1, turn if c <= 6 else (not turn)))

    # The board.
    return b


# The board read from the model's hidden states, as SVG.
def minds_eye(board, moves, orientation):
    """Return HTML with the board the model 'sees', or a note if no probe is available."""

    # No probe file, or a game longer than the context.
    if probe is None or len(moves) + 1 > BLOCK:

        # Nothing to show.
        return "<i>Mind's eye not available: chessgpt_v5_probe.pt not found (it is saved by the last cell of chessgpt_v5.ipynb).</i>"

    # Hidden state after reading the moves, at the depth of the probe.
    h = model.hidden(torch.tensor([context_ids(moves)]), probe_depth)[0, -1]

    # Most likely content of every square.
    labels = probe((h - probe_mean) / probe_std).view(64, 13).argmax(-1)

    # As a board.
    seen = labels_to_board(labels, board.turn)

    # Squares where the model's picture is wrong.
    wrong = [s for s in range(64) if board.piece_at(s) != seen.piece_at(s)]

    # Drawing with the wrong squares in red.
    svg = chess.svg.board(seen, size=300, orientation=orientation, fill={s: "#e0505099" for s in wrong})

    # Caption and drawing.
    return f"<b>The board ChessGPT \"sees\"</b> (linear probe on block {probe_depth}): {64 - len(wrong)}/64 squares right<br>{svg}"


# =====================================================================
# GAME STATE AND INTERFACE
# =====================================================================


# Board after a list of SAN moves.
def replay(moves):
    """Return the chess.Board after the moves."""

    # Starting position.
    board = chess.Board()

    # Each move.
    for m in moves:

        # Play it.
        board.push_san(m)

    # The position.
    return board


# Moves as text with move numbers.
def move_text(moves):
    """['e4', 'e5', 'Nf3'] -> '1. e4 e5 2. Nf3'."""

    # Number before White's moves.
    return " ".join((f"{i // 2 + 1}. " if i % 2 == 0 else "") + m for i, m in enumerate(moves))


# A fresh game state.
def empty_state(human_white=True):
    """Moves, colour, counters, selected square, and the last information about ChessGPT's move."""

    # Everything the interface needs to redraw itself.
    return {"moves": [], "human_white": human_white, "illegal": 0, "model_moves": 0, "selected": None,
            "message": "New game: your move.", "candidates": None, "legal_mass": None}


# All the outputs of the interface for a state.
def render(state, note=""):
    """Return the values of: state, board, status, candidates, mind's eye, move list."""

    # Current position.
    board = replay(state["moves"])

    # Orientation: the human's colour at the bottom.
    orientation = state["human_white"]

    # Last move, highlighted on the board.
    last = board.peek() if board.move_stack else None

    # King in check, highlighted.
    check = board.king(board.turn) if board.is_check() else None

    # Coloured squares: the selected piece in yellow, its legal destinations in green.
    fill = {}

    # A piece is selected.
    if state["selected"] is not None:

        # Its square.
        fill[state["selected"]] = "#f6e05ecc"

        # Every legal move of that piece.
        for m in board.legal_moves:

            # Moves starting from the selected square.
            if m.from_square == state["selected"]:

                # Destination in green.
                fill[m.to_square] = "#68d391cc"

    # Main board (each square of the SVG has a class with its name, used by the click script).
    board_svg = chess.svg.board(board, size=440, orientation=orientation, lastmove=last, check=check, fill=fill)

    # Status: the last message about ChessGPT's move, and an optional note about the click.
    status = state["message"] + (f"  \n*{note}*" if note else "")

    # Probability on legal moves of the last model move.
    if state["legal_mass"] is not None:

        # Add it.
        status += f"  \n{state['legal_mass']:.0%} of ChessGPT's probability was on legal moves."

    # Illegal attempts so far.
    status += f"  \nIllegal attempts: {state['illegal']} out of {state['model_moves']} ChessGPT moves."

    # Game over.
    if board.is_game_over():

        # Result and reason.
        status += f"  \n**Game over: {board.result()} ({board.outcome().termination.name.lower()}).**"

    # All outputs.
    return (state, board_svg, status, state["candidates"], minds_eye(board, state["moves"], orientation),
            move_text(state["moves"]))


# ChessGPT plays one move in the state, if it is its turn.
def model_turn(state, temperature, legal_only):
    """Update the state with ChessGPT's move and the information about it."""

    # Current position.
    board = replay(state["moves"])

    # Nothing to do if the game is over or it is the human's turn.
    if board.is_game_over() or board.turn == state["human_white"]:

        # No change.
        return

    # Choose the move.
    san, message, candidates, legal_mass = choose_move(board, state["moves"], temperature, legal_only)

    # Count the moves of ChessGPT.
    state["model_moves"] += 1

    # Count the illegal attempts.
    state["illegal"] += "illegal here" in message

    # Play it (canonical SAN).
    state["moves"].append(san)

    # Remember what to show.
    state["message"], state["candidates"], state["legal_mass"] = message, candidates, legal_mass


# Start a new game.
def new_game(colour, temperature, legal_only):
    """colour: 'White' or 'Black' (the human's colour)."""

    # Fresh state.
    state = empty_state(colour == "White")

    # If the human plays Black, ChessGPT moves first.
    model_turn(state, temperature, legal_only)

    # Outputs.
    return render(state)


# A click on a square of the board: select a piece, or move the selected piece there.
def on_click(square_name, state, temperature, legal_only):
    """square_name: name of the clicked square, e.g. 'e4' (sent by the script in the page)."""

    # Current position.
    board = replay(state["moves"])

    # Ignore clicks when the game is over or it is not the human's turn.
    if board.is_game_over() or board.turn != state["human_white"]:

        # Redraw unchanged.
        return render(state)

    # Ignore anything that is not a square name.
    if square_name not in chess.SQUARE_NAMES:

        # Redraw unchanged.
        return render(state)

    # Clicked square as an index 0-63.
    square = chess.parse_square(square_name)

    # Second click: a piece is already selected.
    if state["selected"] is not None:

        # Legal moves from the selected square to the clicked one (several only for promotions).
        moves = [m for m in board.legal_moves if m.from_square == state["selected"] and m.to_square == square]

        # It is a legal move: play it.
        if moves:

            # For a promotion choose the queen; otherwise there is a single move.
            move = next((m for m in moves if m.promotion in (None, chess.QUEEN)), moves[0])

            # Deselect.
            state["selected"] = None

            # Play the human move (canonical SAN).
            state["moves"].append(board.san(move))

            # ChessGPT answers.
            model_turn(state, temperature, legal_only)

            # Outputs.
            return render(state)

    # First click (or a click elsewhere): select the piece if it is one of yours and can move.
    piece = board.piece_at(square)

    # A piece of the player to move with at least one legal move.
    if piece and piece.color == board.turn and any(m.from_square == square for m in board.legal_moves):

        # Select it.
        state["selected"] = square

        # Show its destinations.
        return render(state, f"{chess.piece_name(piece.piece_type).capitalize()} on {square_name} selected: click a green square.")

    # Anything else: deselect.
    state["selected"] = None

    # Redraw with a hint.
    return render(state, "Click one of your pieces, then the square where it should go.")


# Script added to the page: when a square of the board is clicked, send its name to Python.
HEAD = """
<script>
// Listen to every click on the page.
document.addEventListener("click", function (event) {

  // Only clicks inside the main board matter.
  if (!event.target.closest("#chessboard")) { return; }

  // All the elements under the mouse: the square (an SVG rect) lies below the piece.
  const stack = document.elementsFromPoint(event.clientX, event.clientY);

  // Name of the clicked square, read from its classes ("square light e4").
  let name = null;

  // Look for the square.
  for (const el of stack) {

    // The first element with class "square".
    if (el.classList && el.classList.contains("square")) {

      // The class that looks like a square name.
      name = Array.from(el.classList).find(c => /^[a-h][1-8]$/.test(c));

      // Stop searching.
      break;
    }
  }

  // Not on a square (e.g. on the coordinates).
  if (!name) { return; }

  // Remember it: the hidden button reads it (see js= in the Python code).
  window.chessClick = name;

  // Press the hidden button, which calls on_click in Python.
  document.querySelector("#click_btn").click();
});
</script>
"""

# Hide the helper components (they must exist in the page, but nobody needs to see them).
CSS = "#click_box, #click_btn {display: none !important;}"

# Text at the top of the page.
INTRO = """
# ChessGPT
A small GPT (6.3M parameters) that learned chess **only by reading 370,000 games as lists of moves** (Lichess, 2013).
It was never told the rules, never shown a board.
- **Click one of your pieces, then the square where it should go** (green squares). Pawns promote to a queen.
- **Only legal moves** on: ChessGPT chooses among the legal moves, using its own probabilities. Off: it samples from everything it knows, and sometimes tries an illegal move (you will see it).
- **The board ChessGPT "sees"**: a linear probe reads the position out of the model's activations; red squares are where its picture is wrong.
"""

# Gradio 6 takes the custom CSS and <head> code in launch(); Gradio 4-5 took them in Blocks().
STYLE_IN_LAUNCH = "head" in inspect.signature(gr.Blocks.launch).parameters

# The style arguments, given to whichever of the two accepts them.
STYLE = {"head": HEAD, "css": CSS}

# The interface.
with gr.Blocks(title="ChessGPT", **({} if STYLE_IN_LAUNCH else STYLE)) as demo:

    # Title and explanation.
    gr.Markdown(INTRO)

    # Game state.
    state = gr.State(empty_state())

    # Two columns.
    with gr.Row():

        # Left: the game.
        with gr.Column(scale=3):

            # Main board (its id is used by the click script).
            board_html = gr.HTML(elem_id="chessboard")

            # New game controls.
            with gr.Row():

                # Colour of the human.
                colour = gr.Radio(["White", "Black"], value="White", label="You play")

                # New game button.
                new_btn = gr.Button("New game", variant="primary")

            # Moves of the game.
            history = gr.Textbox(label="Moves", lines=3, interactive=False)

        # Right: ChessGPT's mind.
        with gr.Column(scale=2):

            # Masking option.
            legal_only = gr.Checkbox(value=True, label="ChessGPT may only choose legal moves")

            # Temperature.
            temperature = gr.Slider(0, 1.5, value=0.5, step=0.1, label="Temperature (0 = always its favourite move)")

            # Status and messages.
            status = gr.Markdown()

            # Candidate moves with probabilities.
            candidates = gr.Label(label="The 8 moves ChessGPT liked most", num_top_classes=8)

            # The board read from the model.
            eye_html = gr.HTML()

    # Hidden helpers for the clicks: a textbox for the square name and a button that sends it.
    click_box = gr.Textbox(elem_id="click_box")

    # Hidden button pressed by the script.
    click_btn = gr.Button("click", elem_id="click_btn")

    # Outputs updated by every action, in the order returned by render().
    outputs = [state, board_html, status, candidates, eye_html, history]

    # A click on the board: js= replaces the textbox value with the square the script stored, then on_click runs.
    click_btn.click(on_click, inputs=[click_box, state, temperature, legal_only], outputs=outputs,
                    js="(square, ...rest) => [window.chessClick, ...rest]")

    # New game.
    new_btn.click(new_game, inputs=[colour, temperature, legal_only], outputs=outputs)

    # Start a game when the page opens.
    demo.load(new_game, inputs=[colour, temperature, legal_only], outputs=outputs)

# Run the app when the file is executed directly (this is what Hugging Face Spaces does).
if __name__ == "__main__":

    # Start the web server; "--share" also creates a temporary public link.
    demo.launch(share="--share" in sys.argv, **(STYLE if STYLE_IN_LAUNCH else {}))
