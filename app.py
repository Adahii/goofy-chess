import streamlit as st
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Tuple
import uuid
import random

# ---------- Types & Globals ----------

Color = str  # "white" or "black"
PieceType = str  # "king","queen","rook","bishop","knight","pawn"
GameMode = str  # "classic","spy","absorption"

BOARD_SIZE = 8

# Global in-memory room store (shared across users on same server)
ROOMS: Dict[str, "GameState"] = {}


@dataclass
class Piece:
    id: str
    type: PieceType
    color: Color              # visible color on the board
    row: int
    col: int

    # Spy mode
    true_owner: Optional[Color] = None   # who secretly owns it (for spies)
    is_spy: bool = False

    # Absorption mode
    absorbed: Set[PieceType] = field(default_factory=set)


@dataclass
class GameState:
    room_code: str
    mode: GameMode
    board: List[List[Optional[Piece]]]
    current_player: Color = "white"
    players: Dict[Color, Optional[str]] = field(default_factory=lambda: {"white": None, "black": None})
    winner: Optional[Color] = None
    move_history: List[str] = field(default_factory=list)


# ---------- Utility helpers ----------

def generate_room_code() -> str:
    return "".join(random.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(5))


def next_player(color: Color) -> Color:
    return "black" if color == "white" else "white"


def in_bounds(row: int, col: int) -> bool:
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def piece_symbol(piece: Piece) -> str:
    symbols_white = {
        "king": "♔",
        "queen": "♕",
        "rook": "♖",
        "bishop": "♗",
        "knight": "♘",
        "pawn": "♙",
    }
    symbols_black = {
        "king": "♚",
        "queen": "♛",
        "rook": "♜",
        "bishop": "♝",
        "knight": "♞",
        "pawn": "♟",
    }
    table = symbols_white if piece.color == "white" else symbols_black
    return table[piece.type]


def square_name(row: int, col: int) -> str:
    file = "abcdefgh"[col]
    rank = str(8 - row)
    return file + rank


# ---------- Board Setup ----------

def empty_board() -> List[List[Optional[Piece]]]:
    return [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]


def add_piece(board: List[List[Optional[Piece]]], piece_type: PieceType, color: Color, row: int, col: int) -> Piece:
    p = Piece(
        id=uuid.uuid4().hex,
        type=piece_type,
        color=color,
        row=row,
        col=col,
        true_owner=None,
        is_spy=False,
        absorbed=set(),
    )
    board[row][col] = p
    return p


def create_starting_board() -> List[List[Optional[Piece]]]:
    board = empty_board()

    # Back ranks
    for color, row_back, row_pawn in [("white", 7, 6), ("black", 0, 1)]:
        # Rooks
        add_piece(board, "rook", color, row_back, 0)
        add_piece(board, "rook", color, row_back, 7)
        # Knights
        add_piece(board, "knight", color, row_back, 1)
        add_piece(board, "knight", color, row_back, 6)
        # Bishops
        add_piece(board, "bishop", color, row_back, 2)
        add_piece(board, "bishop", color, row_back, 5)
        # Queen & King
        add_piece(board, "queen", color, row_back, 3)
        add_piece(board, "king", color, row_back, 4)
        # Pawns
        for col in range(BOARD_SIZE):
            add_piece(board, "pawn", color, row_pawn, col)

    return board


def assign_spies(game: GameState) -> None:
    """Assign one spy per side: a random non-king, non-queen piece that secretly belongs to the opponent."""
    if game.mode != "spy":
        return

    board = game.board

    def choose_candidate(owner_color: Color) -> Optional[Piece]:
        candidates: List[Piece] = []
        for row in board:
            for p in row:
                if p and p.color == owner_color and p.type not in ("king", "queen"):
                    candidates.append(p)
        if not candidates:
            return None
        return random.choice(candidates)

    white_spy_host = choose_candidate("white")
    black_spy_host = choose_candidate("black")

    if white_spy_host:
        # This white-looking piece secretly belongs to black
        white_spy_host.true_owner = "black"
        white_spy_host.is_spy = True

    if black_spy_host:
        # Black-looking piece secretly belongs to white
        black_spy_host.true_owner = "white"
        black_spy_host.is_spy = True


# ---------- Move Generation ----------

Move = Tuple[int, int]


def slide_moves(board: List[List[Optional[Piece]]],
                piece: Piece,
                directions: List[Tuple[int, int]],
                treat_color: Color) -> List[Move]:
    moves: List[Move] = []
    for dr, dc in directions:
        r, c = piece.row + dr, piece.col + dc
        while in_bounds(r, c):
            dest = board[r][c]
            if dest is None:
                moves.append((r, c))
            else:
                if dest.color != treat_color:
                    moves.append((r, c))
                break
            r += dr
            c += dc
    return moves


def generate_moves_for_type(board: List[List[Optional[Piece]]],
                            piece: Piece,
                            as_type: PieceType,
                            treat_color: Color) -> List[Move]:
    moves: List[Move] = []

    if as_type == "rook":
        moves.extend(slide_moves(board, piece, [(-1, 0), (1, 0), (0, -1), (0, 1)], treat_color))
    elif as_type == "bishop":
        moves.extend(slide_moves(board, piece, [(-1, -1), (-1, 1), (1, -1), (1, 1)], treat_color))
    elif as_type == "queen":
        moves.extend(slide_moves(board, piece,
                                 [(-1, 0), (1, 0), (0, -1), (0, 1),
                                  (-1, -1), (-1, 1), (1, -1), (1, 1)],
                                 treat_color))
    elif as_type == "knight":
        for dr, dc in [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                       (1, -2), (1, 2), (2, -1), (2, 1)]:
            r, c = piece.row + dr, piece.col + dc
            if not in_bounds(r, c):
                continue
            dest = board[r][c]
            if dest is None or dest.color != treat_color:
                moves.append((r, c))
    elif as_type == "king":
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                r, c = piece.row + dr, piece.col + dc
                if not in_bounds(r, c):
                    continue
                dest = board[r][c]
                if dest is None or dest.color != treat_color:
                    moves.append((r, c))
    elif as_type == "pawn":
        direction = -1 if treat_color == "white" else 1
        start_row = 6 if treat_color == "white" else 1

        # Forward 1
        r1, c1 = piece.row + direction, piece.col
        if in_bounds(r1, c1) and board[r1][c1] is None:
            moves.append((r1, c1))
            # Forward 2 from starting rank
            r2 = piece.row + 2 * direction
            if piece.row == start_row and in_bounds(r2, c1) and board[r2][c1] is None:
                moves.append((r2, c1))

        # Diagonal captures
        for dc in [-1, 1]:
            r, c = piece.row + direction, piece.col + dc
            if not in_bounds(r, c):
                continue
            dest = board[r][c]
            if dest is not None and dest.color != treat_color:
                moves.append((r, c))

    return moves


def generate_legal_moves_for_piece(game: GameState, piece: Piece, current_player: Color) -> List[Move]:
    board = game.board
    # Spy rule: if current_player is the true_owner, treat the spy as their color for this move
    treat_color = piece.color
    if game.mode == "spy" and piece.is_spy and piece.true_owner == current_player:
        treat_color = current_player

    movement_types: Set[PieceType] = {piece.type}
    if game.mode == "absorption" and piece.absorbed:
        movement_types.update(piece.absorbed)

    all_moves: List[Move] = []
    for t in movement_types:
        all_moves.extend(generate_moves_for_type(board, piece, t, treat_color))

    # Deduplicate
    seen = set()
    unique_moves: List[Move] = []
    for m in all_moves:
        if m not in seen:
            seen.add(m)
            unique_moves.append(m)
    return unique_moves


# ---------- Game Mechanics ----------

def get_game(room_code: str) -> Optional[GameState]:
    return ROOMS.get(room_code)


def create_game(room_code: str, mode: GameMode) -> GameState:
    board = create_starting_board()
    game = GameState(room_code=room_code, mode=mode, board=board)
    if mode == "spy":
        assign_spies(game)
    ROOMS[room_code] = game
    return game


def find_piece_at(board: List[List[Optional[Piece]]], row: int, col: int) -> Optional[Piece]:
    if not in_bounds(row, col):
        return None
    return board[row][col]


def make_move(game: GameState,
              from_rc: Tuple[int, int],
              to_rc: Tuple[int, int],
              player_color: Color) -> Tuple[bool, str]:
    """Attempt to make a move. Returns (success, message)."""
    if game.winner is not None:
        return False, "Game is already over."

    if game.current_player != player_color:
        return False, "Not your turn."

    fr, fc = from_rc
    tr, tc = to_rc
    board = game.board

    piece = find_piece_at(board, fr, fc)
    if piece is None:
        return False, "No piece on that square."

    # Check if player is allowed to move this piece
    can_move = False
    if piece.color == player_color:
        can_move = True
    if game.mode == "spy" and piece.is_spy and piece.true_owner == player_color:
        # Spy's real owner can move it too (this move reveals it)
        can_move = True

    if not can_move:
        return False, "You can't move that piece."

    legal_moves = generate_legal_moves_for_piece(game, piece, player_color)
    if (tr, tc) not in legal_moves:
        return False, "Illegal move for that piece (prototype doesn't enforce check, only movement)."

    dest_piece = find_piece_at(board, tr, tc)

    # Absorption: capture gives new movement pattern
    if dest_piece is not None and game.mode == "absorption":
        if piece.type != dest_piece.type:
            piece.absorbed.add(dest_piece.type)

    # Move the piece
    board[fr][fc] = None

    # Spy reveal: if real owner moves it, flip color and turn off spy flag
    if game.mode == "spy" and piece.is_spy and piece.true_owner == player_color:
        piece.color = player_color
        piece.is_spy = False
        piece.true_owner = None

    piece.row = tr
    piece.col = tc
    board[tr][tc] = piece

    # Promotion: auto-queen when pawn reaches last rank
    if piece.type == "pawn":
        if (piece.color == "white" and piece.row == 0) or (piece.color == "black" and piece.row == 7):
            piece.type = "queen"

    # Very simple end condition: if a king is captured, game over
    if dest_piece is not None and dest_piece.type == "king":
        game.winner = player_color

    move_str = f"{player_color}: {piece.type} {square_name(fr, fc)} -> {square_name(tr, tc)}"
    if dest_piece:
        move_str += f" x {dest_piece.type}"
    if game.mode == "absorption" and piece.absorbed:
        move_str += f" [{'/'.join(sorted(piece.absorbed))}]"
    game.move_history.append(move_str)

    game.current_player = next_player(game.current_player)
    return True, "Move played."


# ---------- Streamlit State & Setup ----------

def init_session():
    if "player_id" not in st.session_state:
        st.session_state["player_id"] = uuid.uuid4().hex
    if "room_code" not in st.session_state:
        st.session_state["room_code"] = None
    if "player_color" not in st.session_state:
        st.session_state["player_color"] = None
    if "selected_square" not in st.session_state:
        st.session_state["selected_square"] = None
    if "legal_moves" not in st.session_state:
        st.session_state["legal_moves"] = []  # list of (r,c)
    if "last_message" not in st.session_state:
        st.session_state["last_message"] = ""


def inject_board_css():
    # Chess.com-ish theme (green + beige), square sizing, highlights
    st.markdown(
        """
<style>
/* Make the main page darkish and centered */
.main > div {
    padding-top: 1rem;
}

/* Style all board buttons to be square-ish */
div[data-testid="stButton"] > button {
    width: 64px;
    height: 64px;
    padding: 0;
    border-radius: 0;
    border: none;
    font-size: 36px;
    line-height: 1;
}

/* Light & dark squares using CSS variables + classes on wrapper divs */
.chess-square-light > div[data-testid="stButton"] > button {
    background-color: #EEEED2;  /* light beige */
    color: #000000;
}
.chess-square-dark > div[data-testid="stButton"] > button {
    background-color: #769656;  /* green */
    color: #000000;
}

/* Selected square outline */
.chess-square-selected > div[data-testid="stButton"] > button {
    box-shadow: 0 0 0 3px #f6f669 inset;
}

/* Legal moves highlight (soft ring) */
.chess-square-legal > div[data-testid="stButton"] > button {
    box-shadow: 0 0 0 3px rgba(255, 255, 0, 0.4) inset;
}

/* Coordinates row/col labels */
.chess-coord {
    font-size: 14px;
    text-align: center;
    color: #9ca3af;
    margin-top: 0.2rem;
}
</style>
        """,
        unsafe_allow_html=True,
    )


# ---------- Host / Join UI ----------

def host_or_join_ui():
    st.sidebar.header("🎮 Game Setup")

    choice = st.sidebar.radio("Choose:", ["Host game", "Join game"])

    if choice == "Host game":
        mode_label = st.sidebar.selectbox(
            "Game mode",
            ["Classic", "Spy", "Absorption"],
            help="Spy: one of your pieces is secretly your opponent's.\nAbsorption: captured movement abilities stack."
        )
        mode_map = {"Classic": "classic", "Spy": "spy", "Absorption": "absorption"}
        mode = mode_map[mode_label]

        color_choice = st.sidebar.radio("Play as", ["White", "Black"])
        host_color = "white" if color_choice == "White" else "black"

        if st.sidebar.button("Create room"):
            room_code = generate_room_code()
            game = create_game(room_code, mode)
            game.players[host_color] = st.session_state["player_id"]

            st.session_state["room_code"] = room_code
            st.session_state["player_color"] = host_color
            st.session_state["selected_square"] = None
            st.session_state["legal_moves"] = []
            st.session_state["last_message"] = "Room created. Share the code with your friend."

    else:  # Join game
        room_code_input = st.sidebar.text_input("Room code", max_chars=5).upper()
        preferred = st.sidebar.radio("Preferred color", ["Auto", "White", "Black"])

        if st.sidebar.button("Join room"):
            if room_code_input not in ROOMS:
                st.session_state["last_message"] = "Room not found."
            else:
                game = ROOMS[room_code_input]
                # Assign color
                if preferred == "White":
                    try_color_order = ["white", "black"]
                elif preferred == "Black":
                    try_color_order = ["black", "white"]
                else:
                    try_color_order = ["white", "black"]

                assigned = None
                for c in try_color_order:
                    if game.players[c] is None:
                        game.players[c] = st.session_state["player_id"]
                        assigned = c
                        break

                if assigned is None:
                    st.session_state["last_message"] = "Room is already full."
                else:
                    st.session_state["room_code"] = room_code_input
                    st.session_state["player_color"] = assigned
                    st.session_state["selected_square"] = None
                    st.session_state["legal_moves"] = []
                    st.session_state["last_message"] = f"Joined as {assigned}."


# ---------- Board Rendering & Interaction ----------

def render_game_ui():
    room_code = st.session_state["room_code"]
    player_color = st.session_state["player_color"]
    if not room_code or room_code not in ROOMS:
        st.write("Create or join a game from the sidebar.")
        return

    game = ROOMS[room_code]

    st.subheader(f"Room: `{room_code}`  |  Mode: {game.mode.capitalize()}")

    # Player info
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown(f"**You are:** `{player_color}`")
        st.markdown(f"**Current turn:** `{game.current_player}`")
        if game.winner:
            st.success(f"Game over! Winner: {game.winner}")

    with col_info2:
        players_display = []
        for c in ["white", "black"]:
            pid = game.players[c]
            label = "connected" if pid is not None else "waiting"
            players_display.append(f"{c}: {label}")
        st.markdown("**Players:**  \n" + "  \n".join(players_display))

    # Variant hints
    if game.mode == "spy":
        st.info("Spy mode: one of your opponent's pieces secretly belongs to you.\n"
                "Click your pieces to move. If you click your hidden spy (shown only to you below), it flips to your color.")
    elif game.mode == "absorption":
        st.info("Absorption mode: when a piece captures a *different* piece type, it permanently gains that movement ability.")

    # Spy info (only show your own spy, if any)
    if game.mode == "spy":
        spy_squares = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                p = game.board[r][c]
                if p and p.is_spy and p.true_owner == player_color:
                    spy_squares.append(square_name(r, c))
        if spy_squares:
            st.markdown(f"🕵️ Your hidden spy is at: `{', '.join(spy_squares)}`")
        else:
            st.markdown("🕵️ You currently have **no active spy** (or it has already been revealed / captured).")

    st.markdown("*(Prototype note: the engine enforces piece movement but **does not enforce check/checkmate** yet.)*")

    st.divider()

    selected = st.session_state["selected_square"]
    legal_moves = st.session_state["legal_moves"]
    board = game.board

    # Board coordinates (files a–h)
    files_row = st.columns(9)
    files_row[0].markdown(" ")
    for c in range(BOARD_SIZE):
        files_row[c + 1].markdown(f"<div class='chess-coord'>{'abcdefgh'[c]}</div>", unsafe_allow_html=True)

    # Render 8 ranks (top rank 8 at row=0)
    for r in range(BOARD_SIZE):
        row_cols = st.columns(9)
        # Rank label
        row_cols[0].markdown(
            f"<div class='chess-coord'>{8 - r}</div>",
            unsafe_allow_html=True,
        )
        for c in range(BOARD_SIZE):
            piece = board[r][c]
            label = piece_symbol(piece) if piece else " "
            is_selected = selected == (r, c)
            is_legal = (r, c) in legal_moves

            # Square color (like chess.com)
            is_dark = (r + c) % 2 == 1
            square_classes = []
            square_classes.append("chess-square-dark" if is_dark else "chess-square-light")
            if is_selected:
                square_classes.append("chess-square-selected")
            if is_legal:
                square_classes.append("chess-square-legal")

            wrapper = row_cols[c + 1].container()
            # Apply CSS class wrapper around the button
            wrapper.markdown(
                f"<div class=\"{' '.join(square_classes)}\">",
                unsafe_allow_html=True,
            )

            btn_key = f"square-{r}-{c}-{room_code}"
            if wrapper.button(label, key=btn_key):
                handle_square_click(game, (r, c), player_color)
                st.experimental_rerun()

            wrapper.markdown("</div>", unsafe_allow_html=True)

    # Move log
    st.divider()
    st.subheader("Move history")
    if game.move_history:
        for m in game.move_history:
            st.write(m)
    else:
        st.write("No moves yet.")

    # Message
    if st.session_state["last_message"]:
        st.info(st.session_state["last_message"])

    # Small reset/leave controls
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("Reset selection"):
            st.session_state["selected_square"] = None
            st.session_state["legal_moves"] = []
            st.session_state["last_message"] = ""
            st.experimental_rerun()
    with col_b:
        if st.button("Leave room"):
            st.session_state["room_code"] = None
            st.session_state["player_color"] = None
            st.session_state["selected_square"] = None
            st.session_state["legal_moves"] = []
            st.session_state["last_message"] = "Left the room."
            st.experimental_rerun()
    with col_c:
        if st.button("Refresh"):
            st.experimental_rerun()


def handle_square_click(game: GameState, rc: Tuple[int, int], player_color: Color):
    """Click handler: first click = select, second click = attempt move."""
    if game.winner is not None:
        st.session_state["last_message"] = "Game over."
        return

    selected = st.session_state["selected_square"]
    r, c = rc
    board = game.board
    piece = find_piece_at(board, r, c)

    # First click: select a piece
    if selected is None:
        if piece is None:
            st.session_state["last_message"] = "Click one of your pieces to move."
            return

        # You can select your own pieces; in Spy mode, you may also select your spy on opponent's side
        can_select = False
        if piece.color == player_color:
            can_select = True
        if game.mode == "spy" and piece.is_spy and piece.true_owner == player_color:
            can_select = True

        if not can_select:
            st.session_state["last_message"] = "You can't move that piece."
            return

        st.session_state["selected_square"] = (r, c)
        # Pre-compute legal moves for highlight
        st.session_state["legal_moves"] = generate_legal_moves_for_piece(game, piece, player_color)
        st.session_state["last_message"] = f"Selected {piece.type} on {square_name(r, c)}."
        return

    # Second click: attempt to move from selected -> rc
    fr, fc = selected
    if (fr, fc) == (r, c):
        # Deselect
        st.session_state["selected_square"] = None
        st.session_state["legal_moves"] = []
        st.session_state["last_message"] = "Selection cleared."
        return

    success, msg = make_move(game, (fr, fc), (r, c), player_color)
    st.session_state["selected_square"] = None
    st.session_state["legal_moves"] = []
    st.session_state["last_message"] = msg


# ---------- Main ----------

def main():
    st.set_page_config(page_title="Variant Chess – Spy & Absorption", layout="wide")
    inject_board_css()
    st.title("♟ Variant Chess – Spy & Absorption (Multiplayer)")
    init_session()
    host_or_join_ui()
    render_game_ui()


if __name__ == "__main__":
    main()
