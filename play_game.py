import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
import joblib

model = joblib.load("regressor_model.pkl")

import chess

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9
}

CENTER_SQUARES = [chess.D4, chess.E4, chess.D5, chess.E5]


def extract_features(board: chess.Board) -> list:
    features = []

    material_white = 0
    material_black = 0
    piece_counts = {(color, ptype): 0 for color in [chess.WHITE, chess.BLACK]
                    for ptype in PIECE_VALUES.keys()}
    center_control_white = 0
    center_control_black = 0
    white_in_black_half = 0
    black_in_white_half = 0
    pawn_files = {chess.WHITE: set(), chess.BLACK: set()}
    all_pawns = {chess.WHITE: [], chess.BLACK: []}
    open_files = set(range(8))
    halfopen_files = {chess.WHITE: set(), chess.BLACK: set()}
    outpost_knights = {chess.WHITE: 0, chess.BLACK: 0}
    developed_minors = {chess.WHITE: 0, chess.BLACK: 0}
    defended_pieces = {chess.WHITE: 0, chess.BLACK: 0}

    attackers_cache = {square: board.attackers(chess.WHITE, square) | board.attackers(chess.BLACK, square) for square in
                       chess.SQUARES}

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if not piece:
            continue
        color = piece.color
        ptype = piece.piece_type
        rank = chess.square_rank(square)
        file = chess.square_file(square)

        # Material
        value = PIECE_VALUES.get(ptype, 0)
        if color == chess.WHITE:
            material_white += value
        else:
            material_black += value

        # Figurenzahl
        if ptype in PIECE_VALUES:
            piece_counts[(color, ptype)] += 1

        # Zentrum
        if square in CENTER_SQUARES:
            if color == chess.WHITE:
                center_control_white += 1
            else:
                center_control_black += 1

        # Gegnerische Hälfte
        if color == chess.WHITE and rank >= 4:
            white_in_black_half += 1
        elif color == chess.BLACK and rank <= 3:
            black_in_white_half += 1

        # Bauernanalyse
        if ptype == chess.PAWN:
            pawn_files[color].add(file)
            all_pawns[color].append(square)
            open_files.discard(file)
            halfopen_files[color].add(file)

        # Vorposten-Springer (grob: in Gegnerhälfte, nicht leicht durch Bauer vertreibbar)
        if ptype == chess.KNIGHT and ((color == chess.WHITE and rank >= 4) or (color == chess.BLACK and rank <= 3)):
            opp_color = not color
            # Kein gegnerischer Bauer kann dieses Feld angreifen (rudimentär)
            attackers = board.attackers(opp_color, square)
            if not any(board.piece_at(a).piece_type == chess.PAWN for a in attackers if board.piece_at(a)):
                outpost_knights[color] += 1

        # Entwicklung (Leichtfiguren nicht auf Grundreihe)
        if ptype in [chess.BISHOP, chess.KNIGHT] and (
                (color == chess.WHITE and rank > 0) or (color == chess.BLACK and rank < 7)):
            developed_minors[color] += 1

        # Verteidigte Figuren
        if len(board.attackers(color, square)) > 0:
            defended_pieces[color] += 1

    # Material
    features.append(material_white)
    features.append(material_black)
    features.append(material_white - material_black)

    # Figurenzählung
    for ptype in PIECE_VALUES:
        features.append(piece_counts[(chess.WHITE, ptype)])
        features.append(piece_counts[(chess.BLACK, ptype)])

    # Rochaden
    features.append(int(board.has_kingside_castling_rights(chess.WHITE)))
    features.append(int(board.has_queenside_castling_rights(chess.WHITE)))
    features.append(int(board.has_kingside_castling_rights(chess.BLACK)))
    features.append(int(board.has_queenside_castling_rights(chess.BLACK)))

    # Königsposition
    features.append(board.king(chess.WHITE))
    features.append(board.king(chess.BLACK))

    # Spieler am Zug
    features.append(int(board.turn))

    # Zentrum
    features.append(center_control_white)
    features.append(center_control_black)

    # Figuren in gegnerischer Hälfte
    features.append(white_in_black_half)
    features.append(black_in_white_half)

    # Mobilität
    features.append(len(list(board.legal_moves)))
    features.append(board.fullmove_number)

    # Neue Features (Reihenfolge beibehalten für separate Liste!)
    # 1. Doppelte Bauern
    features.append(sum(
        len([sq for sq in all_pawns[color] if chess.square_file(sq) == file]) > 1
        for color in [chess.WHITE, chess.BLACK]
        for file in pawn_files[color]
    ))

    # 2. Isolierte Bauern
    features.append(sum(
        all(
            file + offset not in pawn_files[color]
            for offset in [-1, 1]
        )
        for color in [chess.WHITE, chess.BLACK]
        for file in pawn_files[color]
    ))

    # 3. Halboffene Linien für Türme (eigene Bauern fehlen, Gegner evtl. nicht)
    for color in [chess.WHITE, chess.BLACK]:
        features.append(len(halfopen_files[color]))

    # 4. Offene Linien
    features.append(len(open_files))

    # 5. Springer auf Vorposten
    features.append(outpost_knights[chess.WHITE])
    features.append(outpost_knights[chess.BLACK])

    # 6. Entwicklung
    features.append(developed_minors[chess.WHITE])
    features.append(developed_minors[chess.BLACK])

    # 7. Verteidigte Figuren
    features.append(defended_pieces[chess.WHITE])
    features.append(defended_pieces[chess.BLACK])

    return features


def negamax(board, alpha, beta, model, depth=4):
    if depth == 0 or board.is_game_over():
        features = extract_features(board)
        eval = model.predict([features])[0]
        # Negiere, falls schwarzer Spieler am Zug
        score = eval if board.turn == chess.WHITE else -eval
        return score, None  # <<< WICHTIG: Rückgabe immer als Tupel!

    max_eval = -float('inf')
    best_move = None

    for move in board.legal_moves:
        board.push(move)
        eval, _ = negamax(board, -beta, -alpha, model, depth - 1)
        eval = -eval
        board.pop()

        if eval > max_eval:
            max_eval = eval
            best_move = move

        alpha = max(alpha, eval)
        if alpha >= beta:
            break  # Beta-Cutoff

    return max_eval, best_move  # <<< Rückgabe als Tupel



import chess.svg
board = chess.Board()

OUTPUT_FILE = "Board.svg"
# Spiel-Loop
while not board.is_game_over():
    print(board)
    print("Your move (e.g., e2e4):")

    user_move = None
    while True:
        move_input = input(">> ").strip()
        try:
            user_move = chess.Move.from_uci(move_input)
            if user_move in board.legal_moves:
                board.push(user_move)
                break
            else:
                print("Illegal move. Try again.")
        except:
            print("Invalid format. Try again (e.g., e2e4).")

    # SVG nach Nutzerzug speichern
    with open(OUTPUT_FILE, "w") as f:
        f.write(chess.svg.board(board))

    if board.is_game_over():
        break

    print("Computer is thinking...")
    _, best_move = negamax(board, -float("inf"), float("inf"), model, depth=2)
    board.push(best_move)
    print(f"Computer plays: {best_move}")

    # SVG nach Computzug speichern
    with open(OUTPUT_FILE, "w") as f:
        f.write(chess.svg.board(board))

# Spielende anzeigen
print(board)
print("Game over:", board.result())