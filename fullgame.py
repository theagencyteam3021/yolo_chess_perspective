from chess import set_orientation, get_board, next_move, to_fen
from chessMove2 import move, drop_capture
drop_capture()
M, rect_base = set_orientation(True)
cmd = ""
input()
while cmd.lower() not in ["quit","q"]:
    drop_capture()
    fen, val = to_fen(get_board(M, rect_base))
    if val:
        start, end, capture = next_move(fen)
        print(start,end,capture)
        input()
        move(start,end,capture)
        #move(next_move(fen))
        cmd = input()
    else:
        input('Invalid board.')
