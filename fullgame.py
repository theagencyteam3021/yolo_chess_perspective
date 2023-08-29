from chess import set_orientation, get_board, next_move, to_fen
from chessMove2 import move, drop_capture
#from button_utils import wait_for_button
drop_capture()
M, rect_base = set_orientation(True)

input()
d = {'a':0,'b':1,'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7}
CONFIRM_BOARD = True


while True:
    drop_capture()
    board = get_board(M, rect_base)
    fen, val = to_fen(board)  
    if val:
        start, end, capture = next_move(fen)
        print(start,end,capture)
        if CONFIRM_BOARD:
            veto = input()
        else:
            veto = False
        if not veto:
            move(start,end,capture)
        else:
            vetos = veto.split(" ")
            for v in vetos:
                print(v)
                if len(v) >= 3:
                   #board[d[v[0].lower()]][8-int(v[1])] = v[2] if v[2].lower() != 'x' else ' '
                   board[8-int(v[1])][d[v[0].lower()]] = v[2] if v[2].lower() != 'x' else ' '
            fen, val = to_fen(board)
            if val:
                s, e, c = next_move(fen)
                print(s,e,c)
                veto2 = input()
                if not veto2:
                    move(s,e,c)
        #move(next_move(fen))
        drop_capture()
        #wait_for_button()
        input()
    else:
        input('Invalid board.')
