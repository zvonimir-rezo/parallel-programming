import sys
import time
from board import Board

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank

BOARD_CREATOR = 0
HUMAN = 1
CPU = 2
DEPTH = 6


def evaluate(current, last_mover, last_col, depth):
    b_all_lose = True
    b_all_win = True

    ended, winner = current.game_end(last_col)
    if ended:
        if last_mover == CPU:
            return 1
        else:
            return -1
    if depth == 0:
        return 0
    depth -= 1
    if last_mover == CPU:
        new_mover = HUMAN
    else:
        new_mover = CPU

    d_total = 0
    i_moves = 0

    for i_col in range(current.cols):
        if current.move_legal(i_col):
            i_moves += 1
            current.move(i_col, new_mover)
            d_result = evaluate(current, new_mover, i_col, depth)
            current.undo_move(i_col)
            if d_result > -1:
                b_all_lose = False
            if d_result != 1:
                b_all_win = False
            if d_result == 1 and new_mover == CPU:
                return 1
            if d_result == -1 and new_mover == HUMAN:
                return -1
            d_total += d_result

    if b_all_win:
        return 1
    if b_all_lose:
        return -1
    d_total /= i_moves
    return d_total


def game_ended(b):
    for i in range(7):
        ended, winner = b.game_end(i)
        if ended:
            return True, winner
    return False, -1


def check_ended(b):
    ended, winner = game_ended(b)
    if ended:
        if winner == HUMAN:
            print("Human is the winner!")
        elif winner == CPU:
            print("CPU is the winner!")
        sys.stdout.flush()
        exit(0)


def move_and_print_board(b, mover, col):
    b.move(col, mover)
    print(b)
    sys.stdout.flush()


def main():
    if rank == BOARD_CREATOR:
        b = Board(6, 7)

        check_ended(b)

        while True:
            sys.stdout.flush()
            i_depth = DEPTH

            col_move = int(input("Make move in column:"))
            start_time = time.time()
            move_and_print_board(b, HUMAN, col_move)
            check_ended(b)

            worker_index = 1
            non_legal_counter = 0

            d_best = -1
            while i_depth > 0 and d_best == -1:
                d_best = -1
                i_best_col = -1
                for i_col in range(b.cols):
                    if b.move_legal(i_col):
                        if i_best_col == -1:
                            i_best_col = i_col
                        b.move(i_col, CPU)
                        ended, _ = game_ended(b)
                        if ended:
                            d_best = 1
                            i_best_col = i_col
                            b.undo_move(i_col)
                            move_and_print_board(b, CPU, i_best_col)
                            print(f"Time for CPU move: {time.time() - start_time}")
                            sys.stdout.flush()
                            check_ended(b)
                        for d_col in range(b.cols):
                            if b.move_legal(d_col):
                                b.move(d_col, HUMAN)
                                comm.send((b, HUMAN, d_col, i_depth-1, i_col), dest=worker_index)
                                worker_index = (worker_index + 1) % size
                                if worker_index == 0:
                                    worker_index = 1
                                b.undo_move(d_col)
                            else:
                                non_legal_counter += 1
                        b.undo_move(i_col)
                    else:
                        non_legal_counter += b.cols

                r = b.cols * b.cols - non_legal_counter
                result_dict = {}
                for i in range(b.cols):
                    result_dict[i] = -2
                for i in range(r):
                    result, col = comm.recv(source=MPI.ANY_SOURCE)
                    if (result > result_dict[col] != -1) or result == -1:
                        result_dict[col] = result
                for key, value in result_dict.items():
                    if value > d_best:
                        d_best = value
                        i_best_col = key
                i_depth /= 2

            move_and_print_board(b, CPU, i_best_col)

            print(f"Time for CPU move: {time.time()-start_time}")
            sys.stdout.flush()

            check_ended(b)

    # WORKERS
    if rank != BOARD_CREATOR:
        while True:
            board, last_mover, last_col, depth, last_last_col = comm.recv(source=BOARD_CREATOR)
            result = evaluate(board, last_mover, last_col, depth)
            comm.send((result, last_last_col), dest=0)


main()