import sys
from time import sleep
from mpi4py import MPI
from random import random


class Fork:
    def __init__(self, fork_id, dirty, taken):
        self.id = fork_id
        self.dirty = dirty
        self.taken = taken


comm = MPI.COMM_WORLD

size = comm.size
rank = comm.Get_rank()

left = (rank - 1) % size

right = (rank + 1) % size

forks = []

spaces = str(rank) + " " * rank
requests = []


def fork_request(fork_no, side):
    print(f"{spaces} trazim vilicu ({fork_no})")
    comm.send((fork_no, 0), dest=side, tag=1)
    sys.stdout.flush()


def ask_for_forks():
    n_sent = 0
    n_taken, id_taken = number_taken()
    if n_taken == 0:
        fork_request(rank, left)
        fork_request(right, right)
        n_sent += 2
    elif n_taken == 1:
        if id_taken == rank:
            fork_request(right, right)
        else:
            fork_request(rank, left)
        n_sent += 1
    return n_sent


def answer_if_eaten(fork_id, side, done):
    for fork in forks:
        if fork_id == fork.id:
            if fork.dirty and fork.taken:
                fork.taken = False
                fork.dirty = False
                comm.send((fork_id, 1), dest=side, tag=1)
                done = True
    return done


def answer_req(fork_id, side):
    print(f"{spaces} drugi traze moju vilicu ({fork_id})")
    sys.stdout.flush()
    done = answer_if_eaten(fork_id, side, False)

    if not done:
        requests.append(((fork_id, 1), side))


def number_taken():
    c = 0
    fork_id = -1
    for i in forks:
        if i.taken:
            c += 1
            fork_id = i.id
    return c, fork_id


def take_fork(fork_id):
    print(f"{spaces} dobio vilicu ({fork_id})")
    sys.stdout.flush()
    for fork in forks:
        if fork_id == fork.id:
            fork.taken = True


def add_my_forks():
    if rank == 0:
        forks.append(Fork(rank, dirty=True, taken=True))
        forks.append(Fork(right, dirty=True, taken=True))
    elif rank == size - 1:
        forks.append(Fork(rank, dirty=True, taken=False))
        forks.append(Fork(right, dirty=True, taken=False))
    else:
        forks.append(Fork(rank, dirty=True, taken=False))
        forks.append(Fork(right, dirty=True, taken=True))


def eat():
    print(f"{spaces} jedem")
    sys.stdout.flush()
    sleep(2)


def main():
    add_my_forks()

    while True:
        print(f"{spaces} mislim")
        sys.stdout.flush()
        sec = int(random() * 5 + 2)
        for i in range(0, sec):
            sleep(1)
            if comm.Iprobe(source=left, tag=1):
                id, answer = comm.recv(source=left, tag=1)
                answer_req(id, left)
            if comm.Iprobe(source=right, tag=1):
                id, answer = comm.recv(source=right, tag=1)
                answer_req(id, right)
        n_taken, id_taken = number_taken()
        while n_taken < 2:
            n_sent_reqs = ask_for_forks()
            while n_sent_reqs > 0:
                if comm.Iprobe(source=left, tag=1):
                    id, answer = comm.recv(source=left, tag=1)
                    if answer:
                        take_fork(id)
                        n_sent_reqs -= 1
                    else:
                        answer_req(id, left)
                elif comm.Iprobe(source=right, tag=1):
                    id, answer = comm.recv(source=right, tag=1)
                    if answer:
                        take_fork(id)
                        n_sent_reqs -= 1
                    else:
                        answer_req(id, right)
            n_taken, id_taken = number_taken()

        if n_taken == 2:
            eat()
        forks[0].dirty = True
        forks[1].dirty = True

        for m, dest in requests:
            for fork in forks:
                if fork.id == m[0]:
                    fork.taken = False
                    fork.dirty = False
                    comm.send(m, dest=dest, tag=1)
        requests.clear()


main()
