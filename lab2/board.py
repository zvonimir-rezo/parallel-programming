
class Board:

    def __init__(self, n_rows, n_cols):
        m = []
        for i in range(n_rows):
            l = [0 for _ in range(n_cols)]
            m.append(l)
        self.field = m
        self.game_ended = False
        self.cols = n_cols
        self.rows = n_rows
        self.height = [0 for _ in range(n_cols)]
        self.last_mover = -1
        self.last_col = -1

    def move_legal(self, col):
        if col >= self.cols:
            raise ValueError("Column is out of bounds1")
        if self.field[self.rows-1][col] != 0:
            return False
        return True

    def move(self, col, player):
        if not self.move_legal(col):
            return False
        self.field[self.height[col]][col] = player
        self.height[col] += 1
        self.last_mover = player
        self.last_col = col
        return True

    def undo_move(self, col):
        if col >= self.cols:
            raise ValueError("Column is out of bounds2")
        if self.height[col] == 0:
            return False
        self.field[self.height[col]-1][col] = 0
        self.height[col] -= 1
        return True

    def game_end(self, last_col):
        if last_col >= self.cols:
            raise ValueError("Column is out of bound3")
        col = last_col
        row = self.height[last_col] - 1
        if row < 0:
            return False, -1
        player = self.field[row][col]

        # uspravno
        seq = 1
        r = row - 1
        while r >= 0 and self.field[r][col] == player:
            seq += 1
            r -= 1
        if seq > 3:
            return True, player

        # vodoravno
        seq = 0
        c = col
        while (c-1) >= 0 and self.field[row][c-1] == player:
            c -= 1
        while c < self.cols and self.field[row][c] == player:
            seq += 1
            c += 1
        if seq > 3:
            return True, player

        # koso s lijeva na desno
        seq = 0
        r = row
        c = col
        while (c-1) >= 0 and (r-1) >= 0 and self.field[r-1][c-1] == player:
            c -= 1
            r -= 1
        while c < self.cols and r < self.rows and self.field[r][c] == player:
            c += 1
            r += 1
            seq += 1
        if seq > 3:
            return True, player

        # koso s desna na lijevo
        seq = 0
        r = row
        c = col
        while (c-1) >= 0 and (r+1) < self.rows and self.field[r+1][c-1] == player:
            c -= 1
            r += 1
        while c < self.cols and r >= 0 and self.field[r][c] == player:
            c += 1
            r -= 1
            seq += 1
        if seq > 3:
            return True, player
        return False, -1

    def __str__(self):
        s = ""
        for i in range(self.rows-1, -1, -1):
            s += "\n"
            for j in range(self.cols):
                s += str(self.field[i][j])
                s += " "
        return s




