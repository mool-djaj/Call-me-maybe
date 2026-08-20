from pwn import xor

def xtime(a):
    a = int(a, 16)
    b = a << 1
    if b & 0x100:
        b ^= 0x11b
    return hex(b & 0xff)[2:].zfill(2)

def gmul(a, b):
    a = int(a, 16)
    b = int(b, 16)
    p = 0
    for _ in range(8):
        if b & 1:
            p ^= a
        a <<= 1
        if a & 0x100:
            a ^= 0x11b
        b >>= 1
    return hex(p)[2:].zfill(2)

def inv_shift_rows(state):
    # state is 4x4 column-major (list of rows, each row list of cols)
    # Actually original code: state[row][col].
    # Row 0 unchanged.
    row1 = [state[1][0], state[1][1], state[1][2], state[1][3]]
    row2 = [state[2][0], state[2][1], state[2][2], state[2][3]]
    row3 = [state[3][0], state[3][1], state[3][2], state[3][3]]
    # inverse shift: right rotation
    row1 = [row1[-1]] + row1[:-1]   # right 1
    row2 = row2[-2:] + row2[:-2]     # right 2
    row3 = row3[-3:] + row3[:-3]     # right 3
    state[1] = row1
    state[2] = row2
    state[3] = row3
    return state

def inv_mix_columns(state):
    const = [[0x0e, 0x0b, 0x0d, 0x09],
             [0x09, 0x0e, 0x0b, 0x0d],
             [0x0d, 0x09, 0x0e, 0x0b],
             [0x0b, 0x0d, 0x09, 0x0e]]
    new = [[0]*4 for _ in range(4)]
    for col in range(4):
        # We operate on column j
        for row in range(4):
            val = 0
            for k in range(4):
                val ^= int(gmul(state[k][col], hex(const[row][k])[2:].zfill(2)), 16)
            new[row][col] = hex(val)[2:].zfill(2)
    return new

def to_matrix(h):
    m = [[0]*4 for _ in range(4)]
    for i in range(0, 32, 2):
        idx = i//2
        r = idx % 4
        c = idx // 4
        m[r][c] = h[i:i+2]
    return m

def from_matrix(m):
    s = ""
    for c in range(4):
        for r in range(4):
            s += m[r][c]
    return s

def linear_inv(x):
    state = to_matrix(x)
    state = inv_shift_rows(state)   # SR^-1
    for _ in range(9):
        state = inv_mix_columns(state)
        state = inv_shift_rows(state)
    return from_matrix(state)

def hex_xor(a, b):
    return xor(bytes.fromhex(a), bytes.fromhex(b)).hex()

pt0 = "696e636f6d70726568656e7369626c65"
ct0 = "94ae785acdb0d7c919f4893697659c8c"
ct1 = "58f86ce660590bb05495c0dcd2d4d438"

d = hex_xor(ct0, ct1)
delta = linear_inv(d)
pt1_hex = hex_xor(pt0, delta)

print("Recovered flag hex:", pt1_hex)
try:
    print("Recovered flag:", bytes.fromhex(pt1_hex).decode('utf-8'))
except UnicodeDecodeError:
    print("Flag contains non-ASCII bytes; raw bytes:", bytes.fromhex(pt1_hex))