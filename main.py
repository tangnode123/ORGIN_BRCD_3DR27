from queue import Queue
import sys
import numpy as np
from tqdm import tqdm
import timeit

class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __hash__(self):
        return self.x * 100 + self.y * 10 + self.z

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

def RCDM_3D27(arr, visited):
    row = [0, 0, -1, 1, 0, 0]
    col = [1, -1, 0, 0, 0, 0]
    ver = [0, 0, 0, 0, 1, -1]

    que = Queue()
    found_one = False

    for p in list(visited):
        que.put(p)
        visited.remove(p)
        found_one = True
        break

    while not que.empty():
        p = que.get()
        for n in range(6):
            x = p.x + row[n]
            y = p.y + col[n]
            z = p.z + ver[n]
            np = Point(x, y, z)

            if 1 <= x <= 3 and 1 <= y <= 3 and 1 <= z <= 3 and np in visited:
                visited.remove(np)
                que.put(np)

    return len(visited) == 0 and found_one

def rec(p):
    global b, t, l, r, d, u
    B, T, L, R, D, U = sys.maxsize, -sys.maxsize, sys.maxsize, -sys.maxsize, sys.maxsize, -sys.maxsize
    count = 0

    for i in range(0, 3):
        for j in range(0, 3):
            for k in range(0, 3):
                if p[i][j][k] == 1:
                    count += 1
                    B = min(B, i)
                    T = max(T, i)
                    L = min(L, j)
                    R = max(R, j)
                    D = min(D, k)
                    U = max(U, k)

    return b == B and t == T and l == L and r == R and d == D and u == U
def input_3d_matrix(layers, rows, cols):
    matrix = []
    for i in range(layers):
        print(f"正在输入第 {i+1} 层:")
        layer = []
        for j in range(rows):
            while True:
                row_input = input(f"请输入第 {j+1} 行(用空格分隔 {cols} 个数字，只能是0或1): ")
                try:
                    row = list(map(int, row_input.split()))
                    if len(row) != cols:
                        print(f"错误: 需要输入 {cols} 个数字")
                        continue
                    # 检查是否所有数字都是0或1
                    if any(num not in {0, 1} for num in row):
                        print("错误: 只能输入0或1")
                        continue
                    layer.append(row)
                    break
                except ValueError:
                    print("错误: 请输入有效的整数")
        matrix.append(layer)
    return matrix
def setup_globals():
    global b, t, l, r, d, u, chang, kuan, gao, zero_e_num, recnumber, num
    global nodes_generated_pruned, nodes_generated_original
    
    # Reset global variables
    b, t, l, r, d, u = sys.maxsize, -sys.maxsize, sys.maxsize, -sys.maxsize, sys.maxsize, -sys.maxsize
    recnumber = 0
    num = 0
    nodes_generated_pruned = 0
    nodes_generated_original = 0
    # Enter the rectangular cardinal direction relation matrix here. (same as your example)
    matrix = [
        [
            [ 1, 1, 0],
            [ 1, 1, 0],
            [ 0, 0, 0],
        ],
        [
            [ 1, 1, 0],
            [ 1, 1, 0],
            [ 0, 0, 0],
        ],
        [
            [ 1, 1, 0],
            [ 1, 1, 0],
            [ 0, 0, 0],
        ]
    ]
    
    # Calculate bounds
    for i_1 in range(0, 3):
        for j_1 in range(0, 3):
            for k_1 in range(0, 3):
                if matrix[i_1][j_1][k_1] == 1:
                    b = min(b, i_1)
                    t = max(t, i_1)
                    l = min(l, j_1)
                    r = max(r, j_1)
                    d = min(d, k_1)
                    u = max(u, k_1)

    chang = t - b + 1
    kuan = r - l + 1
    gao = u - d + 1
    zero_e_num = 0
    
    return matrix

def ORGIN_BRCD_3DR27(p, a, zero_e_num):
    global num, recnumber, nodes_generated_pruned
    nodes_generated_pruned += 1  # 计数
    if chang * kuan * gao == 8:
        if zero_e_num >= 5:
            return
    if chang * kuan * gao == 12:
        if zero_e_num >= 8:
            return
    if chang * kuan * gao == 18:
        if zero_e_num >= 13:
            return
    if chang * kuan * gao == 27:
        if zero_e_num >= 21:
            return
    if chang * kuan * gao == 9:
        if zero_e_num >= 5:
            return

    if a > chang * kuan * gao:
        cun = [[[0] * 5 for _ in range(5)] for _ in range(5)]
        visited = set()

        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    cun[i][j][k] = p[i][j][k]
                    if p[i][j][k] == 1:
                        visited.add(Point(i, j, k))
        if RCDM_3D27(cun, visited):
            num += 1

            if rec(p):
                recnumber += 1
        return

    i = (a - 1) // (kuan * gao) + b
    j = ((a - 1) % (kuan * gao)) // gao + l
    k = ((a - 1) % (kuan * gao)) % gao + d

    p[i][j][k] = 1
    ORGIN_BRCD_3DR27(p, a + 1, zero_e_num)
    p[i][j][k] = 0
    ORGIN_BRCD_3DR27(p, a + 1, zero_e_num + 1)

def ORGIN_BRCD_3DR27_NP(p, a):
    global num, recnumber, nodes_generated_original
    nodes_generated_original += 1  
    if a > chang * kuan * gao:
        cun = [[[0] * 5 for _ in range(5)] for _ in range(5)]
        visited = set()

        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    cun[i][j][k] = p[i][j][k]
                    if p[i][j][k] == 1:
                        visited.add(Point(i, j, k))

        if RCDM_3D27(cun, visited):
            num += 1
            if rec(p):
                recnumber += 1
        return

    i = (a - 1) // (kuan * gao) + b
    j = ((a - 1) % (kuan * gao)) // gao + l
    k = ((a - 1) % (kuan * gao)) % gao + d

    p[i][j][k] = 1
    ORGIN_BRCD_3DR27_NP(p, a + 1)
    p[i][j][k] = 0
    ORGIN_BRCD_3DR27_NP(p, a + 1)

if __name__ == "__main__":
    # Number of runs for each algorithm
    num_runs = 10
    
    # Test the original algorithm (without pruning)
    print("Testing original algorithm (without pruning)...")
    original_times = []
    original_nodes = []
    
    for _ in tqdm(range(num_runs), desc="Original Algorithm"):
        matrix = setup_globals()
        def run_original():
            global recnumber
            ORGIN_BRCD_3DR27_NP(matrix, 1)
            return recnumber
        time_taken = timeit.timeit(run_original, number=1)
        original_times.append(time_taken)
        original_nodes.append(nodes_generated_original)
    
    # Test the pruned algorithm
    print("\nTesting pruned algorithm...")
    pruned_times = []
    pruned_nodes = []
    
    for _ in tqdm(range(num_runs), desc="Pruned Algorithm"):
        matrix = setup_globals()
        def run_pruned():
            global recnumber, zero_e_num
            ORGIN_BRCD_3DR27(matrix, 1, zero_e_num)
            return recnumber
        time_taken = timeit.timeit(run_pruned, number=1)
        pruned_times.append(time_taken)
        pruned_nodes.append(nodes_generated_pruned)
    
    # Calculate averages
    avg_original_time = np.mean(original_times)
    avg_pruned_time = np.mean(pruned_times)
    avg_original_nodes = np.mean(original_nodes)
    avg_pruned_nodes = np.mean(pruned_nodes)
    
    # Print results
    print("\nResults after", num_runs, "runs:")
    print(f"Original method average time: {avg_original_time:.6f} seconds")
    print(f"Pruned method average time: {avg_pruned_time:.6f} seconds")
    print(f"Original method average nodes generated: {avg_original_nodes:.1f}")
    print(f"Pruned method average nodes generated: {avg_pruned_nodes:.1f}")
    
    # Calculate speedup
    if avg_pruned_time > 0:
        speedup = avg_original_time / avg_pruned_time
        print(f"\nSpeedup: {speedup:.2f}x")
    else:
        print("\nSpeedup: (pruned time was 0, cannot calculate)")