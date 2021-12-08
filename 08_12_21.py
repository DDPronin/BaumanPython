inp = input().split()
N = int(inp[0])
M = int(inp[1])

accounts = []

for i in range(N):
    accounts.append(int(input()))

start = sum(accounts) // M

solutions = [0]
for i in range(start, 0, -1):
    p = 0
    for j in accounts:
        p += j // i
    if (p % M) == 0:
        solutions.append(i)

print(max(solutions))
