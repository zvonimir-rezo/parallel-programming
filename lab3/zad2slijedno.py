import time

start = time.time()

n = 10000000

h = 1.0 / n
suma = 0
for i in range(1, n+1):
    x = h * (i - 0.5)
    suma += 4.0 / (1.0 + x*x)
    

print("Pi:", str(h * suma))
print(f"Time: {time.time() - start}")