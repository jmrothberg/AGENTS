def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

primes = []
num = 2
while len(primes) < 5:
    if is_prime(num):
        primes.append(num)
    num += 1

result = sum(primes)
print(f"The first 5 prime numbers are: {primes}")
print(f"The sum is: {result}")