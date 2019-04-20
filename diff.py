import sys

try:
    file1 = sys.argv[1]
    file2 = sys.argv[2]
except IndexError:
    print('ERROR: need 2 files as arguments')
    sys.exit(1)

cnt = 0
df = 0
for ln1, ln2 in zip(open(file1), open(file2)):
    cnt += 1
    lst1, lst2 = ln1.split(), ln2.split()
    print(lst1[0], end=' ')
    if lst1[1] == lst2[1]:
        print('eq:', lst1[1])
    else:
        print('df:', lst1[1], lst2[1])
        df += 1
        

print('\nall:      ', cnt)
print('same:     ', cnt - df)
print('different:', df)