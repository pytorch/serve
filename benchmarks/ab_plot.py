import matplotlib.pyplot as plt


grid = plt.GridSpec(3, 2, wspace=0.2, hspace=0.2)
a4_dims = (11, 8)
plt.figure(figsize=a4_dims)
plt.style.use('seaborn-white')

fig1=plt.subplot(grid[0, 0])
plt.subplot(grid[0, 1])
plt.subplot(grid[1, 0])
plt.subplot(grid[1, 1])
plt.subplot(grid[2, 0:])
a=[1,2,3,4,5]
plt.plot(a)
plt.savefig('aa.png',bbox_inches='tight')


plt.show()
