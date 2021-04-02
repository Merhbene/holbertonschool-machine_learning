#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

x = np.arange(3)
apples = plt.bar(x, fruit[0], color='red', width=0.5)
bananas = plt.bar(x, fruit[1], color='yellow', bottom=fruit[0], width=0.5)
oranges = plt.bar(x, fruit[2], color='#ff8000',
                  bottom=np.sum(fruit[:2], axis=0), width=0.5)
peaches = plt.bar(x, fruit[3], color='#ffe5b4',
                  bottom=np.sum(fruit[:3], axis=0), width=0.5)
plt.xticks(x, ('Farrah', 'Fred', 'Felicia'))
plt.yticks(np.arange(0, 81, 10))
plt.legend((apples, bananas, oranges, peaches),
           ('apples', 'bananas', 'oranges', 'peaches'))
plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')
plt.show()
