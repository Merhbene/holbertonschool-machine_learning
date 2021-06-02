#!/usr/bin/env python3


quit = ['exit', 'quit', 'goodbye', 'bye']

while True:
    question = input('Q: ').lower()
    if question in quit:
        print('A: Goodbye')
        break
    print('A:')
