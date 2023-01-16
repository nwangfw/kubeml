import pickle
import os



with open('opt.pkl', 'wb') as f:
    pickle.dump("yrdy", f)
print('saved state')

with open('opt.pkl', 'rb') as f:
    state = pickle.load(f)
    print(state)