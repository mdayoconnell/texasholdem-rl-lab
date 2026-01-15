from collections import Counter
import numpy as np

suits = np.asarray([1,1,2,3,4,1,1])

values = np.asarray([1,2,3,5,14,14,8])

values_count = Counter(values)
fours = sorted([val for val, c in values_count.items() if c == 4])
threes = sorted([val for val, c in values_count.items() if c == 3])
pairs = sorted([val for val, c in values_count.items() if c == 2])

ace_idx = [i for i in range(len(values)) if values[i] == 14]
for aces in range(len(ace_idx)):
    values_ext = np.concatenate(values,np.asarray([1]))
    suits_ext = np.concatenate(suits,(np.asarray[(suits[ace_idx[aces]])]))

print(ace_idx)

print(values_ext)
print(suits_ext)
