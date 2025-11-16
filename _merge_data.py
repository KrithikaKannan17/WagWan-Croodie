
import numpy as np
import sys

files = sys.argv[1:]
output = 'final_training_data_merged.npz'

all_boards = []
all_policies = []
all_values = []

for f in files:
    try:
        data = np.load(f)
        all_boards.append(data['boards'])
        all_policies.append(data['policy_targets'])
        all_values.append(data['value_targets'])
        print(f'✓ Loaded {f}: {len(data["boards"])} positions')
    except Exception as e:
        print(f'✗ Failed to load {f}: {e}')

boards = np.concatenate(all_boards)
policies = np.concatenate(all_policies)
values = np.concatenate(all_values)

# Shuffle
indices = np.random.permutation(len(boards))
boards = boards[indices]
policies = policies[indices]
values = values[indices]

np.savez_compressed(output,
    boards=boards,
    policy_targets=policies,
    value_targets=values
)

print(f'\n✅ Merged {len(boards)} positions → {output}')
