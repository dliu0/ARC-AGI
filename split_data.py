import os
import json
import random
import glob
from pathlib import Path

random.seed(42)

def load_tasks(dir_path):
    tasks = []
    for filepath in glob.glob(os.path.join(dir_path, '*.json')):
        with open(filepath, 'r') as f:
            data = json.load(f)
            task_id = os.path.basename(filepath).split('.')[0]
            data['task_id'] = task_id
            tasks.append(data)
    return tasks

training_tasks = load_tasks('data/training')
eval_tasks = load_tasks('data/evaluation')
all_tasks = training_tasks + eval_tasks
random.shuffle(all_tasks)

n = len(all_tasks)
train_split = int(0.8 * n)
val_split = int(0.9 * n)

train_data = all_tasks[:train_split]
val_data = all_tasks[train_split:val_split]
test_data = all_tasks[val_split:]

os.makedirs('data_splits', exist_ok=True)
os.makedirs('tests/testdata', exist_ok=True)

with open('data_splits/trainset.json', 'w') as f:
    json.dump(train_data, f, indent=2)

with open('data_splits/valset.json', 'w') as f:
    json.dump(val_data, f, indent=2)

# Path-guarded test file
with open('tests/testdata/testset.json', 'w') as f:
    json.dump(test_data, f, indent=2)

print(f"Split completed: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
