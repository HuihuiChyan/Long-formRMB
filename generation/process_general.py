import pandas as pd
import json
from collections import defaultdict

# Read the Excel file
df = pd.read_excel('data/general_rag_preference_new.xlsx')

# Create a dictionary to store data for each subset
subset_data = defaultdict(lambda: {
    'prompt': [],
    'chosen': [],
    'rejected': [],
    'subset': [],
    'id': []
})

for _, row in df.iterrows():
    # Determine which answer is chosen based on the 'better' column
    if row['better'] == 1:
        chosen_answer = row['answer_1']
        chosen_model = row['answer_1_model']
        rejected_answer = row['answer_2']
        rejected_model = row['answer_2_model']
    else:  # better == 2
        chosen_answer = row['answer_2']
        chosen_model = row['answer_2_model']
        rejected_answer = row['answer_1']
        rejected_model = row['answer_1_model']

    # Get the current subset's data dictionary
    current_subset = subset_data[row['subset']]
    
    # Append data to the appropriate subset
    current_subset['prompt'].append(row['question'])
    current_subset['chosen'].append([
        {"role": "user", "content": row['question']},
        {"role": "assistant", "content": chosen_answer},
    ])
    current_subset['rejected'].append([
        {"role": "user", "content": row['question']},
        {"role": "assistant", "content": rejected_answer},
    ])
    current_subset['subset'].append(row['subset'])
    current_subset['id'].append(row['id'])

# Write separate JSON files for each subset
for subset_name, data in subset_data.items():
    output_file = f'data/preference_data_{subset_name}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Processed {len(data['prompt'])} entries for subset '{subset_name}' and saved to {output_file}")
    print(f"Data can be loaded using: datasets.load_dataset('json', data_files='{output_file}')")

# Print total number of entries processed
total_entries = sum(len(data['prompt']) for data in subset_data.values())
