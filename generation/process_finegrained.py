import pandas as pd
import json
import re

for aspect in ["Safety"]:

    # Read the Excel file
    df = pd.read_excel(f'data-finegrained/general_rag_preference_0603_{aspect}.xlsx')

    if aspect != "Safety":
        filtered_df = df[df[f'comparison_result'] == 'second']
    else:
        filtered_df = df

    output_data = []
    for _, row in filtered_df.iterrows():

        # Pattern to capture potential newlines after <MODIFIED> and before </MODIFIED>
        pattern = re.compile(re.escape("<MODIFIED>") + r"(\n?)(.*?)(\n?)" + re.escape("</MODIFIED>"), re.DOTALL)
        matches = pattern.finditer(row[f'tagged_response'])
        match = next(matches, None)
        try:
            assert match is not None
        except:
            print(f"Expected one <MODIFIED> tag in {row[f'tagged_response']}")
            continue
        assert next(matches, None) is None, f"Found more than one <MODIFIED> tag in {row[f'tagged_response']}"
        
        # Extract newlines from original text
        start_newline = match.group(1)  # newline after <MODIFIED>
        end_newline = match.group(3)    # newline before </MODIFIED>

        # Add newlines to modified chunk if they existed in original
        pos_replacement = (start_newline if start_newline else '') + \
                            row['positive_chunk'] + \
                            (end_newline if end_newline else '')
        neg_replacement = (start_newline if start_newline else '') + \
                            row['negative_chunk'] + \
                            (end_newline if end_newline else '')
        original_text = row[f'tagged_response'][:match.start()] + \
                        pos_replacement + \
                        row[f'tagged_response'][match.end():]   
        modified_text = row[f'tagged_response'][:match.start()] + \
                        neg_replacement + \
                        row[f'tagged_response'][match.end():]

        output_line = { 'prompt': row['question'],
                        'chosen': [original_text],
                        'rejected': [modified_text],
                        'subset': row['subset'],
                        'id': row['id'],
                        'num_correct': 1,
                        'total_completions': 1,
                        'tag_position': row[f'tagged_position']}
        output_data.append(output_line)            

    # Write to JSON file
    output_file = f'data-finegrained/preference_data_{aspect}.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in output_data:
            json_line = json.dumps(line, ensure_ascii=False)
            f.write(json_line + '\n')

    print(f"Processed {len(output_data)} entries and saved to {output_file}")
    print("Data can be loaded using: datasets.load_dataset('json', data_files='" + output_file + "')")
