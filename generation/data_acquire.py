import pandas as pd
import json
import argparse
import re

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='Input directory path')
    parser.add_argument('--output_path', type=str, required=True, help='Output file path')
    args = parser.parse_args()

    new_data = []
    for i in range(3):
        try:
            data_temp = pd.read_parquet(f"{args.input_path}/00{i}_00000.parquet")
            
            # Filter by text length
            data_temp['length'] = data_temp['text'].apply(lambda x: len(x.split()))
            data_temp = data_temp[data_temp['length'] >= 1500]
            data_temp = data_temp[data_temp['length'] <= 32000]
            
            print(f"File {i}: {len(data_temp)} documents after filtering")
            
            # Sampling
            sample_size = min(int(10000*2), len(data_temp))
            data_temp = data_temp.sample(n=sample_size)

            for _, item in data_temp.iterrows():
                new_item = {
                    'document': item['text'].strip(),
                }
                new_data.append(new_item)

        except Exception as e:
            print(f"Error processing file {i}: {str(e)}")

    print(f"Total documents: {len(new_data)}")

    with open(args.output_path, 'w', encoding='utf-8') as f:
        for item in new_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Successfully written to {args.output_path}")


if __name__ == "__main__":
    main()