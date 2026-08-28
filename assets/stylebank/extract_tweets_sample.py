import csv
import json
import re
import sys
import argparse

# Set csv field size limit to handle large fields
csv.field_size_limit(min(sys.maxsize, 131072 * 10))

# Target celebrities
TARGET_AUTHORS = {
    'jimmyfallon',
    'ArianaGrande',
    'katyperry',
    'selenagomez',
    'rihanna',
    'BarackObama',
    'britneyspears',
    'shakira',
    'Cristiano',
    'jtimberlake',
    'ladygaga',
    'ddlovato',
    'taylorswift13',
    'justinbieber'
}

# CLI argument parser
parser = argparse.ArgumentParser(description='Extract and filter tweets from CSV file')
parser.add_argument('-i', '--input', default='data/tweets/tweets.csv', help='Input CSV file (default: data/tweets/tweets.csv)')
parser.add_argument('-o', '--output', default='data/tweets/tweets_output.json', help='Output JSON file (default: data/tweets/tweets_output.json)')
parser.add_argument('-m', '--max-per-author', type=int, default=64, help='Max tweets per celebrity (default: 64)')
parser.add_argument('-w', '--min-words', type=int, default=10, help='Minimum words per tweet (default: 10)')

args = parser.parse_args()
input_file = args.input
output_file = args.output
max_tweets_per_author = args.max_per_author
min_word_count = args.min_words

tweets_by_author = {author: [] for author in TARGET_AUTHORS}

# Read the CSV file
with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        author = row['author'].strip()
        content = row['content'].strip()
        
        # Check if author is in target list
        if author not in TARGET_AUTHORS:
            continue
        
        # Remove all http links
        content = re.sub(r'\s*https?://\S+\s*', ' ', content)
        content = content.strip()
        
        # Count words (split by whitespace)
        word_count = len(content.split())
        
        # Keep only tweets with at least min_word_count words
        if word_count < min_word_count:
            continue
        
        # Keep max per celebrity
        if len(tweets_by_author[author]) < max_tweets_per_author:
            tweets_by_author[author].append(content)

# Convert to final output format - author: [tweets list]
output = {}
for author in TARGET_AUTHORS:
    if tweets_by_author[author]:
        output[author] = tweets_by_author[author]

# Sort output by author name
output = dict(sorted(output.items()))

# Write to JSON file
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

# Print summary
print("Extraction complete!")
print(f"Output saved to: {output_file}")
print(f"Total celebrities processed: {len([a for a in output if output[a]])}")
for author in sorted(output.keys()):
    print(f"  {author}: {len(output[author])} tweets")
