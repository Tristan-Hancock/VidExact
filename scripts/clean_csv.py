import pandas as pd
import json
import re

# Load the final multimodal results
df = pd.read_csv('../scripts/final_results.csv')

# --- Face Names: strip prefix and handle missing ---
def normalize_face(name):
    if pd.isna(name) or name.lower() == 'nofacedetected':
        return ''
    return name.replace('pins_', '').strip()

df['face_names'] = df['face_detected'].apply(normalize_face)

# --- Object Detection: parse JSON, dedupe, join ---
def parse_objects(json_str):
    try:
        items = json.loads(json_str)
        classes = {obj['class'].lower().replace(' ', '_') for obj in items}
        return ';'.join(sorted(classes))
    except Exception:
        return ''

df['objects'] = df['object_detection'].apply(parse_objects)

# --- Action Terms: split CamelCase into lowercase words ---
def split_camel(label):
    if pd.isna(label) or label.lower() == 'unknown':
        return ''
    tokens = re.findall(r'[A-Z][a-z]*', label)
    return ' '.join(token.lower() for token in tokens)

df['action_terms'] = df['action'].apply(split_camel)

# --- Caption: lowercase, remove punctuation, fill missing ---
def normalize_caption(text):
    if pd.isna(text):
        return ''
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", '', text)
    return text.strip()

df['caption_norm'] = df['caption'].apply(normalize_caption)

# --- Select and reorder columns for search index ---
search_df = df[['clip_name', 'timestamp_start', 'timestamp_end', 'face_names', 'objects', 'action_terms', 'caption_norm']]

# Save cleaned CSV for indexing
search_df.to_csv('../scripts/search_index.csv', index=False)
print(f"Cleaned search index saved to '../scripts/search_index.csv' ({len(search_df)} rows)")
