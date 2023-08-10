import os
import fasttext
import tiktoken


dim = 8

# Initialize the encoding
encoding = tiktoken.encoding_for_model("gpt-4")

from concurrent.futures import ProcessPoolExecutor

def parallel_process_file(file_path):
    """
    Function to read and process a file. This function will be run in parallel.
    """
    with open(file_path, 'r', encoding='utf-8') as in_f:
        text = in_f.read()
        encoded_text = process_text(text)
        return encoded_text

def parallel_digest_data(directory, output_file, max_workers=None):
    """
    Process text files in parallel.
    
    Args:
    - directory (str): Path to the directory containing text files.
    - output_file (str): Path to the output file where processed data will be written.
    - max_workers (int, optional): Maximum number of worker processes. If None, it will use as many as the machine has CPUs.
    """
    # Clean (or clear the contents of) the output_file
    with open(output_file, 'w', encoding='utf-8') as out_f:
        pass
    # Get list of text files in the directory
    txt_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')]
    
    all_encoded_texts = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for idx, encoded_text in enumerate(executor.map(parallel_process_file, txt_files), 1):
            all_encoded_texts.append(encoded_text)
            print(f"Processed file {idx} of {len(txt_files)}")
    
    # Write the processed text to the output file
    with open(output_file, 'w', encoding='utf-8') as out_f:
        out_f.write(" ".join(all_encoded_texts))

def process_text(text):
    encoded_tokens = encoding.encode(text)
    return ' '.join(map(str, encoded_tokens))

def digest_data(directory, output_file):
    all_texts = []
    for idx, filename in enumerate(os.listdir(directory)):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as in_f:
                all_texts.append(in_f.read())
            print(f"Processed file {idx + 1} of {len(os.listdir(directory))}: {filename}")
    processed_text = process_text(" ".join(all_texts))
    with open(output_file, 'w') as out_f:
        out_f.write(processed_text)

def train_fasttext_model(base_path, model_path, min_count=1):
    model = fasttext.train_unsupervised(
        input=base_path,
        model='cbow',
        dim=dim,
        ws=5,
        epoch=70,
        neg=1,
        bucket=200000,
        minCount=min_count  # Adjust this as needed
    )

    words = model.words

    # Write the vectors to the vec.vec file
    with open(model_path, 'w') as out_file:
        out_file.write(f"{len(words)} {dim}\n")
        decoded_words = []
        for word in words:
            vector = model[word]
            try:
                decoded_word = encoding.decode([int(word)])
                decoded_words.append(decoded_word)
            except:
                pass
            vector_str = ' '.join(map(str, vector))
            out_file.write(f"{word} {vector_str}\n")
        print(f'decoded words: {decoded_words}')
    print(f"Vectors saved to {model_path}.")

    print("Training completed.")

if __name__ == "__main__":
    DATA_DIR = './data'
    BASE_PATH = './models/base'
    MODEL_PATH = './models/vec.vec'
    
    try:
        parallel_digest_data(DATA_DIR, BASE_PATH, max_workers=10)
        print("Starting training with FastText...")
        train_fasttext_model(BASE_PATH, MODEL_PATH)
    except Exception as e:
        print(f"An error occurred: {e}")
