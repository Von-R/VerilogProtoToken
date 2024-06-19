"""
Stage 1: Cleaning and Synthesizable Verilog Code Preparation

This script processes a dataset of Verilog files to remove unnecessary and non-synthesizable code. 
The initial dataset contains various issues such as comments, empty lines, invalid code segments 
(simulation, COQ, pragma's, directives). This script performs the following steps:

1. Loads the dataset from the Hugging Face Hub.
2. Removes comments and invalid code.
3. Identifies and removes non-synthesizable Verilog code.
4. Ensures the resulting Verilog code is clean and synthesizable.
5. Saves and pushes the cleaned dataset back to the Hugging Face Hub.

The script uses regex patterns to identify and remove unwanted code segments, 
and it performs Exploratory Data Analysis (EDA) to ensure the cleaning process is effective.

"""
import math
import os
import re
import sys
from datasets import load_dataset
from huggingface_hub import HfApi
from verilog_lists import (
    multiline_begin_words, COQ_keywords, multiline_end_words,
    non_synthesizable_verilog_keywords, single_line_removal, directive_list, 
    non_synth_path_keywords, verilog_keywords
)

# Flags and counters for HPC setup and data processing
HPC = True
match_flag = False
initial_counter = 0
unbalanced_counter = 0
missed_COQ = 0

# Print environment variables for HPC setup
if HPC:
    print("HF_DATASETS_CACHE:", os.getenv("HF_DATASETS_CACHE"))
    print("HF_HOME:", os.getenv("HF_HOME"))

# Ensure stdout and stderr are properly set
if not sys.__stdout__.closed:
    sys.stdout = sys.__stdout__
else:
    print("sys.__stdout__ is closed.", file=sys.__stderr__)

sys.stderr = sys.__stderr__

COQ_keyword_counts = {keyword: 0 for keyword in COQ_keywords}

# Initialize the HfApi
hf_api = HfApi()

# Read the Hugging Face API read token from file
read_token_file = "read_token.txt"
with open(read_token_file, "r") as file:
    hf_read_token = file.read().strip()

# Load the dataset to be cleaned from the Hugging Face Hub
print("Loading dataset")
dataset = load_dataset("Von-R/deduplicated", token=hf_read_token)
print("Dataset loaded")

original_dataset_rows = dataset['train'].num_rows
original_dataset_size = sum(dataset['train']['size'])

# Compile each keyword into a regex pattern for filtering
non_synthesizable_verilog_keywords = [re.compile(keyword) for keyword in non_synthesizable_verilog_keywords]
multiline_begin_patterns = [re.compile(re.escape(word)) for word in multiline_begin_words]
multiline_end_patterns = [re.compile(re.escape(word)) for word in multiline_end_words]
COQ_keyword_patterns = [re.compile(re.escape(keyword), re.IGNORECASE) for keyword in COQ_keywords]
COQ_keyword_patterns.append(re.compile(r"\((\w+)\s*:\s*([^\)]+)\)"))

# Patterns to identify module declarations
incomplete_assignment_pattern = re.compile(r'\b\w+\s*=\s*\n')
module_pattern = re.compile(r'\bmodule\s+([_a-zA-Z\$\\][_a-zA-Z0-9\$\\]*)(\s*#\s*\((?:[^;]*?\([^)]*\))*[^;]*?\))?(\s*\((?:[^;]*?\([^)]*\))*[^;]*?\))?\s*.*?;', re.DOTALL)
COQ_module_pattern = re.compile(r'\bmodule\s+([_a-zA-Z\$\\][_a-zA-Z0-9\$\\]*)\s*\.', re.DOTALL)

def test_ds(dataset, num_samples=100):
    """
    Function to create a test subset from the dataset for testing.
    Shuffles and selects a subset of the dataset.
    """
    test_subset = dataset["train"].shuffle(seed=42).select(range(num_samples))
    test_subset.to_csv('test_subset_stage1.csv', index=False)
    return test_subset

# Convert dataset to 'train' split for processing
dataset = dataset['train']
original_dataset_rows = dataset.num_rows
original_dataset_size = sum(dataset['size'])

#####################################################################################
# REMOVE COMMENTS
#####################################################################################
def remove_verilog_comments(code):
    """
    Function to remove different types of comments from Verilog code.
    """
    if code is None or code == '':
        return ''

    # Remove single-line comments
    code = re.sub(r'//.*\n', '', code)
    code = re.sub(r'//=*\s*\n?', '', code)

    # Remove multi-line comments
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

    # Remove comments enclosed in (** **)
    code = re.sub(r'\(\*.*?\*\)', '', code, flags=re.DOTALL)

    return code

#####################################################################################
# IDENTIFY NON-SYNTH CODE
#####################################################################################
def isNonSynthesizable(code):
    """
    Function to identify non-synthesizable Verilog code.
    Masks module names to prevent false positives and checks for non-synthesizable patterns.
    """
    matches = []

    # Mask modules to prevent false positives
    if masked_modules := module_pattern.finditer(code):
        masked_modules = list(masked_modules)
        if len(masked_modules) > 1:
            for match in masked_modules:
                code = re.sub(re.escape(match.group(0)), '', code)

    # Check for non-synthesizable patterns
    if any(pattern.search(code) for pattern in non_synthesizable_verilog_keywords):
        found_patterns = [pattern for pattern in non_synthesizable_verilog_keywords if pattern.search(code)]
        for pattern in found_patterns:
            if found_matches := pattern.finditer(code, re.DOTALL):
                for match in found_matches:
                    if match_flag:
                        if "parameter" in match.group(0):
                            with open("parameter.txt", "a", encoding='utf-8') as f:
                                f.write(match.group(0) + '\n')
                    if match and match not in verilog_keywords:
                        matches.append(match.group(0))

        return matches, True
    return None, False

#####################################################################################
# REMOVE INVALID CODE
#####################################################################################
def removeInvalidCode(code):
    """
    Function to remove invalid and non-synthesizable Verilog code.
    Uses regex patterns to identify and remove various invalid code segments.
    """
    global initial_counter, unbalanced_counter, missed_COQ
    global COQ_keyword_counts

    COQ_flag = False
    initial_pattern = re.compile(r'^\s*initial(\s+begin)?\s*$')
    begin_pattern = re.compile(r'\bbegin\b\s*', re.IGNORECASE)
    pragma_pattern = re.compile(r'^\s*`pragma\s+', re.IGNORECASE)
    keywords_pattern = '|'.join(re.escape(keyword) for keyword in verilog_keywords)
    def_pattern = re.compile(r'^`\s*(?:' + keywords_pattern + r')\b', re.IGNORECASE)
    begin_counter = 0
    initial_block_flag = False

    cleaned_code = []
    COQ_keyword_list = []
    in_multiline_block = False
    in_COQ_block = False

    # Normalize line endings
    code = code.replace('\r\n', '\n').replace('\r', '\n')
    for word in directive_list:
        code = re.sub(r'^\s*' + re.escape(word) + r'.*\n', '', code, flags=re.MULTILINE)

    # Handle empty content; pass back
    if code == '':
        return [False, '']

    # Remove comments
    code = remove_verilog_comments(code)

    module_results = []

    # Collect all non-synthesizable code within module definitions; remove it
    module_matches = module_pattern.finditer(code)
    module_matches_list = list(module_matches)

    # Each match is a different module definition
    for match in module_matches_list:
        module_results.append(isNonSynthesizable(match.group(1)))

        # If any non-synthesizable code is found, remove it
        if any(module_result[1] == True for module_result in module_results):
            if match_flag:
                with open("./match_list.txt", "a") as f:
                    for module_result in module_results:
                        if module_result[1] is True:
                            for word in module_result[0]:
                                f.write('Masked: ' + word + '\n')
            else:
                for module_result in module_results:
                    if module_result[1] is True:
                        for word in module_result[0]]:
                            code = re.sub(re.escape(word), '', code)

    # Collect all non-synthesizable code outside of module definitions
    results = isNonSynthesizable(code)

    # If non-synthesizable code is found, remove it
    if results[1] is True:
        if match_flag:
            with open("./match_list.txt", "a", encoding="utf-8") as f:
                for word in results[0]]:
                    if word == '':
                        continue
                    f.write(word + '\n')
                    code = re.sub(re.escape(word), '', code)
        else:
            for word in results[0]]:
                if word == '':
                    continue
                code = re.sub(re.escape(word), '', code)

    # Process each line of the code
    lines = [line.strip() for line in code.split('\n')]
    for line in lines:
        # Skip empty lines
        if line == '':
            continue

        ################### BEGIN/END BLOCKS ############################
        if begin_matches := re.findall(r'^\s*begin\s*$', line):
            begin_counter += len(begin_matches)
            cleaned_code.append(line)
            continue

        if begin_counter > 0:
            if re.search(r'\belse\b', line):
                if len(re.findall(r'\bend\b', line)) == 1 and len(re.findall(r'\begin\b', line)) == 1:
                    pass
                elif begin_counter >= 1 and not re.search(r'\bend\b\s*else\b', line) and not re.search(r'\bend\b\s*$', cleaned_code[-1]):
                    begin_counter -= 1
                    cleaned_code.append('end')
                elif re.search(r'^\s*end\s*else\s*$', line):
                    begin_counter -= 1
                    cleaned_code.append(line)
                    continue
            elif end_matches := re.findall(r'\bend\b', line):
                begin_counter -= len(end_matches)

        if (line.strip().startswith('`') and not re.search(def_pattern, line)) \
               or pragma_pattern.search(line) or line.startswith('#'):
            if match_flag:
                with open("match.txt", "a", encoding='utf-8') as f:
                    f.write(line + '\n')
            continue

        ################ INITIAL BLOCKS ############################
        if initial_block_flag:
            continue
        elif initial_pattern.match(line):
            initial_block_flag = True
            continue
        elif re.search(r'^\s*end\s*$', line) and initial_block_flag:
            initial_block_flag = False
            continue

        ################# MULTILINE BLOCKS ####################
        if any(pattern.search(line) for pattern in multiline_begin_patterns):
            if match_flag:
                with open("match.txt", "a", encoding='utf-8') as f:
                    f.write('MULTILINE BEGIN: ' + line + '\n')
            in_multiline_block = True
            continue
        elif any(pattern.search(line) for pattern in multiline_end_patterns):
            if match_flag:
                with open("match.txt", "a", encoding='utf-8') as f:
                    f.write('MULTILINE END: ' + line + '\n')
            in_multiline_block = False
            continue
        elif in_multiline_block:
            if match_flag:
                with open("match.txt", "a", encoding='utf-8') as f:
                    f.write(line + '\n')
            continue

        ############## COQ CHECK ###############################
        for keyword, pattern in zip(COQ_keywords, COQ_keyword_patterns):
            if pattern.search(line):
                COQ_flag = True
                COQ_keyword_list.append(keyword)
                in_COQ_block = True
                if match_flag:
                    with open("match.txt", "a", encoding='utf-8') as f:
                        f.write('COQBLOCK: ' + line + '\n')
                break
            elif in_COQ_block and not any(pattern.search(line) for pattern in COQ_keyword_patterns):
                in_COQ_block = False

        if in_COQ_block:
            continue

        if any(keyword in line for keyword in COQ_keywords):
            print("Escaped COQ keyword found. Dumping line: ", line)
            exit(-1)
            if match_flag:
                with open("match.txt", "a", encoding='utf-8') as f:
                    f.write('Escaped COQ keyword: ' + line + '\n')
            continue

        ############## SINGLE LINE REMOVAL #####################
        if any(re.search(pattern, line) for pattern in single_line_removal):
            if match_flag:
                with open("match.txt", "a", encoding='utf-8') as f:
                    f.write('Single line removal: ' + line + '\n')
            continue

        if incomplete_assignment_pattern.search(line):
            if match_flag:
                with open("match.txt", "a", encoding='utf-8') as f:
                    f.write('Incomplete assignment: ' + line + '\n')
            continue

        ################## REMOVE WHITESPACE AND EMPTY LINES ############################
        line = line.replace('\t', ' ').strip()
        line = re.sub(r'\s+', ' ', line)

        if line == '' or re.match(r'^\s*$', line):
            continue

        if begin_pattern.search(line) and not (len(re.findall(r'\bend\b', line)) == 1 and len(re.findall(r'\begin\b', line)) == 1):
            begin_counter += len(re.findall(r'\bbegin\b', line))

        cleaned_code.append(line)

    cleaned_code = '\n'.join(cleaned_code)
    cleaned_code = re.sub('`', '', cleaned_code)

    results = [COQ_keyword_list, COQ_flag, cleaned_code]
    if len(results) != 3:
        exit(0)
    if begin_counter > 0:
        unbalanced_counter += 1

    return results

def preprocess_file(code):
    """
    Wrapper function to remove comments and invalid code from a Verilog file.
    """
    [COQ_keyword_list, COQ_flag, processed_code] = removeInvalidCode(code)

    if processed_code is None:
        processed_code = ''

    return [COQ_keyword_list, COQ_flag, processed_code]

def preprocess(example):
    """
    Preprocesses a single example from the dataset.
    Handles COQ keywords, file paths, and removes invalid Verilog code.
    """
    global missed_COQ
    example['remove_flag'] = False
    example['COQ_flag'] = False
    example['remove_reason'] = None
    example['COQ_keywords'] = None

    if example['content'] is None:
        example['content'] = ''
        example['remove_reason'] = "Content is None"
        example['remove_flag'] = True
        return example

    example['content'] = remove_verilog_comments(example['content'])

    full_file_path = str(example['repo_name']) + '/' + str(example['path'])

    if 'coq' in (full_file_path.lower()):
        example['COQ_flag'] = True
        example['remove_flag'] = True
        example['remove_reason'] = "COQ file path"
        return example

    if match := COQ_module_pattern.search(example['content']):
        example['COQ_flag'] = True
        example['remove_flag'] = True
        example['remove_reason'] = "COQ module declaration detected:"
        example['remove_reason'] += f" \"{match[0]}\","
        return example

    lower_path = full_file_path.lower()
    for word in non_synth_path_keywords:
        if word in lower_path:
            example['remove_flag'] = True
            example['remove_reason'] = "Non-synthesizable keyword in path: " + word
            return example

    if not module_pattern.search(example['content']):
        example['remove_reason'] = "No module declarations found"
        example['remove_flag'] = True
        return example

    example['COQ_keywords'], example['COQ_flag'], example['content'] = preprocess_file(example['content'])

    if example['COQ_flag']:
        for word in COQ_keywords:
            if word in example['content'] and word not in example['COQ_keywords']:
                print("ERROR: Word in example['content'] but not among COQ keywords extracted from content: ", word)

    for keyword in COQ_keyword_patterns:
        if keyword.search(example['content']):
            print(f"COQ keywords missed. Word \"{keyword}\" in \"{example['path']}\".")
            lines = example['content'].split('\n')
            for line in lines:
                if keyword.search(line):
                    missed_COQ += 1
                    print(line)

    return example

def main(unprocessed_dataset):
    """
    Main function to process the entire dataset.
    Handles batch processing and applies the preprocessing to each example.
    """
    global missed_COQ

    if os.path.exists("match_list.txt"):
        os.remove("match_list.txt")

    if HPC:
        def batch_process(batch):
            """
            Function to process a batch of examples in parallel.
            """
            processed_examples = []
            for i in range(len(batch['content'])):
                example = {key: batch[key][i] for key in batch}
                processed_example = preprocess(example)
                processed_examples.append(processed_example)

            results = {key: [example[key] for example in processed_examples] for key in processed_examples[0]}
            return results

        preprocessed_dataset = unprocessed_dataset.map(batch_process, batched=True, batch_size=1000, num_proc=4)
    else:
        print("Preprocessing dataset")
        preprocessed_dataset = unprocessed_dataset.map(preprocess)
        print("Preprocessing complete")

    # Save and analyze removed COQ module files
    COQ_modules_files = preprocessed_dataset.filter(lambda example: example['remove_reason'] == "COQ module declaration detected")
    COQ_modules_files.to_csv("COQ_modules_files.csv", index=False)
    number_of_COQ_modules = len(COQ_modules_files)

    # Count the number of preprocessing failures due to COQ keywords
    number_of_failures = len(preprocessed_dataset.filter(lambda example: example['remove_reason'] == "COQ keyword found: Preprocessing failure"))
    print("Failures: ", number_of_failures)

    # Count the number of files identified as COQ based on file paths
    number_of_COQ_files = len(preprocessed_dataset.filter(lambda example: example['remove_reason'] == "COQ file path"))
    
    # Count the number of files with non-synthesizable keywords in the path
    number_of_invalid_paths = len(preprocessed_dataset.filter(lambda example: example['remove_reason'] is not None and "Non-synthesizable keyword in path" in example['remove_reason']))

    # Count the number of files with empty content
    number_of_empty_content = len(preprocessed_dataset.filter(lambda example: example['remove_reason'] == "Content is None"))
    
    # Count the number of files without module declarations
    number_of_no_module_declarations = len(preprocessed_dataset.filter(lambda example: example['remove_reason'] == "No module declarations found"))
    
    # Count the number of files removed without a specified reason
    number_of_no_reason_given = len(preprocessed_dataset.filter(lambda example: example['remove_flag'] is True and example['remove_reason'] is None))
    print("NO REASON GIVEN: ", number_of_no_reason_given)

    # Filter out the removed files and save them to a CSV file
    removed_files = preprocessed_dataset.filter(lambda example: example['remove_flag'])
    removed_files = removed_files.remove_columns(['remove_flag', 'COQ_flag'])
    removed_files.to_csv("removed_files.csv", index=False)
    print("Files removed: ", len(removed_files))

    # Filter out files with COQ content for inspection and save them to a CSV file
    COQ_inspection_files = preprocessed_dataset.filter(lambda example: example['COQ_flag'] and not example['remove_flag'])
    COQ_inspection_files = COQ_inspection_files.remove_columns(['remove_flag', 'COQ_flag'])
    COQ_inspection_files.to_csv("COQ_inspection_files.csv", index=False)
    print("COQ inspection files: ", len(COQ_inspection_files))

    columns_to_remove = ['remove_flag', 'COQ_flag', 'remove_reason']
    print("Removing invalid entries")
    
    # Filter out invalid and COQ-contaminated entries from the dataset
    filtered_dataset = preprocessed_dataset.filter(
        lambda example: not example['remove_flag'] and not example['COQ_flag'])
    print("Invalid entries removed")

    if 'remove_flag' not in filtered_dataset.column_names or 'COQ_flag' not in filtered_dataset.column_names:
        print("flag error")
        exit(0)

    # If no invalid or COQ-contaminated entries remain, remove the related columns
    if True not in filtered_dataset['remove_flag'] and True not in filtered_dataset['COQ_flag']:
        filtered_dataset = filtered_dataset.remove_columns(columns_to_remove)
        print("Preprocessing successful")
    else:
        print("Preprocessing failed")
        print("Filtered dataset not saved")
        exit(0)

    print(f"Files contaminated by COQ module declarations: {number_of_COQ_modules} out of {original_dataset_rows} files")
    print(f"Reduction after COQ module declaration removal of {number_of_COQ_modules / original_dataset_rows * 100:.2f}% of original files.")

    print(f"COQ filepaths: {number_of_COQ_files} out of {original_dataset_rows} files")
    print(f"Reduction after COQ removal of {number_of_COQ_files / original_dataset_rows * 100:.2f}% of original files.")

    print(f"Files removed due to non-synthesizable keywords in path: {number_of_invalid_paths}")
    print(f"Reduction after invalid path removal of {number_of_invalid_paths / original_dataset_rows * 100:.2f}% of original files.")

    print(f"Files removed due to empty content: {number_of_empty_content}")
    print(f"Reduction after empty content removal of {number_of_empty_content / original_dataset_rows * 100:.2f}% of original files.")

    print(f"Files removed due to no module declarations: {number_of_no_module_declarations}")
    print(f"Reduction after no module declarations removal of {number_of_no_module_declarations / original_dataset_rows * 100:.2f}% of original files.")

    print(f"Total files removed: {number_of_empty_content + number_of_no_module_declarations + number_of_invalid_paths + number_of_COQ_files}")

    print("Unbalanced begin/end blocks: ", unbalanced_counter)
    print("COQ keywords missed: ", missed_COQ)

    return filtered_dataset

if __name__ == '__main__':
    filtered_dataset = main(dataset)
    new_dataset_rows = filtered_dataset.num_rows
    new_dataset_size = sum(filtered_dataset['size'])

    print("Original dataset rows: ", original_dataset_rows)
    print("New dataset rows: ", new_dataset_rows)
    print(f"Original dataset size: {original_dataset_size / (2 ** 30):.2f} GB")
    print(f"New dataset size: {new_dataset_size / (2 ** 30):.2f} GB")
    print("Row reduction of ", math.ceil((original_dataset_rows - new_dataset_rows) / original_dataset_rows * 100),
          "% in dataset rows")
    print(
        f"Size reduction of {((original_dataset_size - new_dataset_size) / original_dataset_size * 100):.2f}% in dataset size")

    print("Conclusion")

    # Read the Hugging Face API write token from file
    write_token_file = "write_token.txt"
    with open(write_token_file, "r") as file:
        hf_write_token = file.read().strip()
    filtered_dataset.push_to_hub('Von-R/test', token=hf_write_token)
