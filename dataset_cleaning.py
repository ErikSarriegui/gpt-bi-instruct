from collections import Counter
from datasets import DatasetDict

def limit_post_id_occurrences(dataset_dict, max_occurrences=5):
    post_id_counts = Counter()
    
    def count_post_ids(examples):
        for post_id in examples['post_id']:
            post_id_counts[post_id] += 1
        return examples
    
    _ = dataset_dict['train'].map(count_post_ids, batched=True)
    
    kept_counts = Counter()
    
    def filter_examples(examples):
        keep = []
        for post_id in examples['post_id']:
            if kept_counts[post_id] < max_occurrences:
                keep.append(True)
                kept_counts[post_id] += 1
            else:
                keep.append(False)
        
        return {
            'post_id': [pid for pid, k in zip(examples['post_id'], keep) if k],
            'messages': [msg for msg, k in zip(examples['messages'], keep) if k]
        }
    
    filtered_train = dataset_dict['train'].map(
        filter_examples, 
        batched=True, 
        remove_columns=dataset_dict['train'].column_names
    )
    
    return DatasetDict({
        'train': filtered_train,
        'test': dataset_dict['test']
    })

def token_filter(example, block_size, tokenizer):
    tokens = tokenizer.apply_chat_template(example["messages"])
    return_element = True if len(tokens) < block_size else False
    return return_element