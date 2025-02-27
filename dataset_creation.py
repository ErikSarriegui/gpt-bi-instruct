"""
=================================================
PREPROCESS_DATA.py
=================================================
"""
from collections import defaultdict
from datasets import Dataset
import huggingface_hub
import datasets

# Variables globales
HF_TOKEN = "<token>"
DATASET_1 = "xezpeleta/oasst1_eu"
DATASET_2 = "xezpeleta/oasst2_eu"
OUTPUT_DATASET = "AuriLab/combined_oasst_2"

def convert_oa_to_threaded_json(dataset_name):
    """
    Convierte un dataset de Open Assistant de Hugging Face a un formato JSON con hilos anidados.
    
    Args:
        dataset_name: Nombre del dataset en Hugging Face
    
    Returns:
        Lista de diccionarios con el formato de hilos anidados
    """
    dataset = datasets.load_dataset(dataset_name)
    message_map = {}
    replies_map = defaultdict(list)
    root_messages = set()
    
    for split in dataset.keys():
        for item in dataset[split]:
            message_id = item.get('message_id')
            parent_id = item.get('parent_id')
            
            try:
                rank = 0
                if 'rank' in item and item['rank'] is not None:
                    rank = float(item['rank'])
            except (ValueError, TypeError):
                rank = 0
            
            message_map[message_id] = {
                'text': item.get('text', ''),
                'role': 'prompter' if item.get('role', '') == 'prompter' else 'assistant',
                'meta': {
                    'lang': item.get('lang', 'en'),
                    'rank': rank
                },
                'message_id': message_id,
                'parent_id': parent_id
            }
            
            if parent_id:
                replies_map[parent_id].append(message_id)
            else:
                root_messages.add(message_id)
    
    def build_replies_tree(message_id):
        if message_id not in message_map:
            print(f"Advertencia: ID de mensaje {message_id} no encontrado")
            return {"text": "", "role": "assistant", "meta": {"rank": 0}, "replies": []}
        
        message = message_map[message_id].copy()
        reply_ids = replies_map.get(message_id, [])
        replies = []
        
        for reply_id in reply_ids:
            reply_tree = build_replies_tree(reply_id)
            replies.append(reply_tree)
        
        for reply in replies:
            if 'meta' not in reply:
                reply['meta'] = {}
            if 'rank' not in reply['meta'] or reply['meta']['rank'] is None:
                reply['meta']['rank'] = 0
        
        replies.sort(key=lambda x: x.get('meta', {}).get('rank', 0))
        message['replies'] = replies
        
        if 'message_id' in message:
            del message['message_id']
        if 'parent_id' in message:
            del message['parent_id']
            
        return message
    
    result = []
    for root_id in root_messages:
        try:
            thread = build_replies_tree(root_id)
            thread_obj = {
                'thread': thread,
                'source': 'open_assistant',
                'meta': {
                    'post_id': root_id
                }
            }
            result.append(thread_obj)
        except Exception as e:
            print(f"Error procesando el hilo {root_id}: {str(e)}")
    
    return result

def extract_conversations(data_list):
    """
    Extrae todas las posibles conversaciones de un conjunto de datos con formato de hilos anidados.
    
    Args:
        data_list: Lista de diccionarios con hilos de conversación
        
    Returns:
        Lista de todas las posibles conversaciones con identificadores de publicación
    """
    all_conversations = []

    def traverse_replies(thread_path, current_replies, post_id):
        for reply in current_replies:
            new_path = thread_path.copy()
            new_path.append({
                'text': reply['text'],
                'role': reply['role']
            })
            if not reply['replies']:
                all_conversations.append({
                    'post_id': post_id,
                    'messages': new_path
                })
            else:
                traverse_replies(new_path, reply['replies'], post_id)

    for item in data_list:
        thread = item['thread']
        post_id = item.get('meta', {}).get('post_id')
        base_message = [{
            'text': thread['text'],
            'role': thread['role']
        }]
        if not thread['replies']:
            all_conversations.append({
                'post_id': post_id,
                'messages': base_message
            })
        else:
            traverse_replies(base_message, thread['replies'], post_id)

    return all_conversations

def procesar(example):
    """
    Procesa cada ejemplo del dataset para convertir los roles a formatos estándar.
    
    Args:
        example: Diccionario con los datos de conversación
        
    Returns:
        Diccionario con mensajes procesados
    """
    example["processed_messages"] = [{"role" : "user" if message["role"] == "prompter" else "assistant", "content" : message["text"]} for message in example["messages"]]
    return example

def main():
    """
    Función principal que ejecuta todo el proceso de conversión y publicación del dataset.
    """
    huggingface_hub.login(HF_TOKEN)

    threaded_data_1 = convert_oa_to_threaded_json(DATASET_1)
    threaded_data_2 = convert_oa_to_threaded_json(DATASET_2)

    conversaciones_1 = extract_conversations(threaded_data_1)
    conversaciones_2 = extract_conversations(threaded_data_2)

    full_dataset = conversaciones_1 + conversaciones_2
    dataset = Dataset.from_list(full_dataset)
    processed_dataset = dataset.map(procesar)
    processed_dataset = processed_dataset.remove_columns(["messages"])
    processed_dataset = processed_dataset.rename_column("processed_messages", "messages")

    processed_dataset = processed_dataset.train_test_split(test_size = 0.02)
    
    processed_dataset.push_to_hub(OUTPUT_DATASET)

if __name__ == "__main__":
    main()