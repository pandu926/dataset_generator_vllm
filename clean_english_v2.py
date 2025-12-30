import json
import re

def clean_dataset(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cleaned_count = 0
    total_removed_messages = 0

    # Pattern for English instructions/metadata
    # We add ^ at start to match beginning of the string
    instruction_patterns = [
        r"^Process\s",
        r"^Directly\s",
        r"^Respond\s",
        r"^Briefly\s",
        r"^Provide\s",
        r"^Confirm\s",
        r"^Acknowledge\s",
        r"^I will\s",
        r"^I regret\s",
        r"^State\s",
        r"^Explain\s",
        r"^Clarify\s",
        r"^Inform\s",
        r"^Present\s",
        r"^The answer is\s",
        r"^Based on\s",
        r"^Clearly\s",
        r"^The following\s",
        r"^Note\s",
        r"^Correct\s",
        r"^Yes, correct\s",  # Sometimes model says "Yes, correct" in English then translation? Or maybe simple answer. 
                             # But "Yes, correct" might be a valid answer.
                             # Let's stick to instruction-like verbs.
        r"^Answer:\s",
        r"^Instruction:\s",
    ]

    combined_pattern = re.compile('|'.join(instruction_patterns), re.IGNORECASE)

    for entry in data:
        conversation = entry.get('conversation', [])
        new_conversation = []
        
        # We process the conversation to remove unwanted model messages
        # Strategy: 
        # Iterate through messages. If it's a model message, check if it matches pattern.
        # If it matches pattern AND we suspect it's an instruction (maybe check if there is another model message right after or before? 
        # Actually user said "remove any remaining model responses that are in English and are instructions".
        # So if it matches the instruction pattern, we delete it.
        
        for msg in conversation:
            if msg['role'] == 'model':
                content = msg['content']
                if combined_pattern.match(content):
                    # It matches an instruction pattern
                    total_removed_messages += 1
                    continue # Skip adding this message
                
                # Check for "I ..." which is common in "I will..."
                # Also "The..."
                if content.startswith("I ") and ("provide" in content or "answer" in content or "state" in content or "inform" in content):
                     total_removed_messages += 1
                     continue
                
                new_conversation.append(msg)
            else:
                new_conversation.append(msg)
        
        if len(new_conversation) != len(conversation):
            cleaned_count += 1
            entry['conversation'] = new_conversation
            # Update num_turns if necessary (usually counts pairs or total turns)
            entry['num_turns'] = len(new_conversation)

    print(f"Removed {total_removed_messages} messages from {cleaned_count} entries.")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    input_path = '/workspace/dataset_generator_vllm/data/raw/categories/multiturn_snk_consecutive_model.json'
    # formatting back to the same file or a new one? 
    # User said "clean the `multiturn_snk_consecutive_model.json` dataset".
    # I'll overwrite it to be direct, or write to a temp one then rename.
    # Let's write to the same file to be consistent with previous steps.
    clean_dataset(input_path, input_path)
