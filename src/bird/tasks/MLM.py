import json
import os
import random
from bird.tasks.common import Task

class MaskedLanguageModeling(Task):
    """
    Masked Language Modeling (MLM) evaluation task.
    Evaluates whether a model can reconstruct a DNA sequence where 
    15% of the nucleotides have been randomly replaced with an 'M' mask token.
    """

    def __init__(self, filepath="DNA.jsonl", **kwargs):
        # Initialize the base class which handles slicing (start, stop, step)
        super().__init__(**kwargs)
        
        self.data = []
        # Load the JSONL dataset into memory
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.data.append(json.loads(line))
        else:
            print(f"Warning: Dataset file {filepath} not found.")

    @property
    def eval_type(self):
        # The model must generate the fully restored sequence
        return 'generative'

    def num_examples(self):
        return len(self.data)

    def get_example(self, index):
        """ Corrupt the sequence with masks and return the input/target pair. """
        row = self.data[index]
        original_sequence = row.get('full_sequence', "")
        
        # We seed random with the index to ensure the exact same base pairs 
        # are masked for this sequence every time it is evaluated.
        rng = random.Random(index)
        mask_prob = 0.15 # Standard BERT masking ratio
        
        masked_chars = []
        for char in original_sequence:
            if rng.random() < mask_prob:
                masked_chars.append('M') # 'M' acts as our [MASK] token
            else:
                masked_chars.append(char)
                
        masked_sequence = "".join(masked_chars)
        
        # Format the final answer mapping for the model
        messages = [
            {"role": "user", "content": masked_sequence},
            {"role": "assistant", "content": original_sequence},
        ]
        
        return {"messages": messages}

    def evaluate(self, conversation, assistant_response):
        """
        Given (conversation, completion), return the accuracy of the unmasked tokens.
        Instead of a binary 1 or 0, this returns a float representing the fraction 
        of masked tokens the model successfully predicted.
        """
        assert isinstance(assistant_response, str), "Assuming simple string response for now"
        
        # Extract the original masked input to know WHERE to check
        masked_input = conversation['messages'][0]['content'].strip()
        
        # Extract the ground truth (unmasked) sequence
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == "assistant", "Last message must be from the Assistant"
        ref_seq = assistant_message['content'].strip() 
        
        # Get the model's generated sequence
        pred_seq = assistant_response.strip()
        
        # If the model fails to generate a sequence of the correct length, 
        # it fails the generative sequence-to-sequence structure entirely.
        if len(pred_seq) != len(ref_seq):
            return 0.0
            
        # Tally the accuracy ONLY for the tokens that were actually masked
        correct_predictions = 0
        total_masked = 0
        
        for m_char, ref_char, pred_char in zip(masked_input, ref_seq, pred_seq):
            if m_char == 'M':
                total_masked += 1
                if pred_char == ref_char:
                    correct_predictions += 1
                    
        # Edge case: if random chance resulted in 0 masked tokens
        if total_masked == 0:
            return 1.0
            
        # Return a float score (e.g., 0.85 if it got 85% of the masked tokens right)
        return float(correct_predictions) / float(total_masked)