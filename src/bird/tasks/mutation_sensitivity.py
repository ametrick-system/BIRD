import json
import os
import random
from bird.tasks.common import Task

class MutationSensitivity(Task):
    """
    Mutation Sensitivity evaluation task.
    Evaluates whether a model can identify if a given DNA sequence 
    with 7 random point mutations is biologically viable.
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
        return 'categorical'

    def num_examples(self):
        return len(self.data)

    def get_example(self, index):
        """ Get a single sequence problem, mutate it 7 times, and determine viability. """
        row = self.data[index]
        original_sequence = row.get('full_sequence', "")
        structure = row.get('structure', {})
        
        # 1. Inject 7 deterministic random mutations
        # We seed random with the index so the exact same 7 mutations happen 
        # for this specific sequence every time it is evaluated across epochs.
        rng = random.Random(index)
        mutated_chars = list(original_sequence)
        
        for _ in range(4):
            mut_idx = rng.randint(0, len(mutated_chars) - 1)
            orig_base = mutated_chars[mut_idx]
            new_base = rng.choice([b for b in 'ACGT' if b != orig_base])
            mutated_chars[mut_idx] = new_base
            
        mutated_sequence = "".join(mutated_chars)
        
        # 2. Map the boundaries of all critical structural regions
        # Since point mutations don't change the sequence length, 
        # the indices of the regions remain exactly the same.
        current_idx = 0
        
        current_idx += len(structure.get('utr_1', ''))
        
        enhancer_start = current_idx
        current_idx += len(structure.get('enhancer', ''))
        enhancer_end = current_idx
        
        current_idx += len(structure.get('utr_2', ''))
        
        promoter_start = current_idx
        current_idx += len(structure.get('promoter', ''))
        promoter_end = current_idx
        
        current_idx += len(structure.get('utr_3', ''))
        
        exon_boundaries = []
        for orf in structure.get('orf_region', []):
            length = len(orf['sequence'])
            if orf['type'] == 'exon':
                exon_boundaries.append((current_idx, current_idx + length))
            current_idx += length
            
        # 3. Determine biological viability based on the rules applied to the FINAL sequence
        viable = True
        
        # Rule A: The highly conserved TATAAT promoter must be perfectly intact
        mut_promoter = mutated_sequence[promoter_start:promoter_end]
        if mut_promoter != "TATAAT":
            viable = False
            
        # Rule C: No premature stop codons in ANY of the exons
        if viable:
            stop_codons = ["ATT", "ACT", "ATC"]
            for start, end in exon_boundaries:
                mut_exon = mutated_sequence[start:end]
                # Slide a 3-mer window across the entire exon to check for stop codons
                for i in range(len(mut_exon) - 3):
                    if mut_exon[i:i+3] in stop_codons:
                        viable = False
                        break
                if not viable:
                    break # Break out of the outer exon loop if a stop codon is found
                
        # Format the final answer mapping for the model
        answer = "1" if viable else "0"
        
        messages = [
            {"role": "user", "content": mutated_sequence},
            {"role": "assistant", "content": answer},
        ]
        
        return {"messages": messages}

    def evaluate(self, conversation, assistant_response):
        """
        Given (conversation, completion), return evaluation outcome (0 = wrong, 1 = correct)
        """
        assert isinstance(assistant_response, str), "Assuming simple string response for now"
        
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == "assistant", "Last message must be from the Assistant"
        
        ref_answer = assistant_message['content'].strip() 
        pred_answer = assistant_response.strip()
        
        is_correct = int(pred_answer == ref_answer)
        return is_correct

if __name__ == "__main__":
    # 1. Initialize the dataset
    # (If your file is named something else like "DNA (1).jsonl", update the filepath here)
    ds = MutationSensitivity(filepath="DNA.jsonl")
    
    print(f"Loaded {len(ds)} sequences. Running mutations and tallying results...")
    
    good_count = 0
    bad_count = 0
    
    # 2. Loop through every sequence in the dataset
    for i in range(len(ds)):
        # ds[i] calls the get_example() method, which does the mutations and rules checks
        example = ds[i]
        
        # Extract the final answer ("1" or "0") from the assistant's message
        answer = example["messages"][-1]["content"]
        
        if answer == "1":
            good_count += 1
        elif answer == "0":
            bad_count += 1
            
        # Optional: Print progress every 1000 sequences so you know it's working
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1} / {len(ds)}...")
      
    # # 3. Print the final tally
    # print("\n--- Final Results ---")
    # print(f"Good (Viable / '1'): {good_count}")
    # print(f"Bad (Non-viable / '0'): {bad_count}")