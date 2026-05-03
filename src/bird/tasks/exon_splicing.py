import json
import os
from bird.tasks.common import Task

class ExonSplicing(Task):
    """
    Exon Splicing evaluation task.
    Evaluates whether a model can correctly identify and concatenate
    only the exon sequences from a full DNA string, effectively
    splicing out the introns, UTRs, and promoter regions.
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
        # Unlike the first two tasks, this requires the model to generate a full sequence
        return 'generative'

    def num_examples(self):
        return len(self.data)

    def get_example(self, index):
        """ Get a single sequence problem and build the spliced target string. """
        row = self.data[index]
        
        # The prompt is the full, unedited sequence with all junk DNA included
        question = row.get('full_sequence', "")
        
        # To get the ground-truth answer, we iterate through the orf_region map
        # and concatenate only the segments labeled as 'exon'
        structure = row.get('structure', {})
        orf_region = structure.get('orf_region', [])
        
        spliced_sequence = ""
        for segment in orf_region:
            if segment.get('type') == 'exon':
                spliced_sequence += segment.get('sequence', "")
                
        # Format the final answer mapping for the model
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": spliced_sequence},
        ]
        
        return {"messages": messages}

    def evaluate(self, conversation, assistant_response):
        """
        Given (conversation, completion), return evaluation outcome (0 = wrong, 1 = correct)
        """
        assert isinstance(assistant_response, str), "Assuming simple string response for now"
        
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == "assistant", "Last message must be from the Assistant"
        
        # Get the clean reference sequence
        ref_answer = assistant_message['content'].strip() 
        # Get the model's generated sequence
        pred_answer = assistant_response.strip()
        
        # For a strict evaluation, the model's generated sequence must exactly match 
        # the true spliced sequence. 
        is_correct = int(pred_answer == ref_answer)
        return is_correct