import json
import os
from bird.tasks.common import Task

class MotifRecognition(Task):
    """
    Motif Recognition evaluation task.
    Evaluates whether a model can identify if a given DNA sequence 
    contains a functional enhancer.
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
        # Since the output is strictly bounded to 1 or 0, this acts as a categorical task. 
        # You could also use 'generative' depending on how your main loop handles logits.
        return 'categorical'

    def num_examples(self):
        return len(self.data)

    def get_example(self, index):
        """ Get a single sequence problem from the dataset. """
        row = self.data[index]
        
        # The prompt is strictly the full sequence with nothing else, as requested
        question = row.get('full_sequence', "")
        
        # Extract the boolean and convert it to "1" or "0"
        is_enhancer = row['metadata'].get('functional_enhancer', False)
        answer = "1" if is_enhancer else "0"
        
        # Format matching the nanochat style conversation
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        
        conversation = {
            "messages": messages,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        Given (conversation, completion), return evaluation outcome (0 = wrong, 1 = correct)
        """
        assert isinstance(assistant_response, str), "Assuming simple string response for now"
        
        # Extract the ground truth answer
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == "assistant", "Last message must be from the Assistant"
        
        # Get the reference answer ("1" or "0")
        ref_answer = assistant_message['content'].strip() 
        
        # Get the model's prediction, stripping any trailing/leading whitespaces or newlines
        pred_answer = assistant_response.strip()
        
        # Compare and return success as an integer (1 if correct, 0 if wrong)
        is_correct = int(pred_answer == ref_answer)
        return is_correct