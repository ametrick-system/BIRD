import random
import json

class GenomicDataGenerator:
    def __init__(self):
        self.bases = ['A', 'C', 'G', 'T']
        self.antibases = ['T', 'G', 'C', 'A']
        self.promoter_seq = "TATAAT"
        self.stop_codons = ["ATT", "ACT", "ATC"]
        # HMM Probabilities: [A, C, G, T]
        self.exon_dist = [0.25, 0.25, 0.25, 0.25] # Balanced
        self.intron_dist = [0.35, 0.15, 0.15, 0.35]   # A/T rich
        
    def get_random_seq(self, length):
        return "".join(random.choice(self.bases) for _ in range(length))

    def generate_enhancer(self):
        """
        Generate a random enhancer of length 15-25.
        Functional rule:
        - Contains AANN followed by its complementary TTN'N'
        """
        length = random.randint(15, 25)
        base_seq = list(self.get_random_seq(length))

        # Prepare the motifs
        n_indices = [random.randint(0, 3) for _ in range(2)]
        first4 = "AA" + "".join(self.bases[i] for i in n_indices)
        second4 = "TT" + "".join(self.antibases[i] for i in n_indices)

        # Attempt random insertion
        # Ensure there is enough room (8bp for motifs + gap)
        pos1 = random.randint(0, length - 9) 
        pos2 = random.randint(pos1 + 5, length - 4)

        if random.random() < 0.7:
            base_seq[pos1:pos1+4] = list(first4)
        
        if random.random() < 0.7:
            base_seq[pos2:pos2+4] = list(second4)

        # Final Validation Scanner
        # This checks if the rule is met, whether by insertion or random chance
        full_seq = "".join(base_seq)
        is_functional = False
        
        for i in range(len(full_seq) - 7):
            if full_seq[i:i+2] == "AA":
                n1, n2 = full_seq[i+2], full_seq[i+3]
                
                comp_n1 = self.antibases[self.bases.index(n1)]
                comp_n2 = self.antibases[self.bases.index(n2)]
                target = "TT" + comp_n1 + comp_n2
                
                if target in full_seq[i+4:]:
                    is_functional = True
                    break

        return full_seq, is_functional

    def generate_orf(self):
        """Rule: HMM-based Exon/Intron switching."""
        regions = []
        current = ""
        exon = True
        labels = []     # Track which index is Exon (1) or Intron (0)
        
        length = random.randint(60, 100)
        for i in range(length):
            # Set the baseline switch prob
            switch_prob = 0.01

            if len(current) > 2:
                if current[-2:] == "GT" and exon: # Splice donor hint
                    switch_prob = 0.5
                
                if current[-2:] == "AG" and not exon:
                    switch_prob = 0.5
            
            if random.random() < switch_prob and len(current) >= 4:
                labels.append(exon)
                exon = not exon
                regions.append(current)
                current = ""
            
            # Emit nucleotide based on state
            dist = self.exon_dist if exon else self.intron_dist
            char = random.choices(self.bases, weights=dist)[0]
            
            current += char
        
        # Add the last one
        regions.append(current)
        labels.append(exon)

        
        # Final Rule: Insert stop codon at end of last exon
        # Remove it from all previous exon
        for i in range(len(regions)):
            if labels[i] == True:
                for bp in range(len(regions[i]) - 2):
                    if regions[i][bp] +regions[i][bp+1] + regions[i][bp+2] in self.stop_codons:
                        regions[i] = regions[i][:bp] + 'C' + regions[i][bp+1:]

        stop = random.choice(self.stop_codons)
        for i in range(len(regions)-1, -1, -1):
            if labels[i] == True:
                regions[i] += stop
                break
        
        return regions, labels

    def create_unit(self):
        """Assembly of the full gene unit, returning a structured JSON-ready dictionary."""
        # 1. Generate Components
        utr1 = self.get_random_seq(random.randint(4, 7))
        enhancer, functional = self.generate_enhancer()
        utr2 = self.get_random_seq(random.randint(4, 7))
        promoter = self.promoter_seq
        utr3 = self.get_random_seq(random.randint(4, 7))
        
        # We need to modify generate_orf slightly to return segments
        orf_regions, orf_labels = self.generate_orf()
        
        # 2. Parse the ORF into Exons/Introns for the JSON
        orf_segments = []
        
        for i in range(len(orf_labels)):
            if orf_labels[i]:
                current = "exon"
            else:
                current = "intron"
                
            # Append the final segment
            orf_segments.append({"type": current, "sequence": orf_regions[i]})

        orf_seq = "".join(orf_regions)
        # 3. Construct the JSON Object
        gene_data = {
            "full_sequence": f"{utr1}{enhancer}{utr2}{promoter}{utr3}{orf_seq}",
            "structure": {
                "utr_1": utr1,
                "enhancer": enhancer,
                "utr_2": utr2,
                "promoter": promoter,
                "utr_3": utr3,
                "orf_region": orf_segments
            },
            "metadata": {
                "total_length": len(utr1) + len(enhancer) + len(utr2) + len(promoter) + len(utr3) + len(orf_seq),
                "functional_enhancer": functional
            }
        }
        return gene_data, functional

def generate_and_save(num_units, filename="DNA.jsonl"):
    generator = GenomicDataGenerator()

    print(f"Generating {num_units} units to {filename}...")
    num_functional = 0
    with open(filename, 'w') as f:
        for i in range(num_units):
            # Generate the structured dictionary
            unit, functional = generator.create_unit()
            if functional:
                num_functional += 1
            
            # Serialize to string and write as a new line
            f.write(json.dumps(unit) + '\n')
            
            # Optional: Progress tracker
            if (i + 1) % 1000 == 0:
                print(f"Progress: {i + 1}/{num_units}")

    print(f"{num_functional} out of {num_units} had functional enhancers")