"""
Dataset module for Qwen3-EdgeRazor.
"""
# ruff: noqa: F401

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer

try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

ignore_id = -100  # Label ID for ignoring loss computation


class ReasoningDataset(Dataset):
    def __init__(self, dataset_path: str | list[str], tokenizer: AutoTokenizer, max_seq_len=4096, split='train',
                 num_insertions=10, add_system_prompt=True,
                 limit_num_samples=None, limit_num_samples_by_random=False):
        """
        Process multi-turn conversation dataset, training all assistant responses without padding.

        Args:
            dataset_path: Path to HuggingFace dataset or local JSONL/JSON file(s).
            tokenizer: Pretrained tokenizer.
            max_seq_len: Maximum sequence length (sequences exceeding this will be truncated).
            split: Dataset split (default 'train').
            num_insertions: Number of watermark insertions (default 10).
            add_system_prompt: Whether to add system prompt (default False).
            limit_num_samples: Limit the number of loaded samples, None means load all.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.eos_token_id = tokenizer.eos_token_id
        self.add_system_prompt = add_system_prompt

        # Load dataset: supports local JSONL/JSON files (single or multiple), HuggingFace datasets, and mixed mode
        if isinstance(dataset_path, list):
            # Multiple data sources (may be mixed local files and HuggingFace datasets)
            local_files = [p for p in dataset_path if isinstance(p, str) and (p.endswith('.jsonl') or p.endswith('.json'))]
            hf_datasets = [p for p in dataset_path if isinstance(p, str) and not (p.endswith('.jsonl') or p.endswith('.json'))]

            datasets_to_concat = []

            if local_files:
                print(f"Loading local files: {local_files}")
                local_dataset = load_dataset('json', data_files={'train': local_files})
                datasets_to_concat.append(local_dataset['train'])

            for hf_path in hf_datasets:
                print(f"Loading HuggingFace dataset: {hf_path}")
                hf_dataset = load_dataset(hf_path)
                datasets_to_concat.append(hf_dataset[split])

            if len(datasets_to_concat) == 0:
                raise ValueError("No valid datasets found in the list")
            elif len(datasets_to_concat) == 1:
                raw_dataset = {split: datasets_to_concat[0]}
            else:
                from datasets import concatenate_datasets
                merged_dataset = concatenate_datasets(datasets_to_concat)
                raw_dataset = {split: merged_dataset}

        elif isinstance(dataset_path, str) and (dataset_path.endswith('.jsonl') or dataset_path.endswith('.json')):
            raw_dataset = load_dataset('json', data_files={'train': dataset_path})
        else:
            raw_dataset = load_dataset(dataset_path)
        print(f"Dataset loaded: {dataset_path}, total samples: {len(raw_dataset[split])}")
        self.dataset = raw_dataset[split]

        if limit_num_samples is not None and limit_num_samples > 0:
            original_size = len(self.dataset)
            if limit_num_samples < original_size:
                if limit_num_samples_by_random:
                    self.dataset = self.dataset.shuffle(seed=3407).select(range(limit_num_samples))
                    print(f"Randomly sampled {limit_num_samples} samples (original size: {original_size})")
                else:
                    self.dataset = self.dataset.select(range(limit_num_samples))
                    print(f"Selected first {limit_num_samples} samples (original size: {original_size})")
            else:
                print(f"Warning: limit_num_samples ({limit_num_samples}) >= dataset size ({original_size}), using all data")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        messages = item['messages']

        if self.add_system_prompt:
            conversation = [{
                "role": "system",
                "content": "You are a helpful and harmless AI assistant."
            }] + messages
        else:
            conversation = messages

        full_text = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        input_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        labels = [ignore_id] * len(input_ids)

        temp_messages = []
        for _, msg in enumerate(conversation):
            temp_messages.append(msg)

            if msg['role'] == 'assistant':
                current_text = self.tokenizer.apply_chat_template(
                    temp_messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                current_ids = self.tokenizer.encode(current_text, add_special_tokens=False)

                prev_text = self.tokenizer.apply_chat_template(
                    temp_messages[:-1],
                    tokenize=False,
                    add_generation_prompt=True
                )
                prev_ids = self.tokenizer.encode(prev_text, add_special_tokens=False)

                # Mark assistant response tokens for training (from prev_ids end to current_ids end)
                start_pos = len(prev_ids)
                end_pos = len(current_ids)

                for pos in range(start_pos, min(end_pos, len(labels))):
                    labels[pos] = input_ids[pos]

        if len(input_ids) > self.max_seq_len - 1:
            input_ids = input_ids[:self.max_seq_len - 1]
            labels = labels[:self.max_seq_len - 1]

        input_ids.append(self.eos_token_id)
        labels.append(self.eos_token_id)
        
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels)
        }
    
    def get_token_stats(self, plot_distribution=False, output_path="./sequence_length_distribution.png"):
        """
        Compute dataset statistics in a single pass:
        - Total token count (after truncation)
        - Maximum sequence length (without truncation)
        - Maximum CoT (chain-of-thought) length
        - Maximum answer length
        - Sequence length distribution (if plotting enabled)

        Args:
            plot_distribution: Whether to plot sequence length distribution.
            output_path: Path to save the distribution plot.

        Returns:
            tuple: (total_tokens, max_len, max_len_cot, max_len_answer, seq_lengths)
        """
        total_tokens = 0
        max_length = 0
        max_cot_length = 0
        max_answer_length = 0
        seq_lengths = []

        for i in range(len(self.dataset)):
            item = self.dataset[i]
            messages = item['messages']

            if self.add_system_prompt:
                conversation = [{
                    "role": "system",
                    "content": "You are a helpful and harmless AI assistant."
                },] + messages
            else:
                conversation = messages

            full_text = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False
            )
            input_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

            sequence_length = len(input_ids) + 1  # +1 for EOS token
            seq_lengths.append(sequence_length)

            total_tokens += min(sequence_length, self.max_seq_len)

            if sequence_length > max_length:
                max_length = sequence_length

            for msg in messages:
                if msg['role'] == 'assistant':
                    if 'info' in msg and msg['info']:
                        think_content = msg['info'].get('think_content')
                        answer_content = msg['info'].get('answer_content')

                        if think_content:
                            think_tokens = self.tokenizer.encode(think_content, add_special_tokens=False)
                            if len(think_tokens) > max_cot_length:
                                max_cot_length = len(think_tokens)

                        if answer_content:
                            answer_tokens = self.tokenizer.encode(answer_content, add_special_tokens=False)
                            if len(answer_tokens) > max_answer_length:
                                max_answer_length = len(answer_tokens)

        if plot_distribution:
            self._plot_dist_seqlen(seq_lengths, output_path)

        return total_tokens, max_length, max_cot_length, max_answer_length, seq_lengths

    def _plot_dist_seqlen(self, seq_lengths: list, output_path: str):
        """
        Plot sequence length distribution.

        Args:
            seq_lengths: List of sequence lengths.
            output_path: Output image path.
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib not installed, cannot plot distribution. Run: pip install matplotlib")
            return

        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Sequence Length Distribution Analysis', fontsize=16, fontweight='bold')

        seq_array = np.array(seq_lengths)

        # 1. Histogram - overall distribution
        ax1 = axes[0, 0]
        ax1.hist(seq_lengths, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.axvline(self.max_seq_len, color='red', linestyle='--', linewidth=2, label=f'Max seq len ({self.max_seq_len})')
        ax1.axvline(np.median(seq_array), color='green', linestyle='--', linewidth=2, label=f'Median ({np.median(seq_array):.0f})')
        ax1.set_xlabel('Sequence Length (tokens)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Histogram - Overall Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Cumulative Distribution Function (CDF)
        ax2 = axes[0, 1]
        sorted_lengths = np.sort(seq_array)
        cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths) * 100
        ax2.plot(sorted_lengths, cumulative, color='steelblue', linewidth=2)
        ax2.axvline(self.max_seq_len, color='red', linestyle='--', linewidth=2, label=f'Max seq len ({self.max_seq_len})')
        # Mark key percentiles
        for percentile in [50, 75, 90, 95, 99]:
            val = np.percentile(seq_array, percentile)
            ax2.axhline(percentile, color='gray', linestyle=':', alpha=0.5)
            ax2.text(val, percentile, f' {percentile}%: {val:.0f}', fontsize=9, verticalalignment='bottom')
        ax2.set_xlabel('Sequence Length (tokens)')
        ax2.set_ylabel('Cumulative Percentage (%)')
        ax2.set_title('Cumulative Distribution Function (CDF)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Box Plot - outlier detection
        ax3 = axes[1, 0]
        box = ax3.boxplot(seq_lengths, vert=True, patch_artist=True, widths=0.5)
        box['boxes'][0].set_facecolor('lightblue')
        ax3.axhline(self.max_seq_len, color='red', linestyle='--', linewidth=2, label=f'Max seq len ({self.max_seq_len})')
        ax3.set_ylabel('Sequence Length (tokens)')
        ax3.set_title('Box Plot - Outlier Detection')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Statistical summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_text = f"""
        Statistical Summary
        {'='*40}
        
        Total Samples:        {len(seq_lengths):,}
        
        Min Length:           {np.min(seq_array):,.0f} tokens
        Max Length:           {np.max(seq_array):,.0f} tokens
        Mean Length:          {np.mean(seq_array):,.1f} tokens
        Median Length:        {np.median(seq_array):,.0f} tokens
        Std Deviation:        {np.std(seq_array):,.1f} tokens
        
        Percentiles:
          25th (Q1):          {np.percentile(seq_array, 25):,.0f} tokens
          50th (Median):      {np.percentile(seq_array, 50):,.0f} tokens
          75th (Q3):          {np.percentile(seq_array, 75):,.0f} tokens
          90th:               {np.percentile(seq_array, 90):,.0f} tokens
          95th:               {np.percentile(seq_array, 95):,.0f} tokens
          99th:               {np.percentile(seq_array, 99):,.0f} tokens
        
        Truncation Analysis:
          Max Seq Len:        {self.max_seq_len:,} tokens
          Samples > Max:      {np.sum(seq_array > self.max_seq_len):,} ({np.sum(seq_array > self.max_seq_len)/len(seq_array)*100:.2f}%)
          Samples ≤ Max:      {np.sum(seq_array <= self.max_seq_len):,} ({np.sum(seq_array <= self.max_seq_len)/len(seq_array)*100:.2f}%)
        """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.3})

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Sequence length distribution plot saved to: {output_path}")
