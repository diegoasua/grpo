import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, List

class PolicyModel(nn.Module):
    def __init__(
        self,
        base_model_name: str = "gpt2",
        max_length: int = 512,
    ):
        super().__init__()
        self.args = (base_model_name, max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.max_length = max_length
        
        # Add special tokens for XML tags
        special_tokens = [
            "<reasoning>", "</reasoning>",
            "<answer>", "</answer>",
            "\n"
        ]
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the policy model.
        Returns logits over the vocabulary for each position.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return outputs.logits

    def generate_response(
        self,
        question: str,
        temperature: float = 1.0,
        num_return_sequences: int = 1
    ) -> List[Dict]:
        """
        Generate a response in the required XML format.
        """
        # Prepare input
        inputs = self.tokenizer(
            question,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        )
        
        # Generate
        outputs = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=self.max_length,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True
        )
        
        # Decode outputs
        responses = []
        for output in outputs:
            decoded = self.tokenizer.decode(output, skip_special_tokens=False)
            responses.append({
                "role": "assistant",
                "content": self._format_response(decoded)
            })
        
        return responses

    def _format_response(self, text: str) -> str:
        """
        Ensure the response follows the required XML format.
        """
        # Split into reasoning and answer parts
        parts = text.split("<reasoning>")
        if len(parts) < 2:
            return self._create_default_response()
        
        reasoning_and_rest = parts[1].split("</reasoning>")
        if len(reasoning_and_rest) < 2:
            return self._create_default_response()
        
        reasoning = reasoning_and_rest[0].strip()
        
        answer_parts = reasoning_and_rest[1].split("<answer>")
        if len(answer_parts) < 2:
            return self._create_default_response()
        
        answer = answer_parts[1].split("</answer>")[0].strip()
        
        # Format the response properly
        formatted_response = (
            "<reasoning>\n"
            f"{reasoning}\n"
            "</reasoning>\n"
            "<answer>\n"
            f"{answer}\n"
            "</answer>\n"
        )
        
        return formatted_response

    def _create_default_response(self) -> str:
        """
        Create a default response when formatting fails.
        """
        return (
            "<reasoning>\n"
            "Unable to generate proper reasoning.\n"
            "</reasoning>\n"
            "<answer>\n"
            "0\n"
            "</answer>\n"
        )