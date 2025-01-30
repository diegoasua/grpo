import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional

class GRPO:
    def __init__(
        self,
        policy_model: nn.Module,
        reward_model: nn.Module,
        epsilon: float = 0.2,
        beta: float = 0.01,
        group_size: int = 16,
        max_iterations: int = 1000,
        steps_per_iteration: int = 50,
        grpo_iterations: int = 10
    ):
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.epsilon = epsilon
        self.beta = beta
        self.G = group_size
        self.max_iterations = max_iterations
        self.M = steps_per_iteration
        self.mu = grpo_iterations
        
        # Initialize old policy model with same architecture
        self.old_policy = type(policy_model)(*policy_model.args)
        self.old_policy.load_state_dict(policy_model.state_dict())
        
        self.optimizer = torch.optim.Adam(self.policy_model.parameters())
        
    def compute_advantage(self, rewards: torch.Tensor) -> torch.Tensor:
        """Compute advantage using group relative advantage estimation"""
        mean_reward = torch.mean(rewards)
        std_reward = torch.std(rewards) + 1e-8  # Add small constant for numerical stability
        return (rewards - mean_reward) / std_reward
    
    def compute_kl_divergence(self, pi_theta: torch.Tensor, pi_ref: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence between current and reference policy"""
        return (pi_theta / pi_ref) - torch.log(pi_theta / pi_ref) - 1
    
    def grpo_loss(
        self,
        current_probs: torch.Tensor,
        old_probs: torch.Tensor,
        advantages: torch.Tensor,
        ref_probs: torch.Tensor
    ) -> torch.Tensor:
        """Compute GRPO objective function"""
        ratio = current_probs / old_probs
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        
        # First term: policy gradient with clipping
        pg_loss = torch.min(
            ratio * advantages,
            clipped_ratio * advantages
        )
        
        # Second term: KL penalty
        kl_div = self.compute_kl_divergence(current_probs, ref_probs)
        
        return -(pg_loss - self.beta * kl_div).mean()
    
    def sample_outputs(self, question: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample G outputs from the policy for a given question"""
        with torch.no_grad():
            outputs = []
            probs = []
            for _ in range(self.G):
                logits = self.old_policy(question)
                prob = torch.softmax(logits, dim=-1)
                output = torch.multinomial(prob, 1)
                outputs.append(output)
                probs.append(prob.gather(-1, output))
            
            return torch.cat(outputs), torch.cat(probs)
    
    def train_step(self, batch: List[torch.Tensor]):
        """Perform one training step"""
        questions = batch
        
        for step in range(self.M):
            # Sample G outputs for each question
            all_outputs = []
            all_old_probs = []
            
            for q in questions:
                outputs, probs = self.sample_outputs(q)
                all_outputs.append(outputs)
                all_old_probs.append(probs)
            
            # Compute rewards using reward model
            rewards = self.reward_model(torch.stack(all_outputs))
            advantages = self.compute_advantage(rewards)
            
            # Store reference policy
            ref_policy = self.policy_model
            
            # GRPO iterations
            for _ in range(self.mu):
                current_probs = self.policy_model(questions)
                ref_probs = ref_policy(questions)
                
                loss = self.grpo_loss(
                    current_probs,
                    torch.stack(all_old_probs),
                    advantages,
                    ref_probs
                )
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Update old policy
            self.old_policy.load_state_dict(self.policy_model.state_dict())

class GRPOTrainer:
    def __init__(
        self,
        policy_model: PolicyModel,
        reward_functions: List[callable],
        reward_weights: List[float],
        epsilon: float = 0.2,
        beta: float = 0.01,
        group_size: int = 16,
    ):
        self.policy = policy_model
        self.reward_functions = reward_functions
        self.reward_weights = reward_weights
        self.epsilon = epsilon
        self.beta = beta
        self.G = group_size
        
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=1e-5)
    
    def compute_total_reward(
        self,
        prompts: List[Dict],
        completions: List[Dict],
        answer: Optional[List[str]] = None
    ) -> torch.Tensor:
        """Compute weighted sum of all rewards"""
        total_reward = torch.zeros(len(completions))
        
        for func, weight in zip(self.reward_functions, self.reward_weights):
            kwargs = {"prompts": prompts, "answer": answer} if answer else {}
            reward = torch.tensor(func(completions=completions, **kwargs))
            total_reward += weight * reward
            
        return total_reward
    
    def train_step(
        self,
        question: str,
        answer: Optional[str] = None
    ):
        """Perform one training step"""
        # Create prompt
        prompt = [{"role": "user", "content": question}]
        
        # Generate group of responses
        completions = []
        for _ in range(self.G):
            completion = self.policy.generate_response(
                question,
                temperature=1.0,
                num_return_sequences=1
            )
            completions.append(completion)
        
        # Compute rewards
        rewards = self.compute_total_reward(
            prompts=[prompt] * self.G,
            completions=completions,
            answer=[answer] if answer else None
        )
        
        # Compute advantage
        advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Compute policy loss
        loss = -advantage.mean()  # Simplified policy gradient loss
        
        # Update policy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), rewards.mean().item()