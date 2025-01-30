from grpo.grpo import GRPOTrainer
from grpo.policy import PolicyModel
from grpo.reward import correctness_reward_func, int_reward_func, strict_format_reward_func, soft_format_reward_func, xmlcount_reward_func


# Initialize models and trainer
policy_model = PolicyModel(base_model_name="gpt2")

reward_functions = [
    correctness_reward_func,
    int_reward_func,
    strict_format_reward_func,
    soft_format_reward_func,
    xmlcount_reward_func
]

reward_weights = [1.0, 0.5, 0.3, 0.3, 0.4]

trainer = GRPOTrainer(
    policy_model=policy_model,
    reward_functions=reward_functions,
    reward_weights=reward_weights,
    epsilon=0.2,
    beta=0.01,
    group_size=16
)

# Training loop
questions = ["What is 2+2?"]
answers = ["4"]

for question, answer in zip(questions, answers):
    loss, avg_reward = trainer.train_step(question, answer)
    print(f"Loss: {loss:.4f}, Average Reward: {avg_reward:.4f}")