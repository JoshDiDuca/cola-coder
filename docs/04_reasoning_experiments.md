# Reasoning Experiments

How to teach your code model to "think" before writing code -- using
chain-of-thought training and reinforcement learning.

This is where things get interesting. A base model just predicts the next token.
A reasoning model plans, considers edge cases, and produces better solutions to
hard problems. The techniques here are what separate a basic code completer from
something that can actually solve coding challenges.

---

## Table of Contents

1. [What "Reasoning" Means in LLMs](#1-what-reasoning-means-in-llms)
2. [Chain-of-Thought (CoT)](#2-chain-of-thought-cot)
3. [How Thinking Tokens Work](#3-how-thinking-tokens-work)
4. [Adding Thinking Tokens to an Existing Model](#4-adding-thinking-tokens-to-an-existing-model)
5. [CoT Fine-Tuning](#5-cot-fine-tuning)
6. [Introduction to RLVR](#6-introduction-to-rlvr)
7. [GRPO Explained](#7-grpo-explained)
8. [GRPO vs PPO](#8-grpo-vs-ppo)
9. [The Reward Function](#9-the-reward-function)
10. [Advantages Explained](#10-advantages-explained)
11. [Running the Reasoning Experiments](#11-running-the-reasoning-experiments)
12. [What Results to Expect](#12-what-results-to-expect)
13. [Ideas for Further Experimentation](#13-ideas-for-further-experimentation)

---

## 1. What "Reasoning" Means in LLMs

### Extended thinking before answering

A standard language model generates output token by token, left to right. It commits
to each token immediately. For simple tasks like completing `function add(a, b) {
return a +`, this works fine. The answer is obvious.

But for harder problems -- ones that require planning, considering multiple approaches,
or handling edge cases -- immediate generation fails. The model starts writing code
before it has a plan, and paints itself into a corner.

**Reasoning models** solve this by generating a "thinking" step before the answer:

```
Standard model:
  Prompt:  "Write a function to check if a binary tree is balanced"
  Output:  def is_balanced(root):   ...starts writing immediately...

Reasoning model:
  Prompt:  "Write a function to check if a binary tree is balanced"
  Output:  <think>
           1. A balanced tree has left/right subtree heights differing by at most 1
           2. I need to check this recursively for every node
           3. Naive approach: compute height at every node -> O(n^2)
           4. Better: return height along with balance status -> O(n)
           5. Base case: empty tree has height 0 and is balanced
           6. Recursive: check left, check right, compare heights
           </think>
           def is_balanced(root):
               def check(node):
                   if not node:
                       return 0, True
                   ...correct, efficient implementation...
```

The thinking step lets the model plan before committing. This is the same idea as
models like DeepSeek-R1 and OpenAI's o1. We implement a simplified version of this
approach.

---

## 2. Chain-of-Thought (CoT)

### Teaching the model to show its work

Chain-of-thought is a technique where the model generates step-by-step reasoning
before producing the final answer. The idea comes from a simple observation: if you
ask a person to solve a hard math problem, they do better when they write out their
work. The same applies to LLMs.

There are two ways to get a model to do CoT:

1. **Prompting:** Just ask the model to "think step by step" in the prompt. This
   works for large models (70B+) but not for small ones -- they have not seen enough
   examples of step-by-step reasoning in their training data.

2. **Training:** Fine-tune the model on examples that include reasoning traces.
   This is what we do. It works even for small models because we explicitly teach
   them the format and pattern of reasoning.

A CoT training example looks like this:

```
def has_close_elements(numbers: list[float], threshold: float) -> bool:
    """Check if any two numbers are closer than threshold."""
<think>
Let me think through this step by step:
1. I need to check if ANY two numbers in the list are closer than the threshold
2. "Closer" means the absolute difference is less than the threshold
3. I need to compare every pair of numbers - that's a nested loop
4. For each pair (i, j) where i < j, check if |numbers[i] - numbers[j]| < threshold
5. If I find any such pair, return True immediately
6. If no pair is found after checking all, return False
</think>
def has_close_elements(numbers: list[float], threshold: float) -> bool:
    """Check if any two numbers are closer than threshold."""
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
```

The model sees the prompt (function signature + docstring), then the thinking, then
the solution. After training on many such examples, it learns to generate the
thinking step naturally.

---

## 3. How Thinking Tokens Work

### `<think>` and `</think>` as boundaries

We use two special tokens to delimit the reasoning section:

```
<think>   -- start of reasoning (the model is "thinking out loud")
</think>  -- end of reasoning (time to write the actual code)
```

These are added to the tokenizer as special tokens, just like `<bos>` (beginning of
sequence) or `<eos>` (end of sequence). They get their own token IDs and their own
embedding vectors.

The full generation flow:

```
Input:  [prompt tokens]
Output: [<think>] [reasoning tokens...] [</think>] [code tokens...]

Example:
Input:  "def fib(n):\n    \"\"\"Return nth Fibonacci number.\"\"\"\n"
Output: "<think>Fibonacci: each number is sum of previous two.
         Base cases: fib(0)=0, fib(1)=1.
         Use iterative approach to avoid exponential recursion.
         </think>
         def fib(n):
             if n <= 0: return 0
             a, b = 0, 1
             for _ in range(n):
                 a, b = b, a + b
             return a"
```

During inference, you have two options:
1. **Keep the thinking:** Show the full output including `<think>...</think>` so the
   user can see the model's reasoning.
2. **Strip it:** Remove everything between `<think>` and `</think>` and just return
   the code.

Our code provides helpers for both (`src/codeformer/reasoning/thinking_tokens.py`):

```python
from codeformer.reasoning.thinking_tokens import strip_thinking, extract_thinking

output = model.generate("def fib(n): ...")

# Option 1: get just the code
code = strip_thinking(output)

# Option 2: get both parts
thinking, code = extract_thinking(output)
print(f"Reasoning: {thinking}")
print(f"Code: {code}")
```

---

## 4. Adding Thinking Tokens to an Existing Model

### Expanding the vocabulary of a trained model

When you train a base model (tiny, small, medium), it has a vocabulary of 32,768
tokens. The thinking tokens are not in that vocabulary. To add them, we need to:

1. **Add the tokens to the tokenizer** (so it knows `<think>` is a single token, not
   five separate characters).
2. **Resize the model's embedding layer** (add 2 new rows to the embedding matrix).
3. **Resize the output projection** (add 2 new columns to the final linear layer).

```python
from codeformer.reasoning.thinking_tokens import add_thinking_tokens

# tokenizer and model are already loaded from a trained checkpoint
think_open_id, think_close_id = add_thinking_tokens(tokenizer, model)
# Vocab size: 32768 -> 32770
# Embedding shape: (32768, dim) -> (32770, dim)
# Output shape: (dim, 32768) -> (dim, 32770)
```

The new embedding rows are initialized with small random values (mean=0, std=0.02).
They will be trained during the fine-tuning phase. The existing embeddings are not
modified -- the model retains all its pre-trained knowledge.

Because we use **weight tying** (the embedding and output layers share the same weight
matrix), resizing one automatically resizes the other. This is handled in the
`_resize_embeddings` function.

---

## 5. CoT Fine-Tuning

### Training the model on reasoning examples

CoT fine-tuning is straightforward supervised training, identical to pre-training
except:
- The data includes `<think>...</think>` sections before the code.
- The learning rate is much lower (1e-5 vs 6e-4) because we are fine-tuning, not
  training from scratch. We want to add the reasoning capability without destroying
  what the model already knows.
- We train for fewer steps (5,000 vs 100,000).

The training data lives in `src/codeformer/reasoning/cot_data.py`. We provide a set
of hand-crafted reasoning traces for coding problems:

```python
COT_EXAMPLES = [
    CoTExample(
        task_id="has_close_elements",
        prompt='def has_close_elements(numbers: list[float], ...',
        thinking="""Let me think through this step by step:
1. I need to check if ANY two numbers are closer than the threshold
2. "Closer" means absolute difference < threshold
3. Compare every pair with a nested loop
...""",
        solution='def has_close_elements(numbers: list[float], ...',
    ),
    # ... more examples
]
```

Each example gets formatted as: `prompt + <think>reasoning</think>\nsolution`

The model learns: "when I see a coding problem, I should generate `<think>`, then
reasoning, then `</think>`, then the actual code."

### The reasoning config

```yaml
# configs/reasoning.yaml
model:
  vocab_size: 32768  # Will be expanded by +2 for <think> and </think>
  max_seq_len: 4096  # Longer context for reasoning traces

training:
  learning_rate: 1.0e-5   # 60x lower than pretraining (fine-tuning!)
  min_lr: 1.0e-6
  warmup_steps: 200
  max_steps: 5000
  weight_decay: 0.01       # Less regularization for fine-tuning
  grad_clip: 0.5           # Tighter clipping for stability
```

Key differences from pre-training:
- `learning_rate: 1.0e-5` -- very small steps to preserve existing knowledge.
- `max_seq_len: 4096` -- doubled to fit the thinking trace + code.
- `max_steps: 5000` -- much fewer steps since we start from a good model.
- `grad_clip: 0.5` -- tighter clipping because fine-tuning gradients can be noisy.

---

## 6. Introduction to RLVR

### Reinforcement Learning with Verifiable Rewards

CoT fine-tuning teaches the model the **format** of reasoning -- it learns to generate
`<think>...</think>` before code. But it does not optimize for the **quality** of the
reasoning. The model might generate plausible-sounding reasoning that leads to wrong
code.

This is where reinforcement learning (RL) comes in. RL lets the model learn from
trial and error: generate solutions, see which ones work, and reinforce the good ones.

For most RL applications (like chatbots), you need a separate **reward model** trained
on human preferences. This is expensive and complex. But for code, we have something
much better:

**We can just run the code and see if it passes tests.**

This is called **RLVR -- Reinforcement Learning with Verifiable Rewards**. The reward
is binary and objective:

```
Generated code passes all tests  ->  reward = 1.0  (good, reinforce this)
Generated code fails any test    ->  reward = 0.0  (bad, discourage this)
```

No reward model needed. No subjective judgments. No human labelers. Just: does the
code work?

This is the same insight behind DeepSeek-R1's success on code tasks. Code is one of
the best domains for RL because the reward signal is clean and automatic.

```
The RL loop for code:

  +---> Generate code solution
  |            |
  |            v
  |     Run against test cases
  |            |
  |      pass? | fail?
  |       |        |
  |       v        v
  |   reward=1  reward=0
  |       |        |
  |       +---+----+
  |           |
  |           v
  +---- Update model (reinforce good solutions,
        discourage bad ones)
```

---

## 7. GRPO Explained

### Generate multiple answers, see which work, reinforce the good ones

**GRPO** (Group Relative Policy Optimization) is the RL algorithm we use. It is a
simplified version of PPO (Proximal Policy Optimization) that does not need a
separate critic or value model.

Here is GRPO step by step, for a single coding problem:

```
Step 1: GENERATE a group of G solutions (G=8 in our config)
  Solution 1: <think>Use recursion...</think> def fib(n): ...   [wrong]
  Solution 2: <think>Iterative approach...</think> def fib(n): ... [correct]
  Solution 3: <think>Formula...</think> def fib(n): ...            [wrong]
  Solution 4: <think>Dynamic programming...</think> def fib(n): ... [correct]
  Solution 5: <think>Simple loop...</think> def fib(n): ...        [correct]
  Solution 6: <think>Use cache...</think> def fib(n): ...          [wrong]
  Solution 7: <think>Bottom up...</think> def fib(n): ...          [correct]
  Solution 8: <think>Math formula...</think> def fib(n): ...       [wrong]

Step 2: SCORE each solution by running tests
  Rewards: [0, 1, 0, 0, 1, 0, 1, 0]  (4 correct out of 8)

Step 3: COMPUTE ADVANTAGES relative to group mean
  Mean reward = 4/8 = 0.5
  Advantages = [-0.5, +0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5]
  (normalized by standard deviation)

  Positive advantage: "this solution was BETTER than average in the group"
  Negative advantage: "this solution was WORSE than average in the group"

Step 4: UPDATE the model
  - Solutions 2, 4, 5, 7 (correct): increase their probability
  - Solutions 1, 3, 6, 8 (wrong): decrease their probability
```

The key idea: we do not need to know what a "good" solution looks like in absolute
terms. We just need to know which solutions in the group were better than others.
The group provides its own baseline.

For a TS dev: think of GRPO like A/B testing with 8 variants. Generate several
solutions, see which ones pass the tests, and adjust the model to produce more
solutions like the winners.

### The GRPO training loop

```python
# Simplified from src/codeformer/reasoning/grpo.py
for problem in problems:
    # 1. Generate G solutions
    generations = []
    for _ in range(group_size):
        output = generator.generate(prompt=problem["prompt"])
        generations.append(output)

    # 2. Score with reward function (run code, check tests)
    rewards, infos = compute_batch_rewards(generations, problem["test_code"])

    # 3. Compute advantages
    rewards_tensor = torch.tensor(rewards)
    mean_reward = rewards_tensor.mean()
    std_reward = rewards_tensor.std() + 1e-8
    advantages = (rewards_tensor - mean_reward) / std_reward

    # 4. Policy gradient update
    for i in range(group_size):
        # Get current model's probability of generating this solution
        logits = model(tokenize(generations[i]))
        current_log_prob = compute_log_prob(logits, generations[i])

        # Compute probability ratio (new policy / old policy)
        ratio = exp(current_log_prob - old_log_probs[i])

        # PPO-style clipped objective
        unclipped = ratio * advantages[i]
        clipped = clamp(ratio, 1-eps, 1+eps) * advantages[i]
        loss = -min(unclipped, clipped)

    loss.backward()
    optimizer.step()
```

---

## 8. GRPO vs PPO

### Simpler -- no critic model needed

PPO (Proximal Policy Optimization) is the standard RL algorithm used to train models
like ChatGPT. It works, but it requires a **critic model** -- a second neural network
that estimates how good each state is. This doubles the memory usage and adds
significant complexity.

GRPO eliminates the critic. Instead of learning a value function, it uses the group
of generated solutions as its own baseline.

```
PPO architecture:
  Actor (the language model)  +  Critic (separate model, same size)
  Memory: 2x model size
  Complexity: need to train and synchronize two models

GRPO architecture:
  Actor (the language model)  only
  Memory: 1x model size
  Complexity: just generate, score, and update
```

Detailed comparison:

| Feature              | PPO                         | GRPO                       |
|----------------------|-----------------------------|----------------------------|
| Critic model         | Required (extra model)      | Not needed                 |
| Baseline             | Learned value function      | Group mean of rewards      |
| Memory usage         | 2x (actor + critic)         | 1x (actor only)            |
| KL penalty           | Against reference model     | Optional (off in our code) |
| Complexity           | High                        | Medium                     |
| Training stability   | Can be finicky              | Generally more stable      |
| Works for code?      | Yes                         | Yes, often better          |

For our use case (small models, verifiable rewards, limited VRAM), GRPO is the clear
choice. PPO's complexity pays off mainly for larger-scale RLHF with subjective
rewards, which is not what we are doing.

---

## 9. The Reward Function

### Run the code, did tests pass?

The reward function is what tells the RL algorithm whether a generated solution is
good or bad. Ours lives in `src/codeformer/reasoning/reward.py` and has three
components:

```python
def compute_reward(generated_text, test_code, max_thinking_tokens=512):
    reward = 0.0

    # Extract the thinking and code parts
    thinking, code = extract_thinking(generated_text)

    # 1. CORRECTNESS (the main signal): +1.0 if all tests pass
    full_code = code + "\n\n" + test_code
    success, output = execute_code(full_code, timeout=10.0)
    if success:
        reward += 1.0

    # 2. FORMAT BONUS: +0.1 if proper <think>...</think> structure
    if has_proper_thinking_format(generated_text):
        reward += 0.1

    # 3. LENGTH PENALTY: -0.1 if thinking trace is excessively long
    if word_count(thinking) > max_thinking_tokens:
        reward -= 0.1

    return reward
```

The correctness reward (+1.0) dominates everything else. Format and length bonuses
are minor nudges to encourage good habits. The model quickly learns that the only
thing that really matters is producing correct code.

**Why binary rewards work:** You might think a binary signal (0 or 1) is too coarse.
But combined with GRPO's group comparison, it provides a strong gradient. If 3 out of
8 solutions pass, the model gets a clear signal about which approaches work and which
do not. Over many problems, this adds up to a rich learning signal.

**Safety note:** The reward function executes generated code in a subprocess with a
timeout. This is inherently risky. In a production setting, you would want sandboxing
(Docker, gVisor, etc.). For local experiments with coding problems, the timeout is
sufficient.

---

## 10. Advantages Explained

### "Was this solution better or worse than average in the group?"

The **advantage** is the key quantity in GRPO. It tells the optimizer how to weight
each solution in the update.

```
Given rewards: [0, 1, 0, 0, 1, 0, 1, 0]

Mean = 0.375
Std  = 0.518

Advantages (normalized):
  Solution 1 (wrong):   (0 - 0.375) / 0.518 = -0.72   "worse than average"
  Solution 2 (correct): (1 - 0.375) / 0.518 = +1.21   "better than average"
  Solution 3 (wrong):   (0 - 0.375) / 0.518 = -0.72   "worse than average"
  ...
  Solution 5 (correct): (1 - 0.375) / 0.518 = +1.21   "better than average"
  ...
```

The optimizer then:
- **Increases** the probability of solutions with positive advantages.
- **Decreases** the probability of solutions with negative advantages.
- The magnitude of the advantage controls how much the probability changes.

Why normalize by standard deviation? If 7 out of 8 solutions are correct, the
advantages would be tiny without normalization (all close to the mean). Normalizing
ensures the model still gets a meaningful gradient even when most solutions are correct
or most are wrong.

**Edge case:** If ALL solutions in a group are correct (rewards = [1,1,1,1,1,1,1,1]),
all advantages are 0. The model learns nothing from this problem -- it already solves
it consistently. Similarly, if ALL solutions fail, advantages are all 0. The model
needs at least some variance in the group to learn.

This is why **group size matters**. With G=2, you get very noisy estimates. With G=8
(our default), you get a reasonable signal. With G=16 or G=32, estimates are smoother
but generation takes longer.

---

## 11. Running the Reasoning Experiments

### Prerequisites

You need a trained base model to start from. The reasoning experiments fine-tune an
existing model -- they do not train from scratch.

```bash
# Make sure you have a trained small model
ls checkpoints/small/latest
# Should show a path to a checkpoint directory
```

### Step 1: Generate CoT training data

```bash
python -c "
from codeformer.reasoning.cot_data import save_cot_data
save_cot_data('./data/reasoning')
"
# Creates data/reasoning/cot_0000.txt, cot_0001.txt, etc.
```

This generates training examples from the hand-crafted reasoning traces in
`cot_data.py`. For better results, you can add more examples to the `COT_EXAMPLES`
list.

### Step 2: Run CoT fine-tuning

```bash
python scripts/train_reasoning.py \
    --config configs/reasoning.yaml \
    --base-checkpoint ./checkpoints/small/latest
```

This will:
1. Load your trained small model (125M params).
2. Add `<think>` and `</think>` tokens (vocab: 32768 -> 32770).
3. Fine-tune on the CoT examples for 5,000 steps.
4. Save checkpoints to `./checkpoints/reasoning/`.

### Step 3: Run GRPO training

```bash
python scripts/train_reasoning.py \
    --config configs/reasoning.yaml \
    --base-checkpoint ./checkpoints/reasoning/latest \
    --grpo \
    --epochs 3 \
    --group-size 8
```

This will:
1. Load the CoT-fine-tuned model.
2. For each coding problem, generate 8 solutions.
3. Run each solution against test cases.
4. Update the model with GRPO.
5. Repeat for 3 epochs over all problems.

### Step 4: Evaluate

```bash
# Run HumanEval benchmark
python scripts/evaluate.py \
    --checkpoint ./checkpoints/reasoning/latest \
    --thinking  # Enable thinking mode
```

### What the training output looks like

```
Starting GRPO training:
  Problems: 164
  Group size: 8
  Epochs: 3

Epoch 1/3:
  problem 1/164: loss=0.0234 mean_reward=0.250 pass_rate=25.0%
  problem 2/164: loss=0.0189 mean_reward=0.375 pass_rate=37.5%
  ...
  Epoch 1: loss=0.0201, mean_reward=0.312, pass_rate=31.2%

Epoch 2/3:
  ...
  Epoch 2: loss=0.0156, mean_reward=0.418, pass_rate=41.8%

Epoch 3/3:
  ...
  Epoch 3: loss=0.0134, mean_reward=0.489, pass_rate=48.9%
```

You should see `pass_rate` increase across epochs. If it plateaus after epoch 1, the
model may have hit its capacity ceiling for this problem set.

### Monitoring

```bash
# Add wandb logging
python scripts/train_reasoning.py \
    --config configs/reasoning.yaml \
    --base-checkpoint ./checkpoints/small/latest \
    --grpo --wandb
```

Key metrics to watch:
- **pass_rate:** The percentage of generated solutions that pass tests. This is the
  primary success metric.
- **mean_reward:** Average reward across all solutions. Should increase.
- **loss:** The GRPO policy loss. Should decrease, but can be noisy.

---

## 12. What Results to Expect

Results depend heavily on model size. Bigger models have more capacity to learn
reasoning patterns.

### Expected pass@1 on HumanEval

| Model Size   | Base Model | After CoT | After GRPO | Notes                      |
|-------------|------------|-----------|------------|----------------------------|
| 50M (tiny)  | ~5%        | ~6-8%     | ~8-10%     | Very limited capacity      |
| 125M (small)| ~15%       | ~17-20%   | ~20-25%    | Meaningful improvement     |
| 350M (medium)| ~25%      | ~28-32%   | ~35-40%    | Significant gains          |
| 1B+ (large) | ~40%       | ~43-48%   | ~50-55%    | Best results               |

These are rough estimates based on what is achievable at each scale. Your actual
results will depend on:
- Quality and quantity of training data.
- Number of GRPO epochs and problems.
- Hyperparameter tuning (learning rate, group size, etc.).

### Why small models plateau quickly

A 50M parameter model has limited capacity. It can learn the format of reasoning
(`<think>...</think>`) but cannot learn deep algorithmic reasoning. The thinking
trace might say "use dynamic programming" but the model still fails to implement it
correctly.

At 125M, the model can learn some algorithmic patterns. At 350M+, it starts to
generalize reasoning across problem types. This is why we recommend training at
least the small (125M) model for reasoning experiments.

### Timeline for results

```
CoT fine-tuning (5,000 steps):   ~1-2 hours on RTX 4080 (small model)
GRPO (3 epochs, 164 problems):   ~4-8 hours on RTX 4080 (small model)
                                  (mostly spent on generation, not training)
```

GRPO is slower than regular training because it generates `group_size` full
solutions per problem per epoch. With G=8 and 164 problems over 3 epochs, that is
8 * 164 * 3 = 3,936 full code generations. Generation is sequential and slower than
training.

---

## 13. Ideas for Further Experimentation

### 1. Increase group size

More solutions per problem means better advantage estimates. Try G=16 or G=32.
The generation happens sequentially, so this mainly costs time, not VRAM.

```yaml
# configs/reasoning.yaml
reasoning:
  group_size: 16  # Default is 8
```

### 2. Curriculum learning

Start GRPO with easy problems, gradually increase difficulty. Easy problems give
more positive reward signal early on, which helps the model build confidence before
tackling harder problems.

```python
# Sort problems by difficulty (e.g., based on base model pass rate)
problems_sorted = sorted(problems, key=lambda p: p["difficulty"])
for epoch in range(num_epochs):
    # Increase the proportion of hard problems each epoch
    cutoff = len(problems_sorted) * min(1.0, 0.5 + 0.2 * epoch)
    epoch_problems = problems_sorted[:int(cutoff)]
    trainer.train(epoch_problems)
```

### 3. Self-play data generation

After a few GRPO epochs, use the improved model to generate new CoT training data.
Then fine-tune on this new, higher-quality data. Repeat. This creates a virtuous
cycle similar to what DeepSeek-R1 describes.

```python
# After GRPO training
from codeformer.reasoning.cot_data import generate_cot_from_solutions

# Generate solutions with the improved model
solutions = [generator.generate(p["prompt"]) for p in problems]

# Keep only the correct ones
correct = [(p, s) for p, s in zip(problems, solutions)
           if passes_tests(s, p["test_code"])]

# Create new CoT training data from successful solutions
new_examples = generate_cot_from_solutions(
    [c[0] for c in correct],
    [c[1] for c in correct],
)
```

### 4. Multi-turn reasoning

Instead of a single `<think>` block, try interleaved thinking and coding:

```
<think>First, handle the edge case of empty input...</think>
if not nums:
    return 0
<think>Now iterate through the list tracking the maximum...</think>
max_val = nums[0]
for n in nums[1:]:
    max_val = max(max_val, n)
<think>Return the result...</think>
return max_val
```

This requires modifying the generation logic to allow multiple think/code segments.

### 5. Partial credit rewards

Instead of binary (pass/fail), award partial credit:

```python
def compute_reward_with_partial_credit(code, test_cases):
    passed = 0
    for test in test_cases:
        if run_test(code, test):
            passed += 1
    return passed / len(test_cases)  # 0.0 to 1.0
```

This gives a smoother reward signal. A solution that passes 3 out of 5 tests gets
0.6 instead of 0.0. The model can learn from partial successes.

### 6. Test-time compute scaling

At inference time, generate many solutions and pick the best one. This is not RL --
it is a simple technique that works surprisingly well:

```python
# Generate 100 solutions
solutions = [generator.generate(prompt, temperature=0.8) for _ in range(100)]

# Run each against tests
results = [(s, passes_tests(s, test_code)) for s in solutions]

# Pick one that passes (if any)
correct_solutions = [s for s, passed in results if passed]
if correct_solutions:
    return correct_solutions[0]
```

Even without RL, generating 100 solutions and filtering dramatically boosts the
effective pass rate. This is called **pass@k** evaluation: what fraction of problems
can the model solve if it gets k attempts?

### 7. Different reward signals

The reward function is modular. You can experiment with alternatives:

```python
# Efficiency reward: bonus for lower time complexity
reward += 0.1 if solution_is_O_n(code) else 0.0

# Style reward: bonus for clean, readable code
reward += 0.05 if has_docstring(code) else 0.0

# Brevity reward: prefer shorter solutions (within reason)
reward += 0.05 if len(code) < median_solution_length else 0.0
```

These additional signals can shape the model's behavior beyond just correctness.
Keep the correctness reward dominant (1.0) and use small bonuses (0.05-0.1) for
secondary objectives.
