src/
│
├── data/
│   ├── swiss_roll.npz
│   ├── gaussians.npz
│   ├── circles.npz
│
├── dataloader.py        ← ❗老师给你的（不要乱改）
├── generate_data.py     ← ❗老师给的（不用动）
│
├── models/
│   ├── flow_mlp.py      ← model（必须复用）
│
├── training/
│   ├── train.py         ← 通用训练逻辑（Part1/2/3都用）
│   ├── losses.py
│
├── sampling/
│   ├── euler.py         ← 统一采样
│
├── experiments/
│   ├── part1_run.py     ← ❗只负责“跑Part1”
│   ├── part2_grid.py
│   ├── part3_rescue.py
│   ├── part4_meanflow.py
│
├── utils/
│   ├── plot.py
│   ├── logger.py
│
└── outputs/

• Part 1: Verify a basic flow matching setup on 2D data.
• Part 2: Systematically compare prediction parameterizations and reproduce the paper’s
central result.
• Part 3: Investigate whether v-prediction’s failure at high dimensions can be overcome.
• Part 4: Implement MeanFlow for single-step generation.
