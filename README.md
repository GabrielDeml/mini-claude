# mini-claude

Training a small transformer language model from scratch in plain PyTorch, on Google Colab GPUs.

Start with the design doc: [docs/DESIGN.md](docs/DESIGN.md). It explains how LM training works, the staged plan (char-level Shakespeare → 30M-parameter TinyStories model → tiny instruction-tuned chat model), the architecture, the Colab workflow, testing, and milestones.

Status: design phase, no code yet.
