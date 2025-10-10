"""Check anchor extraction."""

from agentic_rag.anchors.validators import required_anchors

questions = [
    "What is a movie to feature a person who can create and control a device that can manipulate the laws of physics?",
    "What is the difference between the 1989 and 2017 versions of the Hugo Award for Best Novel?",
    "What are the top 3 grossing Pixar films?",
]

for q in questions:
    anchors = required_anchors(q)
    print(f"Q: {q}")
    print(f"A: {anchors}\n")
