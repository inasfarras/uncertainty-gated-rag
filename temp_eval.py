from eval.gpt_agent_judge import judge_example

question = "which movie won the oscar best visual effects in 2021?"
gold = "Tenet"
answer = "The movie that won the Oscar for Best Visual Effects in 2021 is \"Tenet\" [CIT:129da772-7047-418c-9871-98801c6dbee8_3]."
passages = {
    "129da772-7047-418c-9871-98801c6dbee8_3": "Tenet won the Oscar for Best Visual Effects in 2021."
}
print(judge_example(question, gold, answer, passages))
