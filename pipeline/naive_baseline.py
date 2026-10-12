from pipeline import llm, store

PROMPT = """You are a helpful tax compliance assistant for Pakistan.
Answer using only the context below and cite section numbers. If the context is not enough, say so.

Context:
{context}

Question: {question}"""


def answer(question, as_of=None):
    docs = store.search("", llm.embed([question])[0], as_of, k=4)
    context = "\n\n".join(f"({d['section']})\n{d['content']}" for d in docs)
    return {"docs": docs, "response": {"answer": llm.ask(PROMPT.format(context=context, question=question), model=llm.SMART)}}


if __name__ == "__main__":
    print(answer("What is the tax rate on IT export services and freelancers?")["response"]["answer"])
 