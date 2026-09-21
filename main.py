from langgraph.graph import StateGraph, MessagesState, START, END


def echo(state: MessagesState):
    return {"messages": [{"role": "ai", "content": "hello world"}]}


builder = StateGraph(MessagesState)
builder.add_node(echo)
builder.add_edge(START, "echo")
builder.add_edge("echo", END)
graph = builder.compile()

result = graph.invoke({"messages": [{"role": "user", "content": "hii"}]})

for m in result["messages"]:
    print(f"{m.type}:{m.content}")
