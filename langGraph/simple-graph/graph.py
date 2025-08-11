from typing import TypedDict, Literal
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
import random

def decide_mood(state) -> Literal["node_2", "node_3"]:
    # Often, we will use state to decide on the next node to visit
    user_input = state['graph_state'] 
    # Here, let's just do a 50 / 50 split between nodes 2, 3
    if random.random() < 0.5:
        # 50% of the time, we return Node 2
        return "node_2"
    # 50% of the time, we return Node 3
    return "node_3"

class State(TypedDict):
    graph_state: str

def node_1(state):
    print('---Node 1---')
    return {"graph_state": state['graph_state'] + " I am"}

def node_2(state):
    print('---Node 2---')
    return {"graph_state": state['graph_state'] + " happy!"}

def node_3(state):
    print('---Node 3---')
    return {"graph_state": state['graph_state'] + " sad!"}

# Build graph
graphBuilder = StateGraph(State)
graphBuilder.add_node("node_1", node_1)
graphBuilder.add_node("node_2", node_2)
graphBuilder.add_node("node_3", node_3)

#Logic
graphBuilder.add_edge(START, "node_1")
graphBuilder.add_conditional_edges("node_1", decide_mood)
graphBuilder.add_edge("node_2", END)
graphBuilder.add_edge("node_3", END)

# Construct the graph
graph = graphBuilder.compile()

updated_state = graph.invoke({"graph_state" : "Hi, this is Lance."})
print(updated_state)

# View the graph
with open("graph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())