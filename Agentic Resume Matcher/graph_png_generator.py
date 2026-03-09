from graphs.ingestion_graph import ingestion_graph
from graphs.query_graph import query_graph
from IPython.display import Image, display

ingestion_graph_data = ingestion_graph.get_graph().draw_mermaid_png()
query_graph_data = query_graph.get_graph().draw_mermaid_png()

def save_png(data, filename):
    with open(filename, "wb") as f:
        f.write(data)

save_png(ingestion_graph_data, "gingestion_graph_data_visualization.png")
save_png(query_graph_data, "query_graph_data_visualization.png")

display(Image(ingestion_graph_data))
display(Image(query_graph_data))

