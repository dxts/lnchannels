from graph import Graph, Node
import math
import plotly.graph_objects as go


def draw_graph(g: Graph):
    edge_x = []
    edge_y = []

    node_x = []
    node_y = []

    node_adjacencies = []
    node_text = []

    for node in g.nodes:
        # node position is (id, id)
        nx, ny = math.cos(node.id), math.sin(node.id)

        node_x.append(nx)
        node_y.append(ny)

        node_adjacencies.append(len(node.edges))
        node_text.append('#{:d}'.format(node.id))

        for edge in node.edges:
            target = g.nodes[edge.to_node.id]
            # edge is built between src and trg node
            x1, y1 = math.cos(target.id), math.sin(target.id)
            x0, y0 = nx, ny

            edge_x.extend((x0, x1, None))
            edge_y.extend((y0, y1, None))

    # build edge graph
    edge_trace = go.Scatter(x=edge_x, y=edge_y,
                            line=dict(width=0.5, color='#888'),
                            hoverinfo='none',
                            mode='lines')

    # build node graph
    node_trace = go.Scatter(x=node_x, y=node_y,
                            mode='markers',
                            hoverinfo='text',
                            text=node_text,
                            marker=dict(showscale=True,
                                        colorscale='Bluered',
                                        reversescale=True,
                                        color=node_adjacencies,
                                        size=node_adjacencies,
                                        colorbar=dict(
                                            thickness=15,
                                            title='node degree',
                                            xanchor='left',
                                            titleside='right'
                                        ),
                                        line_width=2))

    # create network graph

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
        title='Barabási–Albert random graph',
        titlefont_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    )

    fig.show()
