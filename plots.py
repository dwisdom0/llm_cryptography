import plotly.express as px
import plotly.graph_objects as go


data = {
    'method': ['AES-CTR', 'LLM'],
    'decrypt_time_s': [0.0000348, 0.9149070],
}

fig = px.bar(
    data,
    x='method',
    y='decrypt_time_s',
    log_y=True
)

x_start = [0]
x_end = [1]
y_start = [1]
y_end = [0]

arrow = go.layout.Annotation(dict(
    x=0.5,
    y=-1,
    ax=0,
    ay=-4,
    xref="x",
    yref="y",
    axref="x",
    ayref="y",
    showarrow=True,
    arrowhead=4,
    arrowwidth=7,
    arrowcolor='#ff0000',
))
text_annotation = go.layout.Annotation(dict(
    x=0.25,
    y=-2.1,
    xref='x',
    yref='y',
    xanchor='right',
    yanchor='bottom',
    showarrow=False,
    text='100,000x',
    font=dict(
        color='#ff0000',
        size=50,
        weight=800,
    )
))

fig.update_layout(annotations=[arrow, text_annotation])
fig.show()