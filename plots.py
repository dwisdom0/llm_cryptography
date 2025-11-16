from datetime import datetime
from csv import DictReader
import plotly.express as px
import plotly.graph_objects as go

# log bar plot of decryption time

THEME = 'plotly_white'

data = {
    'method': ['AES-CTR', 'LLM'],
    'decrypt_time_s': [0.0000041, 0.3587279],
}

fig = px.bar(
    data,
    x='method',
    y='decrypt_time_s',
    title='The LLM cipher takes about 85,000 times longer to decrypt',
    log_y=True,
    template=THEME,
)
fig.update_layout(dict(
    xaxis=dict(
        title=dict(
            text=""
        )
    ),
    yaxis=dict(
        title=dict(
            text="Seconds to decrypt (log scale)"
        )
    ),

)

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
    text='85,000x',
    font=dict(
        color='#ff0000',
        size=20,
        weight=700,
    )
))

fig.update_layout(annotations=[arrow, text_annotation])
fig.show()
fig.write_html('log_bar.html', include_plotlyjs=False)


# training loss curve

records = []
with open('loss.csv', 'r') as f:
    reader = DictReader(f)
    for r in reader:
        records.append({
            'timestamp': datetime.fromtimestamp(int(r['timestamp'])//1000),
            'step': int(r['step']),
            'training_loss': float(r['value']),
        })

cols = {'timestamp': [], 'step': [], 'training_loss': []}
for r in records:
    for c in cols.keys():
        cols[c].append(r[c])

fig = px.line(
    cols,
    x='step',
    y='training_loss',
    markers=True,
    template=THEME,
    title="Training loss"
)
fig.update_layout(dict(
    xaxis=dict(
        title=dict(
            text='Step'
        )
    ),
    yaxis=dict(
        title=dict(
            text='Training loss'
        )
    )
))

fig.show()
fig.write_html('training_loss.html', include_plotlyjs='cdn')